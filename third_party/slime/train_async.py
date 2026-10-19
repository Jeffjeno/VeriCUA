import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action

def _rollout_host_v2_enabled(args) -> bool:
    value = getattr(args, "guirl_rollout_host_v2", None)
    if value is None:
        value = getattr(args, "rollout_host_v2", False)
    return (isinstance(value, int) and value != 0) or (
        isinstance(value, str)
        and value.strip().lower() in {"1", "true", "yes", "on"}
    )

def _best_effort_mark_in_doubt(rollout_manager, rollout_id, error) -> None:
    try:
        ray.get(
            rollout_manager.mark_in_doubt.remote(
                rollout_id,
                f"{type(error).__name__}: {error}",
            )
        )
    except BaseException:
        pass

def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
                       
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    rollout_host_v2 = _rollout_host_v2_enabled(args)
    rollout_data_next_future = (
        None
        if rollout_host_v2
        else rollout_manager.generate.remote(args.start_rollout_id)
    )
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if rollout_host_v2:
            rollout_data_curr_ref = ray.get(
                rollout_manager.generate.remote(rollout_id)
            )
                                                          
        elif rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        if not rollout_host_v2 and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        try:
            if rollout_host_v2:
                ray.get(rollout_manager.begin_update.remote(rollout_id))
            if args.use_critic:
                value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            if rollout_host_v2:
                ray.get(
                    rollout_manager.optimizer_succeeded.remote(rollout_id)
                )
                checkpointed_models = []
                if actor_trains_this_step:
                    actor_model.save_model(rollout_id, force_sync=True)
                    checkpointed_models.append("actor")
                if args.use_critic:
                    critic_model.save_model(rollout_id, force_sync=True)
                    checkpointed_models.append("critic")
                if not checkpointed_models:
                    raise RuntimeError(
                        "v2 optimizer step has no model checkpoint boundary"
                    )
                ray.get(
                    rollout_manager.checkpoint_succeeded.remote(
                        rollout_id,
                        {
                            "rollout_id": int(rollout_id),
                            "models": checkpointed_models,
                            "force_sync": True,
                        },
                    )
                )
                ray.get(rollout_manager.commit.remote(rollout_id))
        except BaseException as exc:
            if rollout_host_v2:
                _best_effort_mark_in_doubt(rollout_manager, rollout_id, exc)
            raise

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if not rollout_host_v2:
                if actor_trains_this_step:
                    actor_model.save_model(
                        rollout_id,
                        force_sync=rollout_id == args.num_rollout - 1,
                    )
                if args.use_critic:
                    critic_model.save_model(
                        rollout_id,
                        force_sync=rollout_id == args.num_rollout - 1,
                    )
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
                                                                                                      
            if not rollout_host_v2:
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)

if __name__ == "__main__":
    args = parse_args()
    train(args)
