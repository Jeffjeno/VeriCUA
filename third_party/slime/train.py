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
    configure_logger()
                       
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):

        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
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

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        rollout_host_v2 = _rollout_host_v2_enabled(args)
        try:
            if rollout_host_v2:
                ray.get(rollout_manager.begin_update.remote(rollout_id))
            if args.use_critic:
                value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
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
                save(rollout_id)

        offload_train(actor_trains_this_step)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)

if __name__ == "__main__":
    args = parse_args()
    train(args)
