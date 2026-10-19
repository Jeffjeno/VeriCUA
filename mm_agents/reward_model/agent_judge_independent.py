import asyncio
from typing import Any, Dict, List, Tuple

from mm_agents.reward_model.agent_judge import AgentJudgeEvaluator
from mm_agents.reward_model.evaluation_result import EvaluationResult
from mm_agents.reward_model.task_sample import TaskSample


class IndependentRolloutAgentJudgeEvaluator(AgentJudgeEvaluator):
    """AgentJudge variant where each rollout rebuilds memory independently."""

    def _build_rollout_result_extra(
        self,
        meta: Dict[str, Any],
        history_len: int,
        rollout_idx: int,
        rollout_num: int,
        judge_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra = super()._build_rollout_result_extra(
            meta=meta,
            history_len=history_len,
            rollout_idx=rollout_idx,
            rollout_num=rollout_num,
            judge_output=judge_output,
        )
        extra["history_n"] = self.history_n
        return extra

    def _evaluate_sync(self, task: TaskSample) -> List[EvaluationResult]:
        return asyncio.run(self._evaluate_async(task))

    async def _evaluate_async(self, task: TaskSample) -> List[EvaluationResult]:
        history = self._build_history(task)
        if not history:
            return [self._error(task, "empty_history")]

        rollout_results = await self._run_agent_judge(task.instruction, history)
        rollout_num = len(rollout_results)
        return [
            self._result(
                task,
                llm_judge=item["decision"],
                content=item["content"],
                reasoning="",
                **self._build_rollout_result_extra(
                    meta=item["meta"],
                    history_len=len(history),
                    rollout_idx=rollout_idx,
                    rollout_num=rollout_num,
                    judge_output=item,
                ),
            )
            for rollout_idx, item in enumerate(rollout_results)
        ]

    def _init_rollout_meta(self) -> Dict[str, Any]:
        return {
            "mr_rounds": 0,
            "mr_calls_gen_mem": 0,
            "mr_calls_judge": 0,
            "mr_trace": [],
            "judge_msg": [],
            "memory_msgs": [],
            "decision_raw": None,
        }

    async def _run_agent_judge(
        self, instruction: str, history: List[Tuple[Dict[str, Any], Any]]
    ) -> List[Dict[str, Any]]:
        chunks = self._chunk_history(history)
        rollout_num = int(self.rollout_num)
        return await asyncio.gather(
            *[
                self._run_single_rollout(
                    instruction=instruction,
                    history=history,
                    chunks=chunks,
                )
                for _ in range(rollout_num)
            ]
        )

    async def _run_single_rollout(
        self,
        instruction: str,
        history: List[Tuple[Dict[str, Any], Any]],
        chunks: List[List[Tuple[Dict[str, Any], Any]]],
    ) -> Dict[str, Any]:
        meta = self._init_rollout_meta()
        previous_memory, history_records, meta = await self._run_memory_phase(
            instruction=instruction,
            chunks=chunks,
            meta=meta,
        )
        judge_outputs = await self._run_judge_phase(
            instruction=instruction,
            history=history,
            history_records=history_records,
            memory=previous_memory,
            meta=meta,
            rollout_num=1,
        )
        judge_output = dict(judge_outputs[0])
        judge_output["meta"] = meta
        return judge_output
