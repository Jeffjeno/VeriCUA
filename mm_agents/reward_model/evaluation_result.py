import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from mm_agents.reward_model.task_sample import TaskSample


class EvaluationResult(BaseModel):
    """
    评估流水线的标准输出实体。

    评估脚本在调用 LLM 后，会把判决结果、推理内容、与 GT 的一致性等信息保存为该结构，
    并由结果管理组件统一落盘与汇总分析。

    字段说明（核心）
    - task_sample：TaskSample 原始 Task 信息
    - task_id：任务 ID（来自 TaskSample.task_id）
    - reward_model：执行评估的模型名（当前批次的评估模型）
    - policy_model：轨迹来源模型名（来自 TaskSample.policy_model）
    - instruction：任务指令文本（来自 TaskSample.instruction）
    - llm_judge：LLM 判决（success/failure 映射为 bool）
    - ground_truth：GT 标签（来自 TaskSample.ground_truth）
    - status：correct/incorrect/unknown/error（用于反映 llm_judge 与 ground_truth 的一致性或异常状态）
    - content：LLM 原始输出（完整文本）
    - reasoning_content：从响应中提取的推理内容（来自 reasoning_content 字段或 <think>）
    - error_message：异常信息（仅当 status=error 时通常有值）
    - timestamp / duration：评估完成时间戳与耗时（秒）
    - extra：额外元数据（例如数据集标签、附加字段等）
    - safe_id 定义为：{policy_model}:{task_id}（来自 TaskSample.safe_id）
    """

    reward_model: str = Field(..., description="执行评估的模型名（当前批次的评估模型）")

    llm_judge: Optional[bool] = Field(
        ..., description="LLM 判决（success/failure 映射为 bool）"
    )
    status: str = Field(
        ...,
        description="评估状态：correct/incorrect/unknown/error",
    )

    content: str = Field(default="", description="LLM 原始输出（完整文本）")
    reasoning_content: str = Field(
        default="",
        description="从响应中提取的推理内容（来自 reasoning_content 字段或 <think>）",
    )
    error_message: Optional[str] = Field(default=None, description="异常信息（可选）")
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="额外元数据（字典）"
    )
    timestamp: float = Field(
        default_factory=time.time, description="评估完成的时间戳（Unix seconds）"
    )
    duration: float = Field(default=0.0, description="评估耗时（秒）")
    task_sample: TaskSample = Field(..., description="Task 原始信息")

    @property
    def task_id(self) -> str:
        return self.task_sample.task_id

    @property
    def policy_model(self) -> str:
        return self.task_sample.policy_model

    @property
    def instruction(self) -> str:
        return self.task_sample.instruction

    @property
    def ground_truth(self) -> Optional[bool]:
        return self.task_sample.ground_truth

    @property
    def safe_id(self) -> str:
        """生成可安全用于文件名的复合 ID：{policy_model}:{task_id}"""
        return self.task_sample.safe_id
