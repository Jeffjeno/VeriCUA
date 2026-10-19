from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskSample(BaseModel):
    """
    评估流水线的标准输入实体。

    所有数据集适配器都应当产出该结构；评估脚本只依赖该结构进行 prompt 构造与调用，
    从而保证不同数据集/来源的数据可以用统一的接口进入评估与结果落盘流程。

    字段说明（核心）
    - task_id：任务 ID（建议在同一数据集/拆分内唯一）
    - instruction：任务指令文本（面向模型的自然语言任务描述）
    - policy_model：生成该轨迹/样本的策略模型名（用于区分数据来源，并参与 safe_id 生成）
    - ground_truth：GT 标签（可选；True/False/None 三态）
    - history_images：历史截图序列（本地相对路径或 URL）
    - history_actions：历史动作序列（字符串列表）
    - current_image：最终/当前截图（通常是重点观测的内容，但大多数情况需要结合历史截图才能准确判定）
    - last_action：最后一步动作（任务成功时，通常为表示finish或answer状态的action；如轨迹中未记录最终截图后的action则填写"None"）
    - images_stored_in_minio：图片字段是否已替换为可访问的 URL（例如 MinIO）
    - extra：数据集特有字段（字典；用于扩展而不破坏标准字段）
    - trajectory：遗留/全量轨迹字段（可选；用于兼容历史数据或存储更完整的 step 结构）

    图片路径约定
    当图片字段是本地路径且 images_stored_in_minio=False 时，路径必须相对于仓库根目录的 assets/。
    评估前通常会在数据集的 prepare 阶段将其转换为 data:image/png;base64,... 形式以便模型调用。

    safe_id
    safe_id 定义为：{policy_model}:{task_id}，用于评估结果落盘的文件名与断点续跑识别。
    """

    task_id: str = Field(..., description="任务 ID（建议在同一数据集/拆分内唯一）")
    instruction: str = Field(
        ..., description="任务指令文本（面向模型的自然语言任务描述）"
    )
    policy_model: str = Field(
        ..., description="生成该轨迹/样本的策略模型名（参与 safe_id 生成）"
    )

    ground_truth: Optional[bool] = Field(
        default=None,
        description="GT 标签（可选；True/False/None 三态；None 表示无标签）",
    )

    current_image: str = Field(
        ...,
        description="最终/当前截图（本地相对路径或 URL；通常是评估重点观测）",
    )
    history_images: List[str] = Field(
        ...,
        description="历史截图序列（本地相对路径或 URL）",
    )

    last_action: str = Field(
        ..., description="最后一步动作（任务成功时，通常为finish或answer）"
    )
    history_actions: List[str] = Field(
        ..., description="历史动作序列（字符串列表）"
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="数据集特有字段（字典；用于扩展而不破坏标准字段）",
    )

    trajectory: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="遗留/全量轨迹字段（可选；用于兼容历史数据或存储更完整的 step 结构）",
    )

    images_stored_in_minio: bool = Field(
        default=False,
        description="图片字段是否已替换为可访问的 URL（例如 MinIO）；False 表示仍可能是本地相对路径",
    )

    @property
    def safe_id(self) -> str:
        """生成可安全用于文件名的复合 ID：{policy_model}:{task_id}"""
        composite = f"{self.policy_model}:{self.task_id}"
        return composite.replace("/", "_").replace("\\", "_")

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)
