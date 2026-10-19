import asyncio
import base64
import json
import mimetypes
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from mm_agents.reward_model.evaluation_result import EvaluationResult
from mm_agents.reward_model.task_sample import TaskSample
from mm_agents.reward_model.client import Client


class BaseEvaluator(ABC):
    """LLM-as-a-Judge 评估器抽象基类。

    采用模板方法模式，将评估流程分解为四个阶段::

        prepare → validate → evaluate → finalize

    基类负责流程编排（计时、异常兜底、结果持久化），
    子类负责核心评估逻辑（提示词构造、LLM 调用、响应解析）。

    子类必须实现:
        - ``_evaluate_sync``: 同步评估逻辑。
        - ``_evaluate_async``: 异步评估逻辑。

    子类可选覆写:
        - ``_prepare``: 自定义任务预处理。
        - ``_validate``: 自定义前置校验。

    Attributes:
        llm: LLM 服务实例，提供同步与异步聊天补全接口。
        result_manager: 评估结果持久化管理器。
        model_name: 当前评估使用的模型标识，取自 ``llm.model``。

    Example::

        class MyEvaluator(BaseEvaluator):
            def _evaluate_sync(self, task):
                resp = self.llm.chat_completion_sync(messages)
                return self._result(task, llm_judge=True, content=resp["content"])

            async def _evaluate_async(self, task):
                resp = await self.llm.chat_completion_async(messages)
                return self._result(task, llm_judge=True, content=resp["content"])

        evaluator = MyEvaluator(llm, result_manager)
        result = evaluator.evaluate_sync(task)
    """

    def __init__(self, llm) -> None:
        """初始化评估器实例。

        Args:
            llm: LLM 服务实例，须提供 ``chat_completion_sync`` 与
                ``chat_completion_async`` 方法。
            result_manager: 评估结果持久化管理器，须提供 ``save_result``
                与 ``get_existing_ids`` 方法。
        """
        self.model_name = str(llm.model)
        self.llm = Client(llm.model, 
                          llm.base_url, 
                          llm.api_key,
                          temperature=llm.temperature, 
                          max_tokens=llm.max_tokens, 
                          top_p=llm.top_p if hasattr(llm, "top_p") else None,
                          top_k=llm.top_k if hasattr(llm, "top_k") else None,
                          min_p=llm.min_p if hasattr(llm, "min_p") else None,
                          presence_penalty=llm.presence_penalty if hasattr(llm, "presence_penalty") else None,
                          repetition_penalty=llm.repetition_penalty if hasattr(llm, "repetition_penalty") else None,
                          enable_thinking=llm.enable_thinking if hasattr(llm, "enable_thinking") else True,
                          request_timeout=llm.request_timeout if hasattr(llm, "request_timeout") else None,
                            )

    # ==========================================================
    # 对外入口（模板方法）
    # ==========================================================

    def evaluate_sync(self, task: TaskSample) -> EvaluationResult:
        """同步评估单个任务。

        依次执行 ``_prepare`` → ``_validate`` → ``_evaluate_sync`` → ``_finalize``。
        任何阶段抛出的未捕获异常将被转换为 ``status="error"`` 的结果。

        Args:
            task: 原始任务样本（未经预处理）。

        Returns:
            完整的评估结果，已包含计时信息并已持久化。
        """
        start = time.time()
        try:
            task = self._prepare(task)
            err = self._validate(task)
            result = err if err else self._evaluate_sync(task)
        except Exception as e:
            logger.exception(f"Task {task.safe_id} failed: {e}")
            result = self._error(task, str(e))
        return self._finalize(result, start)

    def save_reward_results(
        self, result: EvaluationResult, result_dir: str,
    ) -> str:
        """将 reward model 结果保存为 JSON 文件。

        Args:
            result: 已完成评估的结果对象。
            result_dir: 保存目录。

        Returns:
            保存后的 JSON 文件绝对路径。
        """
        result_path = Path(result_dir) / "reward_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        task_payload = payload.get("task_sample", {})
        task_extra = dict(task_payload.get("extra") or {})

        raw_current_image = task_extra.pop("_reward_input_current_image", None)
        raw_history_images = task_extra.pop("_reward_input_history_images", None)
        if raw_current_image is not None:
            task_payload["current_image"] = raw_current_image
        if raw_history_images is not None:
            task_payload["history_images"] = raw_history_images

        task_payload["extra"] = task_extra
        payload["task_sample"] = task_payload
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(result_path)

    async def evaluate_async(
        self, task: TaskSample, pool: ProcessPoolExecutor,
    ) -> EvaluationResult:
        """异步评估单个任务。

        流程与 ``evaluate_sync`` 一致，区别在于:

        - ``_prepare`` 通过 ``ProcessPoolExecutor`` 在独立进程中执行，
          避免 CPU 密集型操作阻塞事件循环。
        - ``_evaluate_async`` 以协程方式执行，适配异步 LLM 调用。

        Args:
            task: 原始任务样本（未经预处理）。
            pool: 进程池执行器，用于运行 CPU 密集型的 ``_prepare`` 操作。

        Returns:
            完整的评估结果，已包含计时信息并已持久化。
        """
        start = time.time()
        try:
            loop = asyncio.get_running_loop()
            task = await loop.run_in_executor(pool, self._prepare, task)
            err = self._validate(task)
            result = err if err else await self._evaluate_async(task)
        except Exception as e:
            logger.exception(f"Task {task.safe_id} failed: {e}")
            result = self._error(task, str(e))
        return self._finalize(result, start)

    # ==========================================================
    # 子类必须实现
    # ==========================================================

    @abstractmethod
    def _evaluate_sync(self, task: TaskSample) -> EvaluationResult:
        """同步评估核心逻辑。

        子类在此实现提示词构造、LLM 同步调用及响应解析。
        调用时 task 已完成预处理与校验，可直接使用其字段。

        推荐使用 ``self._result()`` 构造返回值。
        无需手动捕获异常，基类将统一处理。

        Args:
            task: 已预处理且已通过校验的任务样本。

        Returns:
            评估结果实例。
        """
        ...

    @abstractmethod
    async def _evaluate_async(self, task: TaskSample) -> EvaluationResult:
        """异步评估核心逻辑。

        职责与 ``_evaluate_sync`` 相同，使用异步 LLM 调用接口。

        Args:
            task: 已预处理且已通过校验的任务样本。

        Returns:
            评估结果实例。
        """
        ...

    # ==========================================================
    # 可覆写钩子
    # ==========================================================

    def _prepare(self, task: TaskSample) -> TaskSample:
        """任务预处理。

        默认实现会：
        - 复制任务对象，避免原地污染调用方数据；
        - 清理基础文本字段；
        - 将本地图片路径解析并编码为 ``data:image/...``；
        - 保留远程 URL / 已编码 data URL 不变。

        子类覆写时建议先调用 ``super()._prepare(task)``，再追加
        自定义处理（如图片降采样、尺寸归一化）。

        Note:
            异步模式下此方法在独立进程中执行，
            实现中应避免依赖共享可变状态或不可序列化的对象。

        Args:
            task: 原始任务样本。

        Returns:
            预处理后的任务样本。
        """
        prepared = self._clone_task(task)

        prepared.instruction = (prepared.instruction or "").strip()
        prepared.last_action = (prepared.last_action or "").strip()

        history_images = list(prepared.history_images or [])
        if prepared.images_stored_in_minio:
            prepared.history_images = history_images
            return prepared

        prepared.current_image = self._prepare_image_field(prepared.current_image)
        prepared.history_images = [
            self._prepare_image_field(image_ref)
            for image_ref in history_images
            if image_ref
        ]
        return prepared
 
    def _validate(self, task: TaskSample) -> Optional[EvaluationResult]:
        """任务前置校验。

        在预处理之后、核心评估之前执行。用于检查任务是否满足
        评估前提条件（如必需图片是否存在、字段是否完整）。

        默认实现不执行任何校验。子类可覆写以添加特定检查逻辑，
        推荐使用 ``self._error()`` 构造校验失败的返回值。

        Args:
            task: 已预处理的任务样本。

        Returns:
            ``None`` 表示校验通过，流程继续执行核心评估。
            ``EvaluationResult`` 实例表示校验未通过，流程将跳过
            核心评估，直接进入 finalize 阶段。
        """
        return None

    # ==========================================================
    # 结果构造工具
    # ==========================================================

    def _result(
        self, task: TaskSample, llm_judge: Optional[bool],
        content: str = "", reasoning: str = "", **extra,
    ) -> EvaluationResult:
        """构造标准评估结果。

        自动根据 ``llm_judge`` 与 ``task.ground_truth`` 计算 ``status`` 字段。

        Args:
            task: 任务样本。
            llm_judge: LLM 判决结果。``True`` 表示任务成功，
                ``False`` 表示任务失败，``None`` 表示无法判定。
            content: LLM 原始输出文本，默认为空字符串。
            reasoning: LLM 推理过程文本（思维链），默认为空字符串。
            **extra: 附加元数据键值对，将存入 ``EvaluationResult.extra``。

        Returns:
            已填充 ``status`` 的评估结果实例。
        """
        return EvaluationResult(
            task_sample=task, reward_model=self.model_name,
            llm_judge=llm_judge, content=content, reasoning_content=reasoning,
            status=self._resolve_status(llm_judge, task.ground_truth),
            extra=extra or {},
        )

    def _error(self, task: TaskSample, message: str, **extra) -> EvaluationResult:
        """构造错误评估结果。

        生成 ``status="error"`` 的结果，用于表示评估流程异常终止
        或前置校验未通过。

        Args:
            task: 任务样本。
            message: 错误描述信息。
            **extra: 附加元数据键值对，将存入 ``EvaluationResult.extra``。

        Returns:
            ``status="error"`` 的评估结果实例。
        """
        return EvaluationResult(
            task_sample=task, reward_model=self.model_name,
            llm_judge=None, status="error",
            error_message=message, extra=extra or {},
        )

    @staticmethod
    def _resolve_status(
        llm_judge: Optional[bool], ground_truth: Optional[bool],
    ) -> str:
        """根据 LLM 判决与标注标签计算评估状态。

        Args:
            llm_judge: LLM 判决结果，``None`` 表示未能产生有效判决。
            ground_truth: 数据集标注标签，``None`` 表示该样本无标注。

        Returns:
            评估状态字符串:

            - ``"correct"``: ``llm_judge`` 与 ``ground_truth`` 一致。
            - ``"incorrect"``: ``llm_judge`` 与 ``ground_truth`` 不一致。
            - ``"unknown"``: 任一方为 ``None``，无法判定一致性。
        """
        if ground_truth is None or llm_judge is None:
            return "unknown"
        return "correct" if llm_judge == bool(ground_truth) else "incorrect"

    # ==========================================================
    # 内部流程
    # ==========================================================

    def _finalize(self, result: EvaluationResult, start: float) -> EvaluationResult:
        """评估收尾处理。

        为结果填充计时信息并执行持久化。所有评估路径（正常完成、
        校验未通过、异常捕获）均经由此方法统一收尾。

        持久化失败时仅记录错误日志，不会抛出异常或影响结果返回。

        Args:
            result: 待收尾的评估结果。
            start: 评估起始时间戳（``time.time()`` 返回值）。

        Returns:
            同一结果实例，已填充 ``duration`` 与 ``timestamp`` 字段。
        """
        result.duration = time.time() - start
        result.timestamp = time.time()
        return result

    @staticmethod
    def _clone_task(task: TaskSample) -> TaskSample:
        if hasattr(task, "model_copy"):
            return task.model_copy(deep=True)
        return task.copy(deep=True)

    @staticmethod
    def _is_data_url(image_ref: str) -> bool:
        return str(image_ref).startswith("data:image/")

    @staticmethod
    def _is_remote_url(image_ref: str) -> bool:
        scheme = urlparse(str(image_ref)).scheme.lower()
        return scheme in {"http", "https"}

    @classmethod
    def _prepare_image_field(cls, image_ref: str) -> str:
        if not image_ref:
            raise ValueError("image reference is empty")
        if cls._is_data_url(image_ref) or cls._is_remote_url(image_ref):
            return image_ref

        image_path = cls._resolve_image_path(image_ref)
        return cls._encode_image_as_data_url(image_path)

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_image_path(cls, image_ref: str) -> Path:
        raw_path = Path(str(image_ref))
        candidates = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            repo_root = cls._repo_root()
            candidates.extend(
                [
                    repo_root / raw_path,
                    repo_root / "assets" / raw_path,
                    Path.cwd() / raw_path,
                ]
            )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        raise FileNotFoundError(f"image file not found for reference: {image_ref}")

    @staticmethod
    def _encode_image_as_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"
