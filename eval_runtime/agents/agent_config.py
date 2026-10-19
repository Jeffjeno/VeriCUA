from __future__ import annotations

from typing import Literal, Optional

import pydantic
import yaml

class BaseConfig(pydantic.BaseModel):
    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

class ModelConfig(BaseConfig):
    model_name: str
    api_key: str
    base_url: str
    max_retry: int = 3
    retry_waiting_seconds: int = 2
    max_tokens: int = 1024
    temperature: float = 0.0
    max_pixels: int = 12845056
    save_io: bool = False
    save_dir: Optional[str] = None

    class Config:
        extra = "allow"

class AgentConfig(BaseConfig):
    max_steps: int = 100
    enable_log: bool = False
    log_dir: Optional[str] = None

class Qwen3VLAgentConfig(AgentConfig):
    model: Optional[ModelConfig] = None
    platform: str = "ubuntu"
    action_space: str = "pyautogui"
    observation_type: str = "screenshot"
    coordinate_type: Literal["absolute", "relative"] = "relative"
    add_thought_prefix: bool = False
    api_backend: str = "openai"
    enable_thinking: bool = False
    thinking_budget: int = 32768
    client_password: Optional[str] = None
    history_n: int = 4

class Qwen35VLAgentConfig(AgentConfig):
    model: Optional[ModelConfig] = None
    platform: str = "ubuntu"
    action_space: str = "pyautogui"
    observation_type: str = "screenshot"
    coordinate_type: Literal["absolute", "relative"] = "relative"
    add_thought_prefix: bool = False
    api_backend: str = "openai"
    client_password: Optional[str] = None
    history_n: int = 100
    image_max: int = 20
    fold_size: int = 10
    collapse_text: Optional[str] = None

class SamplerAgentConfig(AgentConfig):
    base_agent_type: str = "Qwen3VLAgent"
    base_agent_config_path: Optional[str] = None
    prompt_config: Optional[str] = None

class ReplayAgentConfig(AgentConfig):
    source_result_root: str
    source_model_name: Optional[str] = None
    stamp: Optional[str] = None
    default_rollout_id: str = "0"
    source_rollout_repeat: int = 1
    strict_missing: bool = True
