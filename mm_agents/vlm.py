import random
import time
import asyncio
import logging
from typing import Any, Optional
from openai import OpenAI, ChatCompletion

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover
    def retry(*_a, **_k):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator

    def stop_after_attempt(*_a, **_k):  # type: ignore
        return None

    def wait_exponential(*_a, **_k):  # type: ignore
        return None

logger = logging.getLogger(__name__)
import os
import json
import time
import uuid


def save_vlm_io(messages, request_kwargs, response, save_dir="vlm_logs", model_name=None):
    os.makedirs(save_dir, exist_ok=True)

    record_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(save_dir, f"{record_id}.json")

    def _to_jsonable(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, list):
            return [_to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}

        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass

        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            try:
                return {k: _to_jsonable(v) for k, v in vars(obj).items()}
            except Exception:
                pass

        return repr(obj)

    def _sanitize_messages(msgs):
        import copy
        msgs = copy.deepcopy(msgs)

        for msg in msgs:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    if "image_url" in item:
                        image_url = item.get("image_url")
                        if isinstance(image_url, dict) and "url" in image_url:
                            url = image_url["url"]
                            if isinstance(url, str) and url.startswith("data:"):
                                image_url["url"] = f"<base64_image len={len(url)}>"

                    if "image" in item and isinstance(item["image"], str):
                        item["image"] = f"<image_data len={len(item['image'])}>"

        return msgs

    data = {
        "id": record_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "messages": _sanitize_messages(messages),
        "request_kwargs": _to_jsonable(request_kwargs),
        "response": _to_jsonable(response),
    }

    try:
        choice = response.choices[0]
        msg = choice.message
        data["parsed"] = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "content": getattr(msg, "content", None),
            "tool_calls": _to_jsonable(getattr(msg, "tool_calls", None)),
            "role": getattr(msg, "role", None),
        }
    except Exception as e:
        data["parsed_error"] = repr(e)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return save_path

class VLMWrapper:
    """VLM Wrapper for UI Operation.
    """

    def __init__(
            self,
            model_name: str,
            api_key: str,
            base_url: str | list[str],
            max_retry: int = 3,
            retry_waiting_seconds: int = 2,
            max_tokens: int = 1024,
            temperature: float = 0.0,
            max_pixels: Optional[int] = None,
            policy: str = "random",
            save_io: bool = False,
            save_dir: Optional[str] = None,
            **vlm_kwargs
        ):
        self.model_name = model_name
        if isinstance(base_url, str):
            base_url = [base_url]
        elif isinstance(base_url, list):  
            pass
        self.clients = [OpenAI(api_key=api_key, base_url=url) for url in base_url]
        # self.client = OpenAI(api_key=api_key, base_url=base_url)
        # self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.max_retry = max_retry
        self.retry_waiting_seconds = retry_waiting_seconds
        self.policy = policy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_pixels = max_pixels
        self.save_io = save_io
        self.save_dir = save_dir

        self.vlm_kwargs = vlm_kwargs

    @staticmethod
    def _build_request_kwargs(kwargs: dict) -> dict:
        request_kwargs = dict(kwargs)
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})

        enable_thinking = request_kwargs.pop("enable_thinking", None)
        if enable_thinking is not None:
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
            chat_template_kwargs["enable_thinking"] = enable_thinking
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        top_k = request_kwargs.pop("top_k", None)
        if top_k is not None:
            extra_body["top_k"] = top_k

        min_p = request_kwargs.pop("min_p", None)
        if min_p is not None:
            extra_body["min_p"] = min_p

        repetition_penalty = request_kwargs.pop("repetition_penalty", None)
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty

        if extra_body:
            request_kwargs["extra_body"] = extra_body

        return request_kwargs

    @retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=10))
    def predict(self, messages, stream: bool=False, **kwargs) -> ChatCompletion:
        """Predict the next action given the history and the current screenshot.

        Args:
            messages: The messages to send to the model.
        
        Returns:
            The ChatCompletion of the VLM
        """
        counter = self.max_retry
        wait_seconds = self.retry_waiting_seconds

        # import copy
        # messages_s = copy.deepcopy(messages)
        # for msg in messages_s:
        #     content = msg['content']
        #     if isinstance(content, list):
        #         for cnt in content:
        #             cnt.pop('image_url', None)
        # import json
        # print("messages: ", json.dumps(messages_s, ensure_ascii=False, indent=2))

        # Add max_pixels to all image content if specified
        if self.max_pixels is not None:
            import copy
            messages = copy.deepcopy(messages)
            for msg in messages:
                content = msg.get('content')
                if isinstance(content, list):
                    for cnt in content:
                        if 'image' in cnt.get('type', ''):
                            cnt['max_pixels'] = self.max_pixels

        request_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        request_kwargs.update(self.vlm_kwargs)
        request_kwargs.update(kwargs)
        request_kwargs = self._build_request_kwargs(request_kwargs)
        logger.info("Sampling Params: %s", request_kwargs)
        if self.policy == "random":
            client = random.choice(self.clients)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                **request_kwargs
            )
        if self.save_io:
            save_vlm_io(
                messages=messages,
                request_kwargs=request_kwargs,
                response=response,
                save_dir=self.save_dir or "vlm_logs",
                model_name=self.model_name,
            )
        return response
