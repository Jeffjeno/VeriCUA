import logging

from openai import APIStatusError, AsyncOpenAI, OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


NON_RETRYABLE_STATUS_CODES = {503}
logger = logging.getLogger(__name__)


def should_retry_openai_error(exc: BaseException) -> bool:
    if isinstance(exc, APIStatusError) and exc.status_code in NON_RETRYABLE_STATUS_CODES:
        response = exc.response
        headers = response.headers if response is not None else {}
        body = response.text if response is not None else ""
        logger.error(
            "Non-retryable OpenAI status %s from worker=%s policy=%s body=%s",
            exc.status_code,
            headers.get("x-router-worker"),
            headers.get("x-router-policy"),
            body[:2000],
        )
        return False
    return True


class Client:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        request_timeout: float | None = None,
    ):
        """
        Initializes the Client for interacting with the OpenAI API.
        Args:
        - model (str): The name of the model to use for chat completions.
        - base_url (str): The base URL of the OpenAI API.
        - api_key (str): The API key for authenticating with the OpenAI API.
        - temperature (float, optional): Sampling temperature for the model. Default is 0.0.
        - max_tokens (int, optional): Max OUTPUT/completion tokens (OpenAI-compatible
          ``max_tokens``). Does not cap prompt/context length — that is the server
          context window. Default is 16384.
        - top_p (float, optional): Nucleus sampling parameter. Default is None (disabled).
        - top_k (int, optional): Top-k sampling parameter. Default is None (disabled).
        - min_p (float, optional): Minimum probability threshold for token generation. Default is None (disabled).
        - presence_penalty (float, optional): Presence penalty for token generation. Default is None (disabled).
        - repetition_penalty (float, optional): Repetition penalty for token generation. Default is None (disabled).
        - enable_thinking (bool, optional): Whether to enable the "thinking" feature in the chat template. Default is True.
        - request_timeout (float, optional): OpenAI-compatible request timeout in seconds.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": 0,
        }
        if request_timeout is not None:
            client_kwargs["timeout"] = float(request_timeout)
        self.sync_client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking
        params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "enable_thinking": enable_thinking,
            "request_timeout": request_timeout,
        }

        provided = [f"{k}={v}" for k, v in params.items() if v is not None]
        if provided:
            print(f"Initialized Client with Sampling Params: {', '.join(provided)}")
    def _build_request_params(self, messages: list[dict]) -> dict:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra_body = {}

        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty

        if self.top_k is not None:
            extra_body["top_k"] = self.top_k

        if self.min_p is not None:
            extra_body["min_p"] = self.min_p

        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty

        if extra_body:
            params["extra_body"] = extra_body

        return params
    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(should_retry_openai_error),
        reraise=True,
    )
    def chat_completion_sync(self, messages):
        return self.sync_client.chat.completions.create(
            **self._build_request_params(messages)
        )

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(should_retry_openai_error),
        reraise=True,
    )
    async def chat_completion_async(self, messages):
        return await self.async_client.chat.completions.create(
            **self._build_request_params(messages)
        )
