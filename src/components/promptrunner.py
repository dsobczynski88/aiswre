"""
This module defines the PromptRunner class, which facilitates running 
prompts asynchronously using multiple chains of actions. It is designed 
to work with large-language models (LLMs) and follows a structured 
approach for input and output management. Additionally, helper functions are defined
to enable the INCOSE reviewer use case.
"""

import logging
import json
from typing import Union, List, Dict, Any, Optional, Callable, Awaitable, TypeVar, Tuple
import random
import time
import nest_asyncio
import asyncio
from collections import deque
from tqdm.asyncio import tqdm_asyncio
from langchain_core.runnables import RunnableSequence
from openai import OpenAI, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set event loop
loop = asyncio.ProactorEventLoop()
asyncio.set_event_loop(loop)

# Set generic data type "T"
T = TypeVar("T")

def _now() -> float:
    return time.time()

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _get_header_ci(headers: Dict[str, str], key: str) -> Optional[str]:
    if not headers:
        return None
    key_lower = key.lower()
    for k, v in headers.items():
        if k.lower() == key_lower:
            return v
    return None

def _estimate_tokens_from_messages(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Estimate tokens for a list of chat messages.
    Tries tiktoken if available; otherwise uses a simple character-based heuristic.
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a default encoding if the model isn't known
            encoding = tiktoken.get_encoding("cl100k_base")
        text_pieces = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # If content is list (tool calls, etc.), flatten text-like fields
                flat = []
                for c in content:
                    if isinstance(c, dict):
                        # A naive attempt to join textual fields
                        flat.append(str(c.get("text", "")))
                    else:
                        flat.append(str(c))
                content_str = "\n".join(flat)
            else:
                content_str = str(content)
            text_pieces.append(f"{role}: {content_str}")
        full_text = "\n".join(text_pieces)
        return len(encoding.encode(full_text))
    except Exception:
        # Heuristic: ~1 token per 4 characters (typical for English text).
        # This is conservative enough for throttling.
        text_pieces = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                flat = []
                for c in content:
                    if isinstance(c, dict):
                        flat.append(str(c.get("text", "")))
                    else:
                        flat.append(str(c))
                content_str = "\n".join(flat)
            else:
                content_str = str(content)
            text_pieces.append(f"{role}: {content_str}")
        full_text = "\n".join(text_pieces)
        return max(1, len(full_text) // 4)
    
async def async_retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    initial_delay: float = 60,
    factor: float = 2.0,
    jitter: bool = True,
    max_retries: int = 5,
    retry_on = RateLimitError,
    # Token-aware additions:
    token_limiter: Optional["OpenAITokenLimiter"] = None,
    est_tokens: int = 0,
    **kwargs
    ) -> T:
    """
    Async retry function with exponential backoff for OpenAI API calls.
    Enhanced to handle token-rate-limit errors by consulting server headers
    and the token limiter when available.
    """
    delay = initial_delay
    last_error: Optional[Exception] = None

    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except retry_on as e:  # Typically RateLimitError
            last_error = e
            # Try to detect token-specific rate limit and use server-provided guidance
            wait_override = None
            try:
                # Newer OpenAI errors may include response and headers
                response = getattr(e, "response", None)
                headers = getattr(response, "headers", {}) if response else {}
                # Prefer server-provided reset time for tokens if present
                reset_tokens_header = _get_header_ci(headers, "x-ratelimit-reset-tokens")
                if reset_tokens_header:
                    # Often in seconds (float). We'll be conservative.
                    wait_override = float(reset_tokens_header)
                else:
                    # If not present, see if the error mentions tokens
                    if "token" in str(e).lower() and token_limiter:
                        wait_override = await token_limiter.suggest_wait_time(est_tokens)
            except Exception:
                # If anything goes wrong while parsing headers, ignore
                pass

            # Compute backoff sleep
            sleep_time = delay * (1.0 + (random.random() if jitter else 0.0))
            if wait_override is not None:
                sleep_time = max(sleep_time, float(wait_override))

            print(f"Encountered rate limit error: {e}, retry attempt {i+1}/{max_retries}, sleeping {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)
            delay *= factor
            continue
        except Exception as e:
            # Non-rate-limit errors: do not retry by default
            raise e

    # If exhausted retries, re-raise the last rate limit error (if any)
    if last_error:
        raise last_error
    raise Exception("Max retries exceeded.")

class OpenAIRateLimiter:
    """Rate limiter for OpenAI API to stay under requests per minute (RPM)."""

    def __init__(self, max_requests_per_minute: int = 490):  # Buffer under the limit
        self.max_requests = max_requests_per_minute
        self.request_timestamps: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if we're approaching the RPM limit."""
        async with self.lock:
            current_time = _now()
            # purge old
            while self.request_timestamps and current_time - self.request_timestamps[0] >= 60:
                self.request_timestamps.popleft()

            if len(self.request_timestamps) >= self.max_requests:
                oldest_timestamp = self.request_timestamps[0]
                wait_time = max(0.0, 60 - (current_time - oldest_timestamp) + 0.1)
                if wait_time > 0:
                    print(f"RPM limit approaching, waiting {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    # purge again after sleep
                    current_time = _now()
                    while self.request_timestamps and current_time - self.request_timestamps[0] >= 60:
                        self.request_timestamps.popleft()

            # record request
            self.request_timestamps.append(_now())

class OpenAITokenLimiter:
    """
    Token-per-minute (TPM) limiter using a rolling 60-second window.

    Records token usage as (timestamp, tokens) entries and enforces a maximum
    total across the last 60 seconds. Provides proactive waiting (using estimates)
    and recording of actual usage after the call returns.
    """

    def __init__(self, max_tokens_per_minute: int):
        if max_tokens_per_minute <= 0:
            raise ValueError("max_tokens_per_minute must be positive")
        self.max_tokens = max_tokens_per_minute
        self.entries: deque[Tuple[float, int]] = deque()
        self.total_tokens_in_window: int = 0
        self.lock = asyncio.Lock()

    def _purge_old(self, now: float):
        while self.entries and now - self.entries[0][0] >= 60:
            _, tokens = self.entries.popleft()
            self.total_tokens_in_window -= tokens
            if self.total_tokens_in_window < 0:
                self.total_tokens_in_window = 0

    async def wait_if_needed(self, est_tokens: int):
        """
        Wait until adding est_tokens would not exceed the TPM.
        This does NOT reserve tokens; it only throttles proactively.
        """
        if est_tokens <= 0:
            return
        async with self.lock:
            now = _now()
            self._purge_old(now)

            while self.total_tokens_in_window + est_tokens > self.max_tokens and self.entries:
                # Need to wait for some tokens to expire
                oldest_ts, _ = self.entries[0]
                wait_time = max(0.0, 60 - (now - oldest_ts) + 0.05)
                print(f"TPM limit approaching, waiting {wait_time:.2f}s (est_tokens={est_tokens})...")
                await asyncio.sleep(wait_time)
                now = _now()
                self._purge_old(now)

    async def record(self, tokens: int):
        """Record actual token usage after a response."""
        if tokens <= 0:
            return
        async with self.lock:
            now = _now()
            self._purge_old(now)
            self.entries.append((now, tokens))
            self.total_tokens_in_window += tokens

    async def suggest_wait_time(self, est_tokens: int) -> float:
        """
        Suggest a wait time (without mutating state) to allow est_tokens to fit.
        """
        if est_tokens <= 0:
            return 0.0
        async with self.lock:
            now = _now()
            # Simulate purge
            tmp_entries = [(ts, tok) for (ts, tok) in self.entries if now - ts < 60]
            total = sum(tok for _, tok in tmp_entries)
            if total + est_tokens <= self.max_tokens:
                return 0.0
            # Find when enough tokens expire
            tmp_entries.sort(key=lambda x: x[0])
            # We'll drop the oldest entries until we can fit est_tokens
            idx = 0
            while idx < len(tmp_entries) and total + est_tokens > self.max_tokens:
                oldest_ts, oldest_tok = tmp_entries[idx]
                total -= oldest_tok
                idx += 1
                if total + est_tokens <= self.max_tokens:
                    return max(0.0, 60 - (now - oldest_ts) + 0.05)

            # If we couldn't compute a better value, fallback to a small wait
            return 1.0

class RateLimitOpenAIClient:
    """
    Async Support: Fully compatible with AsyncOpenAI and async/await patterns

    Proactive Rate Limiting:
    - Tracks request timestamps (RPM)
    - Tracks token usage in a rolling window (TPM) using an estimator
    - Waits when approaching either limit to avoid most rate limit errors

    Reactive Retries:
    - If the API still returns rate limit errors (including token limits),
        retries with exponential backoff, respecting server-provided reset headers.

    Concurrent Request Handling:
    - Safe for asyncio.gather usage

    Usage Notes:
    - Default RPM is set to 490 (buffer under 500)
    - TPM must be supplied to enforce token limits (varies by model/account)
    - Token estimation uses tiktoken if available; otherwise, a heuristic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 490,
        max_tokens_per_minute: Optional[int] = 200000,
        token_estimator: Optional[Callable[[List[Dict[str, Any]], str], int]] = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.rate_limiter = OpenAIRateLimiter(max_requests_per_minute)
        self.token_limiter = OpenAITokenLimiter(max_tokens_per_minute) if max_tokens_per_minute else None
        self._token_estimator_fn = token_estimator or (lambda messages, model: _estimate_tokens_from_messages(messages, model))

    def _estimate_total_tokens(self, model: str, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> int:
        """
        Estimate total tokens for proactive throttling:
        total_est = estimated_prompt_tokens + expected_completion_tokens
        Completion estimate uses max_tokens / max_completion_tokens if provided,
        otherwise a conservative default.
        """
        est_prompt = 0
        try:
            est_prompt = self._token_estimator_fn(messages, model)
        except Exception:
            # Fall back gracefully
            est_prompt = _estimate_tokens_from_messages(messages, model)

        # Try to detect completion limit arg names
        est_completion = 0
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if key in kwargs:
                est_completion = _safe_int(kwargs[key], 0)
                break

        if est_completion <= 0:
            # Conservative default to reduce oversubscription; adjust if you know your outputs
            est_completion = 512

        return est_prompt + est_completion

    async def chat_completion(self, model: str, messages: list, **kwargs) -> ChatCompletion:
        """Make a chat completion request with RPM/TPM limiting and token-aware retries."""
        await self.rate_limiter.wait_if_needed()

        est_tokens = 0
        if self.token_limiter:
            est_tokens = self._estimate_total_tokens(model, messages, kwargs)
            await self.token_limiter.wait_if_needed(est_tokens)

        # Perform the request with token-aware retry/backoff
        completion: ChatCompletion = await async_retry_with_backoff(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            token_limiter=self.token_limiter,
            est_tokens=est_tokens,
            **kwargs,
        )

        # Record actual token usage (if available)
        if self.token_limiter:
            try:
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    total_tokens = getattr(usage, "total_tokens", None)
                    if total_tokens is None:
                        # Fallback if total isn't present
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                        total_tokens = prompt_tokens + completion_tokens
                    if total_tokens:
                        await self.token_limiter.record(int(total_tokens))
            except Exception:
                # Do not fail the request if usage parsing fails
                pass

        return completion

    async def chat_completion_parse(self, model: str, messages: list, **kwargs) -> ChatCompletion:
        """Make a chat completion request with RPM/TPM limiting and token-aware retries (parse endpoint)."""
        await self.rate_limiter.wait_if_needed()

        est_tokens = 0
        if self.token_limiter:
            est_tokens = self._estimate_total_tokens(model, messages, kwargs)
            await self.token_limiter.wait_if_needed(est_tokens)

        completion: ChatCompletion = await async_retry_with_backoff(
            self.client.chat.completions.parse,
            model=model,
            messages=messages,
            token_limiter=self.token_limiter,
            est_tokens=est_tokens,
            **kwargs,
        )

        if self.token_limiter:
            try:
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    total_tokens = getattr(usage, "total_tokens", None)
                    if total_tokens is None:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                        total_tokens = prompt_tokens + completion_tokens
                    if total_tokens:
                        await self.token_limiter.record(int(total_tokens))
            except Exception:
                pass

        return completion
class ResponseClient:
    def init(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.previous_response_ids: List[str] = []
        self.previous_responses: List[str] = []

    def get_response(
        self,
        input: Union[List, str],
        print_response: bool = True,
        store: bool = True,  # Kept for backwards compatibility
        previous_response_id: Optional[str] = None,  # Kept for backwards compatibility
        **kwargs
        ):
        # Provide the essentials, then allow override via kwargs
        params = {
            "model": self.model,
            "input": input,
            "store": store,
            "previous_response_id": previous_response_id,
        }
        params.update(kwargs)  # allow all other supported/needed arguments

        response = self.client.responses.create(**params)
        self.previous_responses.append(response.output_text)
        self.previous_response_ids.append(response.id)
        if print_response:
            print("Printing `response.output_text`:\n\n", response.output_text)
        return response

    def get_structured_response(
        self,
        messages,
        response_format,
        **kwargs
        ):
        params = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
        }
        params.update(kwargs)

        completion = self.client.beta.chat.completions.parse(**params)
        return completion.choices[0].message

    @staticmethod
    def check_structured_output(completion):
        # If the model refuses to respond, you will get a refusal message
        if getattr(completion, "refusal", False):
            print(completion.refusal)
        else:
            print(getattr(completion, "parsed", completion))

class PromptRunner:
    """
    A class to run LLM prompts asynchronously through processing chains.

    This class manages the execution of multiple prompt chains in parallel,
    handling structured output when needed.
    """

    LOGGER_NAME = "projectlog.PromptRunner"

    def __init__(self, llm, pydantic_model=None, use_structured_output=False):
        """
        Initialize the PromptRunner with an LLM and optional structured output configuration.

        Args:
            llm: The language model to use for processing
            pydantic_model: Optional Pydantic model for structured output
            use_structured_output: Whether to use structured output formatting
        """
        self._logger = logging.getLogger(self.LOGGER_NAME)
        self.llm = llm

        if use_structured_output and pydantic_model:
            self.llm = self.llm.with_structured_output(
                pydantic_model,
                method="json_schema"
            )

    @property
    def logger(self):
        """Access the logger instance."""
        return self._logger

    async def run_multiple_chains(
        self,
        chains: List[RunnableSequence],
        args_lists: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute multiple chains asynchronously with their respective inputs.

        Args:
            chains: List of RunnableSequence objects to execute
            args_lists: List of input dictionaries for each chain

        Returns:
            List of results from each chain execution

        Raises:
            ValueError: If the lengths of chains and args_lists don't match
        """
        if len(chains) != len(args_lists):
            raise ValueError("Number of chains must match number of argument lists")

        self.logger.info(f"Preparing to run {len(chains)} chains asynchronously")

        tasks = [
            self.run_chain(chain, args)
            for chain, args in zip(chains, args_lists)
        ]

        self.logger.info('Awaiting results...')
        results = await tqdm_asyncio.gather(*tasks)
        self.logger.info('Results fetched successfully')

        return results

    async def run_chain(
        self,
        chain: RunnableSequence,
        inputs: Optional[Dict[str, Any]]
    ) -> Any:
        """
        Execute a single chain asynchronously with the given inputs.

        Args:
            chain: The RunnableSequence to execute
            inputs: Dictionary of input variables for the chain

        Returns:
            The result of the chain execution

        Raises:
            ValueError: If inputs is None
        """
        if inputs is None:
            raise ValueError("Input arguments cannot be None")

        return await chain.ainvoke(inputs)