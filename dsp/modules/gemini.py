import functools
import json
import time
from typing import Any, Optional

from interlinked import AI
from interlinked.core.clients.googleaiclient import GoogleAIClient

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import LM


class GeminiResponse:
    """Mock response object to maintain compatibility with OpenAI response format"""
    def __init__(self, text: str):
        self.text = text
        self.choices = [{"text": text, "finish_reason": "stop", "message": {"content": text}}]

    def __getitem__(self, key):
        if key == "choices":
            return self.choices
        return getattr(self, key, None)


class Gemini(LM):
    """Wrapper around Google's Gemini API using interlinked library.

    Args:
        model (str, optional): Gemini model to use. Defaults to "gemini-1.5-flash-002".
        api_key (Optional[str], optional): Google AI API key. Defaults to provided key.
        max_retries (int, optional): Maximum number of retries for rate limit errors. Defaults to 5.
        initial_retry_delay (float, optional): Initial delay in seconds for exponential backoff. Defaults to 1.0.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash-002",
        api_key: Optional[str] = "in-8LxOfglvSxalWFfVDbd7ug",
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "gemini"
        self.model_type = "chat"  # Gemini is chat-based

        # Initialize the Google AI client
        self.client = GoogleAIClient(model_name=model, api_key=api_key)

        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 300,
            "top_p": 1,
            "n": 1,
            **kwargs,
        }
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> GeminiResponse:
        """Make a request to Gemini API with retry logic for rate limits"""
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}

        response = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Call Gemini using interlinked library
                observation = AI.ask(prompt=prompt, client=self.client)
                response_text = observation.response

                # Create mock response object for compatibility
                response = GeminiResponse(response_text)
                break  # Success - exit retry loop

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                is_rate_limit = ('rate limit' in error_msg or
                                '429' in error_msg or
                                'quota' in error_msg or
                                'resource exhausted' in error_msg)

                if attempt < self.max_retries - 1:
                    # Not the last attempt - retry with exponential backoff
                    wait_time = self.initial_retry_delay * (2 ** attempt)
                    if is_rate_limit:
                        print(f"⚠️  Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt+2}/{self.max_retries}...")
                    else:
                        print(f"⚠️  Gemini API error (attempt {attempt+1}/{self.max_retries}): {e}")
                        print(f"⚠️  Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    # Last attempt failed - stop the script
                    if is_rate_limit:
                        print(f"❌ Rate limit exceeded after {self.max_retries} retries.")
                    else:
                        print(f"❌ API error persisted after {self.max_retries} retries.")
                    print(f"❌ Error: {e}")
                    print("❌ Stopping script.")
                    raise RuntimeError(f"Gemini API failed after {self.max_retries} retries: {e}")

        # If we exhausted retries without success and no response was set
        if response is None:
            print(f"❌ All retry attempts failed. Stopping script.")
            raise RuntimeError(f"Gemini API failed after {self.max_retries} retries: {last_error}")

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def request(self, prompt: str, **kwargs) -> GeminiResponse:
        """Handles retrieval of Gemini completions with caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]
        
        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        """Extract text from choice object"""
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        return choice.get("text", "")

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """Retrieves completions from Gemini.

        Args:
            prompt (str): prompt to send to Gemini
            only_completed (bool, optional): return only completed responses. Defaults to True.
            return_sorted (bool, optional): sort completions. Defaults to False.

        Returns:
            list[str]: list of completion texts
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)
        choices = response.choices

        completed_choices = [c for c in choices if c.get("finish_reason") != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]

        return completions


# Cache functions for compatibility (though Gemini doesn't need the same caching)
@CacheMemory.cache
def cached_gemini_request(**kwargs):
    """Cached Gemini request - placeholder for compatibility"""
    return kwargs.get("response", "")


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gemini_request_wrapped(**kwargs):
    return cached_gemini_request(**kwargs)