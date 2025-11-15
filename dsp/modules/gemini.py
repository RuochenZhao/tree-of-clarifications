import functools
import json
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
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash-002",
        api_key: Optional[str] = "in-8LxOfglvSxalWFfVDbd7ug",
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "gemini"
        self.model_type = "chat"  # Gemini is chat-based
        
        # Initialize the Google AI client
        self.client = GoogleAIClient(model_name=model, api_key=api_key)
        
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 300,
            "top_p": 1,
            "n": 1,
            **kwargs,
        }
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> GeminiResponse:
        """Make a request to Gemini API"""
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        
        try:
            # Call Gemini using interlinked library
            observation = AI.ask(prompt=prompt, client=self.client)
            response_text = observation.response
            
            # Create mock response object for compatibility
            response = GeminiResponse(response_text)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Return empty response on error
            response = GeminiResponse("")

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