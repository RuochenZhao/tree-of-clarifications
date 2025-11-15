import functools
from typing import Optional, Union, Any
import requests

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict


# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2:
    """Wrapper for the ColBERTv2 Retrieval."""

    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: Optional[Union[str, int]] = None,
        post_requests: bool = False,
    ):
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def __call__(
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@CacheMemory.cache
def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert (
        k <= 100
    ), "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    
    try:
        res = requests.get(url, params=payload, timeout=10)
        res.raise_for_status()  # Raise an exception for bad status codes
        
        response_data = res.json()
        print(f"🔍 ColBERT response keys: {list(response_data.keys())}")
        
        # Handle different possible response formats
        if "error" in response_data and response_data["error"]:
            error_msg = response_data.get("message", "Unknown error")
            print(f"❌ ColBERT server error: {error_msg}")
            raise ConnectionError(f"ColBERT server error: {error_msg}")
        elif "topk" in response_data:
            topk = response_data["topk"][:k]
        elif "results" in response_data:
            topk = response_data["results"][:k]
        elif "passages" in response_data:
            topk = response_data["passages"][:k]
        else:
            print(f"⚠️  Unexpected response format: {response_data}")
            # Try to use the response as-is if it's a list
            if isinstance(response_data, list):
                topk = response_data[:k]
            else:
                raise KeyError(f"Expected 'topk', 'results', or 'passages' in response, got: {list(response_data.keys())}")
        
        # Ensure each item has the required fields
        processed_topk = []
        for d in topk:
            if isinstance(d, dict):
                # Add long_text field if it doesn't exist
                if "long_text" not in d and "text" in d:
                    d = {**d, "long_text": d["text"]}
                elif "long_text" not in d and "content" in d:
                    d = {**d, "long_text": d["content"]}
                elif "long_text" not in d:
                    # If no text field found, use the whole dict as text
                    d = {**d, "long_text": str(d)}
                processed_topk.append(d)
            else:
                # If it's not a dict, wrap it
                processed_topk.append({"long_text": str(d), "text": str(d)})
        
        return processed_topk[:k]
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ColBERT request failed: {e}")
        print(f"   URL: {url}")
        print(f"   Payload: {payload}")
        raise
    except Exception as e:
        print(f"❌ ColBERT response parsing failed: {e}")
        print(f"   Response status: {res.status_code}")
        print(f"   Response text: {res.text[:500]}...")
        raise


@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    return colbertv2_get_request_v2(*args, **kwargs)


colbertv2_get_request = colbertv2_get_request_v2_wrapped


@CacheMemory.cache
def colbertv2_post_request_v2(url: str, query: str, k: int):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        
        response_data = res.json()
        print(f"🔍 ColBERT POST response keys: {list(response_data.keys())}")
        
        # Handle different possible response formats
        if "error" in response_data and response_data["error"]:
            error_msg = response_data.get("message", "Unknown error")
            print(f"❌ ColBERT server error: {error_msg}")
            raise ConnectionError(f"ColBERT server error: {error_msg}")
        elif "topk" in response_data:
            return response_data["topk"][:k]
        elif "results" in response_data:
            return response_data["results"][:k]
        elif "passages" in response_data:
            return response_data["passages"][:k]
        else:
            print(f"⚠️  Unexpected POST response format: {response_data}")
            if isinstance(response_data, list):
                return response_data[:k]
            else:
                raise KeyError(f"Expected 'topk', 'results', or 'passages' in response, got: {list(response_data.keys())}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ ColBERT POST request failed: {e}")
        print(f"   URL: {url}")
        print(f"   Payload: {payload}")
        raise
    except Exception as e:
        print(f"❌ ColBERT POST response parsing failed: {e}")
        print(f"   Response status: {res.status_code}")
        print(f"   Response text: {res.text[:500]}...")
        raise


@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def colbertv2_post_request_v2_wrapped(*args, **kwargs):
    return colbertv2_post_request_v2(*args, **kwargs)


colbertv2_post_request = colbertv2_post_request_v2_wrapped
