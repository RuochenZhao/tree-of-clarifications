import functools
from typing import Optional, Union, Any, List

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict

# Lazy imports to avoid loading heavy dependencies at module level
_pinecone = None
_sentence_transformers = None

def _get_pinecone():
    """Lazy import of Pinecone."""
    global _pinecone
    if _pinecone is None:
        try:
            from pinecone import Pinecone
            _pinecone = Pinecone
        except ImportError:
            raise ImportError("❌ Pinecone package not found. Please install with: pip install pinecone")
    return _pinecone

def _get_sentence_transformer():
    """Lazy import of SentenceTransformer."""
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformers = SentenceTransformer
        except ImportError:
            raise ImportError("❌ SentenceTransformers not found. Please install with: pip install sentence-transformers")
    return _sentence_transformers


class PineconeRetriever:
    """Wrapper for Pinecone-based Wikipedia retrieval."""

    def __init__(
        self,
        api_key: str,
        index_name: str = "wikipedia",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize Pinecone retriever.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index (default: "wikipedia")
            model_name: SentenceTransformer model for encoding queries
        """
        self.api_key = api_key
        self.index_name = index_name
        
        # Initialize Pinecone (lazy)
        Pinecone = _get_pinecone()
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize sentence transformer for query encoding (lazy)
        SentenceTransformer = _get_sentence_transformer()
        self.model = SentenceTransformer(model_name)
        
        print(f"✅ Pinecone retriever initialized with index '{index_name}' and model '{model_name}'")

    def __call__(
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        """
        Retrieve passages using Pinecone.
        
        Args:
            query: Search query
            k: Number of results to return
            simplify: If True, return only text content
            
        Returns:
            List of passages as strings or dotdict objects
        """
        try:
            topk: list[dict[str, Any]] = pinecone_search(
                self.index, self.model, query, k
            )
        except Exception as e:
            print(f"❌ Pinecone search failed: {e}")
            return []

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@CacheMemory.cache
def pinecone_search(index, model, query: str, k: int):
    """
    Perform cached Pinecone search.
    
    Args:
        index: Pinecone index object
        model: SentenceTransformer model
        query: Search query
        k: Number of results to return
        
    Returns:
        List of search results
    """
    try:
        # Encode query using sentence transformer
        query_embedding = model.encode(query).tolist()
        
        # Search Pinecone index
        results = index.query(
            vector=query_embedding, 
            top_k=k, 
            include_metadata=True
        )
        
        # Process results to match ColBERT format
        processed_results = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            
            # Extract text content from metadata
            text_content = metadata.get('text', metadata.get('content', ''))
            if not text_content:
                # If no text in metadata, try to use the ID or other fields
                text_content = metadata.get('title', match.get('id', ''))
            
            # Create result dict compatible with ColBERT format
            result = {
                'long_text': text_content,
                'text': text_content,
                'score': match.get('score', 0.0),
                'id': match.get('id', ''),
                **metadata  # Include all metadata
            }
            
            processed_results.append(result)
        
        print(f"🔍 Pinecone retrieved {len(processed_results)} passages for query: '{query[:50]}...'")
        return processed_results
        
    except Exception as e:
        print(f"❌ Pinecone search error: {e}")
        print(f"   Query: {query}")
        print(f"   K: {k}")
        raise


@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def pinecone_search_cached(*args, **kwargs):
    """Cached wrapper for pinecone_search."""
    return pinecone_search(*args, **kwargs)


# Alternative implementation that mimics ColBERT's interface more closely
class PineconeColBERTv2:
    """Drop-in replacement for ColBERTv2 using Pinecone."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str = "wikipedia",
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs  # Accept any additional kwargs for compatibility
    ):
        """
        Initialize Pinecone retriever with ColBERT-compatible interface.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            model_name: SentenceTransformer model name
        """
        self.retriever = PineconeRetriever(api_key, index_name, model_name)
        
    def __call__(
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        """
        Retrieve passages using Pinecone with ColBERT-compatible interface.
        """
        return self.retriever(query, k, simplify)