import functools
import requests
import re
from typing import Optional, Union, Any, List

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict


class WikipediaRetriever:
    """Wikipedia API-based retrieval for Wikipedia content."""

    def __init__(self, language: str = "en", user_agent: str = "TreeOfClarifications/1.0"):
        """
        Initialize Wikipedia retriever.
        
        Args:
            language: Wikipedia language code (default: "en")
            user_agent: User agent string for API requests
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.user_agent = user_agent
        
        print(f"✅ Wikipedia retriever initialized for {language}.wikipedia.org")

    def __call__(
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        """
        Search Wikipedia and retrieve passages.
        
        Args:
            query: Search query
            k: Number of results to return
            simplify: If True, return only text content
            
        Returns:
            List of passages as strings or dotdict objects
        """
        try:
            results = wikipedia_search(self.base_url, query, k, self.user_agent)
        except Exception as e:
            print(f"❌ Wikipedia search failed: {e}")
            return []

        if simplify:
            return [result["long_text"] for result in results]

        return [dotdict(result) for result in results]


@CacheMemory.cache
def wikipedia_search(base_url: str, query: str, limit: int, user_agent: str):
    """
    Search Wikipedia and get article excerpts with caching.
    
    Args:
        base_url: Wikipedia API base URL
        query: Search query
        limit: Number of results to return
        user_agent: User agent string
        
    Returns:
        List of search results with text content
    """
    headers = {'User-Agent': user_agent}
    
    try:
        # Step 1: Search for articles
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit,
            'srprop': 'snippet|titlesnippet|size|wordcount'
        }
        
        print(f"🔍 Searching Wikipedia for: '{query}'")
        search_response = requests.get(base_url, params=search_params, headers=headers, timeout=10)
        search_response.raise_for_status()
        
        search_results = search_response.json().get('query', {}).get('search', [])
        
        if not search_results:
            print(f"⚠️  No Wikipedia results found for: '{query}'")
            return []
        
        # Step 2: Get full text for each article
        passages = []
        for i, result in enumerate(search_results[:limit]):
            try:
                # Get article extract
                extract_params = {
                    'action': 'query',
                    'prop': 'extracts|info',
                    'pageids': result['pageid'],
                    'explaintext': True,
                    'exintro': False,  # Get full article, not just intro
                    'exsectionformat': 'plain',
                    'format': 'json',
                    'inprop': 'url'
                }
                
                extract_response = requests.get(base_url, params=extract_params, headers=headers, timeout=10)
                extract_response.raise_for_status()
                
                pages = extract_response.json().get('query', {}).get('pages', {})
                page_data = pages.get(str(result['pageid']), {})
                
                # Clean and process the text
                full_text = page_data.get('extract', '')
                snippet = clean_html(result.get('snippet', ''))
                title = result.get('title', '')
                
                # Split long articles into chunks
                chunks = split_into_chunks(full_text, title, max_chunk_size=500)
                
                # Add each chunk as a separate passage
                for j, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                        
                    passage = {
                        'long_text': chunk,
                        'text': chunk,
                        'title': title,
                        'snippet': snippet,
                        'url': page_data.get('fullurl', ''),
                        'pageid': result['pageid'],
                        'score': 1.0 - (i * 0.1) - (j * 0.01),  # Decreasing score
                        'chunk_id': j,
                        'wordcount': result.get('wordcount', 0),
                        'size': result.get('size', 0)
                    }
                    
                    passages.append(passage)
                    
                    # Limit total passages
                    if len(passages) >= limit:
                        break
                
                if len(passages) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️  Failed to get extract for page {result['pageid']}: {e}")
                continue
        
        print(f"✅ Retrieved {len(passages)} Wikipedia passages")
        return passages[:limit]
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Wikipedia API request failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Wikipedia search error: {e}")
        raise


def clean_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    import html
    text = html.unescape(text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def split_into_chunks(text: str, title: str, max_chunk_size: int = 500) -> List[str]:
    """Split long text into smaller chunks."""
    if not text:
        return []
    
    # If text is short enough, return as single chunk
    if len(text.split()) <= max_chunk_size:
        return [f"{title} | {text}"]
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed limit, save current chunk
        if current_chunk and len((current_chunk + " " + paragraph).split()) > max_chunk_size:
            if current_chunk:
                chunks.append(f"{title} | {current_chunk}")
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += " " + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(f"{title} | {current_chunk}")
    
    # If we still have very long chunks, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split by sentences
            sentences = re.split(r'[.!?]+', chunk)
            current_sentence_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if current_sentence_chunk and len((current_sentence_chunk + ". " + sentence).split()) > max_chunk_size:
                    if current_sentence_chunk:
                        final_chunks.append(current_sentence_chunk + ".")
                    current_sentence_chunk = sentence
                else:
                    if current_sentence_chunk:
                        current_sentence_chunk += ". " + sentence
                    else:
                        current_sentence_chunk = sentence
            
            if current_sentence_chunk:
                final_chunks.append(current_sentence_chunk + ".")
    
    return final_chunks if final_chunks else [f"{title} | {text[:1000]}..."]


@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def wikipedia_search_cached(*args, **kwargs):
    """Cached wrapper for wikipedia_search."""
    return wikipedia_search(*args, **kwargs)


# Alternative implementation that mimics ColBERT's interface
class WikipediaColBERTv2:
    """Drop-in replacement for ColBERTv2 using Wikipedia API."""
    
    def __init__(
        self,
        language: str = "en",
        user_agent: str = "TreeOfClarifications/1.0",
        **kwargs  # Accept any additional kwargs for compatibility
    ):
        """
        Initialize Wikipedia retriever with ColBERT-compatible interface.
        
        Args:
            language: Wikipedia language code
            user_agent: User agent string for API requests
        """
        self.retriever = WikipediaRetriever(language, user_agent)
        
    def __call__(
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        """
        Retrieve passages using Wikipedia API with ColBERT-compatible interface.
        """
        return self.retriever(query, k, simplify)