"""
Simple text-based reranker using TF-IDF similarity without neural models.
This avoids all threading/mutex issues with PyTorch/Transformers.
"""

import re
import math
from collections import Counter
from typing import List, Dict


class SimpleTFIDFReranker:
    """Simple TF-IDF based reranker that doesn't use neural models."""
    
    def __init__(self):
        """Initialize the reranker."""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
        }
    
    def _tokenize(self, text) -> List[str]:
        """Tokenize text into words."""
        # Handle both string and dict/dotdict objects
        if isinstance(text, dict):
            # Try common text fields
            text_content = text.get('long_text') or text.get('text') or text.get('content') or str(text)
        elif hasattr(text, 'long_text'):
            # Handle dotdict objects
            text_content = text.long_text
        elif hasattr(text, 'text'):
            text_content = text.text
        else:
            # Assume it's a string
            text_content = str(text)
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text_content.lower())
        # Remove stop words
        return [word for word in words if word not in self.stop_words and len(word) > 2]
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf = Counter(tokens)
        total_terms = len(tokens)
        return {term: count / total_terms for term, count in tf.items()}
    
    def _compute_idf(self, documents: List[List[str]]) -> Dict[str, float]:
        """Compute inverse document frequency."""
        total_docs = len(documents)
        all_terms = set()
        for doc in documents:
            all_terms.update(doc)
        
        idf = {}
        for term in all_terms:
            docs_with_term = sum(1 for doc in documents if term in doc)
            idf[term] = math.log(total_docs / docs_with_term) if docs_with_term > 0 else 0
        
        return idf
    
    def _compute_tfidf_vector(self, tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """Compute TF-IDF vector."""
        return {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two vectors."""
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def rerank(self, query: str, passages: List, k: int) -> List:
        """
        Rerank passages based on TF-IDF similarity to query.
        
        Args:
            query: Search query
            passages: List of passages to rerank (can be strings or dict/dotdict objects)
            k: Number of top passages to return
            
        Returns:
            List of reranked passages (same type as input)
        """
        if not passages:
            return []
        
        # Tokenize query and passages
        query_tokens = self._tokenize(query)
        passage_tokens = [self._tokenize(passage) for passage in passages]
        
        # Compute IDF for all documents (query + passages)
        all_documents = [query_tokens] + passage_tokens
        idf = self._compute_idf(all_documents)
        
        # Compute TF-IDF for query
        query_tf = self._compute_tf(query_tokens)
        query_tfidf = self._compute_tfidf_vector(query_tf, idf)
        
        # Compute similarities
        similarities = []
        for i, tokens in enumerate(passage_tokens):
            passage_tf = self._compute_tf(tokens)
            passage_tfidf = self._compute_tfidf_vector(passage_tf, idf)
            similarity = self._cosine_similarity(query_tfidf, passage_tfidf)
            similarities.append((similarity, i, passages[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k passages (preserving original format)
        reranked = [passage for _, _, passage in similarities[:k]]
        
        print(f"✅ TF-IDF reranked {len(passages)} passages, returning top {len(reranked)}")
        return reranked
    
    def __call__(self, query: str, passages: List) -> List[float]:
        """
        Score passages based on TF-IDF similarity to query (for compatibility with DSP reranker interface).
        
        Args:
            query: Search query
            passages: List of passages to score
            
        Returns:
            List of similarity scores
        """
        if not passages:
            return []
        
        # Tokenize query and passages
        query_tokens = self._tokenize(query)
        passage_tokens = [self._tokenize(passage) for passage in passages]
        
        # Compute IDF for all documents (query + passages)
        all_documents = [query_tokens] + passage_tokens
        idf = self._compute_idf(all_documents)
        
        # Compute TF-IDF for query
        query_tf = self._compute_tf(query_tokens)
        query_tfidf = self._compute_tfidf_vector(query_tf, idf)
        
        # Compute similarities
        similarities = []
        for tokens in passage_tokens:
            passage_tf = self._compute_tf(tokens)
            passage_tfidf = self._compute_tfidf_vector(passage_tf, idf)
            similarity = self._cosine_similarity(query_tfidf, passage_tfidf)
            similarities.append(similarity)
        
        return similarities


def test_simple_reranker():
    """Test the simple TF-IDF reranker."""
    print("🧪 Testing Simple TF-IDF Reranker")
    print("=" * 40)
    
    try:
        reranker = SimpleTFIDFReranker()
        
        query = "machine learning algorithms"
        passages = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Cooking recipes often involve multiple steps and ingredients.",
            "Neural networks are a type of machine learning model inspired by the brain.",
            "Weather patterns can be predicted using meteorological data.",
            "Deep learning uses multiple layers to learn complex patterns in data."
        ]
        
        print(f"Query: {query}")
        print(f"Original passages: {len(passages)}")
        
        reranked = reranker.rerank(query, passages, k=3)
        
        print(f"Reranked top 3:")
        for i, passage in enumerate(reranked, 1):
            print(f"{i}. {passage[:80]}...")
        
        print("✅ Simple TF-IDF reranker test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Simple reranker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_simple_reranker()