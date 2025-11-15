class SentenceTransformersCrossEncoder:
    """Wrapper for sentence-transformers cross-encoder model.
    """
    def __init__(
        self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError:
            raise ModuleNotFoundError(
                "You need to install sentence-transformers library to use SentenceTransformersCrossEncoder."
            )
        self.model = CrossEncoder(model_name_or_path)

    def __call__(self, query: str, passage: list[str]) -> list[float]:
        # Extract text content from passages (handle both strings and objects)
        text_passages = []
        for p in passage:
            if isinstance(p, str):
                text_passages.append(p)
            elif isinstance(p, dict):
                text_passages.append(p.get('long_text') or p.get('text') or str(p))
            elif hasattr(p, 'long_text'):
                text_passages.append(p.long_text)
            elif hasattr(p, 'text'):
                text_passages.append(p.text)
            else:
                text_passages.append(str(p))

        return self.model.predict([[query, p] for p in text_passages]).tolist()
