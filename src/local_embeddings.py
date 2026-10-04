"""Reuse the multilingual retrieval encoder for RAGAS without an API call."""
from langchain_core.embeddings import Embeddings
from src.m2_search import shared_sentence_encoder


class LocalMultilingualEmbeddings(Embeddings):
    def embed_documents(self, texts):
        if not texts:
            return []
        return shared_sentence_encoder().encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return shared_sentence_encoder().encode(text, normalize_embeddings=True).tolist()
