"""
Singleton-обгортка над SentenceTransformer.

Чому singleton: модель ~80MB і ~400MB RAM після завантаження. Якщо створювати
інстанс на кожен запит — вб'є latency і пам'ять. Завантажуємо раз при імпорті.

Чому all-MiniLM-L6-v2: 384-dim, швидкий на CPU, безкоштовний, не потребує API key.
Та сама модель використовується для indexing (scripts/index.py) і для запитів
(/chat/stream) — інакше вектори несумісні для cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import settings

# 384 — фіксована розмірність all-MiniLM-L6-v2. Передаємо в Qdrant
# при створенні колекцій. Якщо змінити модель — змінити і це число.
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Завантажується раз на процес. lru_cache гарантує singleton."""
    print(f"[embedder] loading {settings.embedding_model} ...")
    model = SentenceTransformer(settings.embedding_model)
    print(f"[embedder] loaded · dim={model.get_sentence_embedding_dimension()}")
    return model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Batch embedding. Викликати з list, навіть якщо текст один.
    Внутрішньо SentenceTransformer паралелить — batch швидший за per-item.
    """
    model = get_model()
    # normalize_embeddings=True → l2-norm, потрібно для cosine у Qdrant.
    # convert_to_numpy=True → потім .tolist() для JSON-сумісності з qdrant-client.
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 10,
    )
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
