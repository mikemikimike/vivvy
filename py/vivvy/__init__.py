"""vivvy: Single-machine vector search and durable long-term memory for AI agents."""

from typing import Any, List, Optional, Sequence, Tuple, Union

from ._vivvy import Index, MemoryStore
from .embeddings import (
    DeterministicEmbedder,
    Embedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
)


def _validate_embedder(store: MemoryStore, emb: Embedder) -> None:
    if not isinstance(emb, Embedder):
        raise TypeError(f"Expected an Embedder instance, got {type(emb).__name__}")
    if emb.name != store.embedding_model:
        raise ValueError(
            f"Embedding model mismatch: embedder model '{emb.name}' does not match "
            f"store embedding_model '{store.embedding_model}'. Refusing to mix incompatible embedding models."
        )
    if emb.dimensions != store.dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: embedder declared {emb.dimensions} dimensions, "
            f"but store expects {store.dimensions} dimensions."
        )


# Preserve original Rust MemoryStore.open
_rust_open = MemoryStore.open


@staticmethod
def _open(
    path: str,
    dimensions: Optional[int] = None,
    embedding_model: Optional[str] = None,
    max_recall_limit: int = 100,
    *,
    embedder: Optional[Embedder] = None,
) -> MemoryStore:
    """Open or create a durable MemoryStore.

    Args:
        path: Directory path for storing `memory.db` and vector index files.
        dimensions: Exact dimensionality of vector embeddings. Inferred from `embedder` if omitted.
        embedding_model: Canonical model name identifier. Inferred from `embedder` if omitted.
        max_recall_limit: Upper limit on candidate records returned by recall. Default: 100.
        embedder: Optional `Embedder` adapter to bind to this store.
    """
    if embedder is not None:
        if not isinstance(embedder, Embedder):
            raise TypeError(f"Expected an Embedder instance, got {type(embedder).__name__}")
        if dimensions is None:
            dimensions = embedder.dimensions
        elif embedder.dimensions != dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: embedder declared {embedder.dimensions} dimensions, "
                f"but store expects {dimensions} dimensions."
            )
        if embedding_model is None:
            embedding_model = embedder.name
        elif embedder.name != embedding_model:
            raise ValueError(
                f"Embedding model mismatch: embedder model '{embedder.name}' does not match "
                f"store embedding_model '{embedding_model}'. Refusing to mix incompatible embedding models."
            )

    if dimensions is None:
        raise ValueError("dimensions must be specified or an embedder must be provided")
    if embedding_model is None:
        raise ValueError("embedding_model must be specified or an embedder must be provided")

    store = _rust_open(path, dimensions, embedding_model, max_recall_limit)

    if embedder is not None:
        store.embedder = embedder

    return store


_orig_embedder_descr = MemoryStore.embedder


def _get_embedder(self: MemoryStore) -> Optional[Embedder]:
    return _orig_embedder_descr.__get__(self, MemoryStore)


def _set_embedder(self: MemoryStore, value: Optional[Embedder]) -> None:
    if value is not None:
        _validate_embedder(self, value)
    _orig_embedder_descr.__set__(self, value)


def remember_text(
    self: MemoryStore,
    tenant_id: str,
    namespace: str,
    content: str,
    kind: str = "fact",
    importance: float = 0.5,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    expires_at_ms: Optional[int] = None,
    metadata: Optional[dict] = None,
    *,
    embedder: Optional[Embedder] = None,
) -> str:
    """Embed text content using the configured embedder and store into durable memory.

    Fails with a clear ValueError if the embedder model or dimensions do not match the store.
    """
    emb = embedder if embedder is not None else self.embedder
    if emb is None:
        raise ValueError(
            "No embedder provided. Pass embedder=... or configure an embedder on the store."
        )
    _validate_embedder(self, emb)

    vec = emb.embed(content)
    if len(vec) != self.dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: embedder produced {len(vec)} dimensions, "
            f"but store expects {self.dimensions} dimensions."
        )

    return self.remember(
        tenant_id=tenant_id,
        namespace=namespace,
        content=content,
        embedding=vec,
        kind=kind,
        importance=importance,
        agent_id=agent_id,
        user_id=user_id,
        operation_id=operation_id,
        expires_at_ms=expires_at_ms,
        metadata=metadata,
    )


def remember_text_batch(
    self: MemoryStore,
    records: Sequence[dict],
    *,
    embedder: Optional[Embedder] = None,
) -> List[str]:
    """Batch-embed and remember multiple text records.

    Each record in `records` must have 'tenant_id', 'namespace', and 'content'.
    """
    if not records:
        return []

    emb = embedder if embedder is not None else self.embedder
    if emb is None:
        raise ValueError(
            "No embedder provided. Pass embedder=... or configure an embedder on the store."
        )
    _validate_embedder(self, emb)

    texts = [r["content"] for r in records]
    vecs = emb.embed_batch(texts)

    prepared_records = []
    for r, vec in zip(records, vecs):
        if len(vec) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: embedder produced {len(vec)} dimensions, "
                f"but store expects {self.dimensions} dimensions."
            )
        rec_copy = dict(r)
        rec_copy["embedding"] = vec
        prepared_records.append(rec_copy)

    return self.remember_batch(prepared_records)


def recall_text(
    self: MemoryStore,
    tenant_id: str,
    namespace: str,
    query_text: str,
    limit: int = 5,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    include_explanations: bool = True,
    mmr_lambda: Optional[float] = None,
    filter_metadata: Optional[dict] = None,
    created_after_ms: Optional[int] = None,
    created_before_ms: Optional[int] = None,
    *,
    embedder: Optional[Embedder] = None,
) -> List[Tuple[str, str, float]]:
    """Embed query text and perform hybrid recall (dense vector + FTS5 lexical search).

    Fails with a clear ValueError if the embedder model or dimensions do not match the store.
    """
    emb = embedder if embedder is not None else self.embedder
    if emb is None:
        raise ValueError(
            "No embedder provided. Pass embedder=... or configure an embedder on the store."
        )
    _validate_embedder(self, emb)

    query_vec = emb.embed(query_text)
    if len(query_vec) != self.dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: embedder produced {len(query_vec)} dimensions, "
            f"but store expects {self.dimensions} dimensions."
        )

    return self.recall(
        tenant_id=tenant_id,
        namespace=namespace,
        query_embedding=query_vec,
        query_text=query_text,
        limit=limit,
        agent_id=agent_id,
        user_id=user_id,
        include_explanations=include_explanations,
        mmr_lambda=mmr_lambda,
        filter_metadata=filter_metadata,
        created_after_ms=created_after_ms,
        created_before_ms=created_before_ms,
    )


def format_context_text(
    self: MemoryStore,
    tenant_id: str,
    namespace: str,
    query_text: str,
    max_tokens: int = 1500,
    limit: int = 10,
    template: Optional[str] = None,
    header: Optional[str] = None,
    footer: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    filter_metadata: Optional[dict] = None,
    created_after_ms: Optional[int] = None,
    created_before_ms: Optional[int] = None,
    *,
    embedder: Optional[Embedder] = None,
) -> str:
    """Embed query text and format retrieved memories into a prompt context within a token budget."""
    emb = embedder if embedder is not None else self.embedder
    if emb is None:
        raise ValueError(
            "No embedder provided. Pass embedder=... or configure an embedder on the store."
        )
    _validate_embedder(self, emb)

    query_vec = emb.embed(query_text)
    if len(query_vec) != self.dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: embedder produced {len(query_vec)} dimensions, "
            f"but store expects {self.dimensions} dimensions."
        )

    return self.format_context(
        tenant_id=tenant_id,
        namespace=namespace,
        query_embedding=query_vec,
        query_text=query_text,
        max_tokens=max_tokens,
        limit=limit,
        template=template,
        header=header,
        footer=footer,
        agent_id=agent_id,
        user_id=user_id,
        filter_metadata=filter_metadata,
        created_after_ms=created_after_ms,
        created_before_ms=created_before_ms,
    )


# Attach methods to MemoryStore class
MemoryStore.open = _open
MemoryStore.embedder = property(_get_embedder, _set_embedder)
MemoryStore.remember_text = remember_text
MemoryStore.remember_text_batch = remember_text_batch
MemoryStore.recall_text = recall_text
MemoryStore.format_context_text = format_context_text

__all__ = [
    "Index",
    "MemoryStore",
    "Embedder",
    "OpenAIEmbedder",
    "SentenceTransformersEmbedder",
    "OllamaEmbedder",
    "DeterministicEmbedder",
]
