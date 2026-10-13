import tempfile
import pytest

import vivvy
from vivvy.embeddings import DeterministicEmbedder, Embedder


def test_store_properties():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivvy.MemoryStore.open(tmpdir, dimensions=8, embedding_model="my-test-model")
        assert store.dimensions == 8
        assert store.embedding_model == "my-test-model"
        assert store.embedder is None


def test_store_open_with_embedder():
    with tempfile.TemporaryDirectory() as tmpdir:
        emb = DeterministicEmbedder(name="auto-model", dimensions=12)

        # Inferred dimensions and embedding_model
        store = vivvy.MemoryStore.open(tmpdir, embedder=emb)
        assert store.dimensions == 12
        assert store.embedding_model == "auto-model"
        assert store.embedder is emb


def test_store_open_with_mismatched_embedder():
    with tempfile.TemporaryDirectory() as tmpdir:
        emb = DeterministicEmbedder(name="model-x", dimensions=8)

        # Explicit model name mismatch
        with pytest.raises(ValueError, match="Embedding model mismatch"):
            vivvy.MemoryStore.open(tmpdir, dimensions=8, embedding_model="model-y", embedder=emb)

        # Explicit dimension mismatch
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            vivvy.MemoryStore.open(tmpdir, dimensions=16, embedding_model="model-x", embedder=emb)


def test_remember_and_recall_text_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        emb = DeterministicEmbedder(name="test-emb", dimensions=8)
        store = vivvy.MemoryStore.open(tmpdir, embedder=emb)

        # 1. remember_text
        mem_id = store.remember_text(
            tenant_id="tenant-a",
            namespace="kb",
            content="PostgreSQL is an object-relational database system.",
            kind="fact",
            importance=0.9,
        )
        assert mem_id is not None

        # Verify record was stored with proper text and revision
        rec = store.get("tenant-a", "kb", mem_id)
        assert rec is not None
        assert rec["content"] == "PostgreSQL is an object-relational database system."

        # 2. recall_text (hybrid vector + lexical)
        hits = store.recall_text(
            tenant_id="tenant-a",
            namespace="kb",
            query_text="relational database system",
            limit=5,
        )
        assert len(hits) == 1
        assert hits[0][0] == mem_id
        assert "PostgreSQL" in hits[0][1]

        # 3. format_context_text
        ctx = store.format_context_text(
            tenant_id="tenant-a",
            namespace="kb",
            query_text="PostgreSQL database",
            max_tokens=500,
        )
        assert "PostgreSQL" in ctx


def test_remember_text_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        emb = DeterministicEmbedder(name="batch-model", dimensions=6)
        store = vivvy.MemoryStore.open(tmpdir, embedder=emb)

        records = [
            {
                "tenant_id": "org1",
                "namespace": "docs",
                "content": "Doc 1 about python",
                "kind": "fact",
            },
            {
                "tenant_id": "org1",
                "namespace": "docs",
                "content": "Doc 2 about rust",
                "kind": "fact",
            },
        ]

        ids = store.remember_text_batch(records)
        assert len(ids) == 2

        hits = store.recall_text("org1", "docs", "python", limit=5)
        assert any("Doc 1" in h[1] for h in hits)


def test_missing_embedder_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivvy.MemoryStore.open(tmpdir, dimensions=4, embedding_model="model-1")

        with pytest.raises(ValueError, match="No embedder provided"):
            store.remember_text("t1", "ns1", "Some content")

        with pytest.raises(ValueError, match="No embedder provided"):
            store.recall_text("t1", "ns1", "Some query")


def test_dimension_mismatch_guards():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Store expects 8 dims
        store = vivvy.MemoryStore.open(tmpdir, dimensions=8, embedding_model="model-1")

        # 1. Embedder declared dimension mismatch
        emb_wrong_decl = DeterministicEmbedder(name="model-1", dimensions=4)
        with pytest.raises(ValueError, match="Embedding dimension mismatch: embedder declared 4 dimensions"):
            store.remember_text("t1", "ns1", "text", embedder=emb_wrong_decl)

        # 2. Embedder produced vector dimension mismatch
        class BadVectorEmbedder(Embedder):
            @property
            def name(self) -> str:
                return "model-1"

            @property
            def dimensions(self) -> int:
                return 8  # Claims 8

            def embed(self, text: str):
                return [1.0, 2.0]  # Actually returns 2

        bad_emb = BadVectorEmbedder()
        with pytest.raises(ValueError, match="Embedding dimension mismatch: embedder produced 2 dimensions"):
            store.remember_text("t1", "ns1", "text", embedder=bad_emb)


def test_refuse_to_mix_models_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivvy.MemoryStore.open(tmpdir, dimensions=4, embedding_model="model-A")

        # Calling remember_text with a different model must fail
        emb_b = DeterministicEmbedder(name="model-B", dimensions=4)
        with pytest.raises(ValueError, match="Embedding model mismatch: embedder model 'model-B' does not match store embedding_model 'model-A'"):
            store.remember_text("t1", "ns1", "text", embedder=emb_b)

        # Setting store.embedder with different model must fail
        with pytest.raises(ValueError, match="Embedding model mismatch"):
            store.embedder = emb_b


def test_refuse_to_mix_models_on_store_reopen():
    """Verify that an existing store refuses to open if embedding_model or dimensions differ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create and store an item with model-A (dims 4)
        emb_a = DeterministicEmbedder(name="model-A", dimensions=4)
        store1 = vivvy.MemoryStore.open(tmpdir, embedder=emb_a)
        store1.remember_text("t1", "ns1", "Persistent fact")
        del store1

        # 2. Reopen with different model-B -> must fail
        with pytest.raises(ValueError, match="refuse to mix incompatible embedding models"):
            vivvy.MemoryStore.open(tmpdir, dimensions=4, embedding_model="model-B")

        # 3. Reopen with different dimensions -> must fail
        with pytest.raises(ValueError, match="refuse to mix incompatible dimensions"):
            vivvy.MemoryStore.open(tmpdir, dimensions=8, embedding_model="model-A")

        # 4. Reopen with matching model-A -> succeeds
        store2 = vivvy.MemoryStore.open(tmpdir, embedder=emb_a)
        hits = store2.recall_text("t1", "ns1", "Persistent fact")
        assert len(hits) == 1
        assert "Persistent fact" in hits[0][1]
