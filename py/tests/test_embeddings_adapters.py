import sys
from unittest.mock import MagicMock, patch
import pytest

import vivvy
from vivvy.embeddings import (
    DeterministicEmbedder,
    Embedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
)


def test_embedder_abstract_interface():
    class IncompleteEmbedder(Embedder):
        pass

    with pytest.raises(TypeError):
        IncompleteEmbedder()


def test_deterministic_embedder():
    emb = DeterministicEmbedder("test-model", dimensions=16)
    assert emb.name == "test-model"
    assert emb.dimensions == 16

    v1 = emb.embed("hello world")
    v2 = emb.embed("hello world")
    v3 = emb.embed("different content")

    assert len(v1) == 16
    assert v1 == v2
    assert v1 != v3

    # Batched embedding
    batch = emb.embed_batch(["hello world", "different content"])
    assert len(batch) == 2
    assert batch[0] == v1
    assert batch[1] == v3

    # Invalid dimensions
    with pytest.raises(ValueError, match="dimensions must be positive"):
        DeterministicEmbedder("test", dimensions=0)


def test_openai_embedder_missing_package():
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match="The 'openai' package is required"):
            OpenAIEmbedder()


def test_openai_embedder_with_mock_client():
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_item1 = MagicMock()
    mock_item1.embedding = [0.1, 0.2, 0.3]
    mock_item1.index = 0
    mock_item2 = MagicMock()
    mock_item2.embedding = [0.4, 0.5, 0.6]
    mock_item2.index = 1

    mock_resp_single = MagicMock()
    mock_resp_single.data = [mock_item1]

    mock_resp_batch = MagicMock()
    mock_resp_batch.data = [mock_item2, mock_item1]  # reverse order to test index sorting

    mock_client.embeddings.create.side_effect = [mock_resp_single, mock_resp_batch]

    with patch.dict(sys.modules, {"openai": mock_openai}):
        emb = OpenAIEmbedder(
            model="text-embedding-3-small",
            dimensions=3,
            client=mock_client,
        )
        assert emb.name == "text-embedding-3-small"
        assert emb.dimensions == 3

        res_single = emb.embed("query 1")
        assert res_single == [0.1, 0.2, 0.3]

        res_batch = emb.embed_batch(["query 1", "query 2"])
        assert res_batch == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_sentence_transformers_missing_package():
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(ImportError, match="The 'sentence-transformers' package is required"):
            SentenceTransformersEmbedder()


def test_sentence_transformers_with_mock_model():
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 4
    
    mock_ndarray_single = MagicMock()
    mock_ndarray_single.tolist.return_value = [0.1, 0.2, 0.3, 0.4]

    mock_ndarray_batch = MagicMock()
    mock_ndarray_batch.tolist.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

    mock_model.encode.side_effect = [mock_ndarray_single, mock_ndarray_batch]

    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        emb = SentenceTransformersEmbedder(
            model_name_or_path="all-MiniLM-L6-v2",
            model=mock_model,
        )
        assert emb.name == "all-MiniLM-L6-v2"
        assert emb.dimensions == 4

        res_single = emb.embed("doc 1")
        assert res_single == [0.1, 0.2, 0.3, 0.4]

        res_batch = emb.embed_batch(["doc 1", "doc 2"])
        assert res_batch == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


def test_ollama_embedder_with_mock_http():
    import json
    import io

    emb = OllamaEmbedder(model="nomic-embed-text", dimensions=3)
    assert emb.name == "nomic-embed-text"
    assert emb.dimensions == 3

    mock_response = io.BytesIO(json.dumps({"embedding": [0.2, 0.4, 0.6]}).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=mock_response):
        vec = emb.embed("local prompt")
        assert vec == [0.2, 0.4, 0.6]
