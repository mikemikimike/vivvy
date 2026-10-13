import tempfile
import pytest
import vivvy
from vivvy import ScoringConfig, DeterministicEmbedder


def test_scoring_config_defaults():
    cfg = ScoringConfig()
    assert cfg.weight_similarity == 0.70
    assert cfg.weight_importance == 0.15
    assert cfg.weight_recency == 0.10
    assert cfg.weight_reinforcement == 0.05
    assert cfg.recency_half_life_days == 7.0
    d = cfg.to_dict()
    assert d["weight_similarity"] == 0.70


def test_recall_with_none_or_empty_dict_uses_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivvy.MemoryStore.open(tmpdir, 3, "test-model")
        store.remember(
            tenant_id="t",
            namespace="n",
            content="fact 1",
            embedding=[1.0, 0.0, 0.0],
            kind="fact",
            importance=0.5,
        )

        res_none = store.recall(
            tenant_id="t",
            namespace="n",
            query_embedding=[1.0, 0.0, 0.0],
            scoring_weights=None,
        )
        assert len(res_none) == 1

        res_empty = store.recall(
            tenant_id="t",
            namespace="n",
            query_embedding=[1.0, 0.0, 0.0],
            scoring_weights={},
        )
        assert len(res_empty) == 1
        assert abs(res_none[0][2] - res_empty[0][2]) < 1e-6


def test_recall_with_custom_scoring_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivvy.MemoryStore.open(tmpdir, 3, "test-model")
        store.remember(
            tenant_id="t",
            namespace="n",
            content="fact 1",
            embedding=[1.0, 0.0, 0.0],
            kind="fact",
            importance=0.5,
        )

        # Pure similarity: sim=1.0 -> total_score = 1.0 * 1.0 = 1.0
        res = store.recall(
            tenant_id="t",
            namespace="n",
            query_embedding=[1.0, 0.0, 0.0],
            scoring_weights={
                "weight_similarity": 1.0,
                "weight_importance": 0.0,
                "weight_recency": 0.0,
                "weight_reinforcement": 0.0,
            },
        )
        assert len(res) == 1
        assert abs(res[0][2] - 1.0) < 1e-4

        # Pure importance: importance=0.5 -> total_score = 1.0 * 0.5 = 0.5
        res_imp = store.recall(
            tenant_id="t",
            namespace="n",
            query_embedding=[1.0, 0.0, 0.0],
            scoring_weights={
                "weight_similarity": 0.0,
                "weight_importance": 1.0,
                "weight_recency": 0.0,
                "weight_reinforcement": 0.0,
            },
        )
        assert len(res_imp) == 1
        assert abs(res_imp[0][2] - 0.5) < 1e-4


def test_recall_rejects_negative_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivvy.MemoryStore.open(tmpdir, 3, "test-model")
        with pytest.raises(ValueError, match="must be non-negative"):
            store.recall(
                tenant_id="t",
                namespace="n",
                query_embedding=[1.0, 0.0, 0.0],
                scoring_weights={"weight_similarity": -0.5},
            )


def test_recall_text_and_format_context_text_with_scoring():
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = DeterministicEmbedder("test-model", dimensions=4)
        store = vivvy.MemoryStore.open(tmpdir, embedder=embedder)

        store.remember_text(
            tenant_id="t",
            namespace="n",
            content="Important guideline",
            importance=0.9,
        )
        store.remember_text(
            tenant_id="t",
            namespace="n",
            content="Minor note",
            importance=0.1,
        )

        custom_scoring = ScoringConfig(
            weight_similarity=0.1,
            weight_importance=0.9,
            weight_recency=0.0,
            weight_reinforcement=0.0,
        )

        res = store.recall_text(
            tenant_id="t",
            namespace="n",
            query_text="Important guideline",
            limit=2,
            scoring=custom_scoring,
        )
        assert len(res) == 2
        # Important guideline should rank first because importance weight is 0.9
        assert res[0][1] == "Important guideline"

        ctx = store.format_context_text(
            tenant_id="t",
            namespace="n",
            query_text="Important guideline",
            scoring=custom_scoring,
        )
        assert "Important guideline" in ctx
