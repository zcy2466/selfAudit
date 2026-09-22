"""Tests for evaluation metrics."""

import pytest


class TestMetrics:
    def test_accuracy(self):
        from app.utils.metrics import compute_accuracy
        preds = ["Entailment", "Contradiction", "Not Mentioned"]
        golds = ["Entailment", "Contradiction", "Entailment"]
        acc = compute_accuracy(preds, golds)
        assert acc == pytest.approx(2 / 3)

    def test_accuracy_perfect(self):
        from app.utils.metrics import compute_accuracy
        preds = ["Entailment", "Contradiction", "Not Mentioned"]
        golds = ["Entailment", "Contradiction", "Not Mentioned"]
        assert compute_accuracy(preds, golds) == 1.0

    def test_weighted_f1(self):
        from app.utils.metrics import compute_weighted_f1
        preds = ["Entailment", "Entailment", "Contradiction"]
        golds = ["Entailment", "Contradiction", "Contradiction"]
        f1 = compute_weighted_f1(preds, golds)
        assert 0.0 <= f1 <= 1.0

    def test_precision_at_k(self):
        from app.utils.metrics import compute_precision_at_k
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        p1 = compute_precision_at_k(retrieved, relevant, 1)
        p4 = compute_precision_at_k(retrieved, relevant, 4)
        assert p1 == 1.0
        assert p4 == 0.5

    def test_recall_at_k(self):
        from app.utils.metrics import compute_recall_at_k
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "e"}
        r2 = compute_recall_at_k(retrieved, relevant, 2)
        r4 = compute_recall_at_k(retrieved, relevant, 4)
        assert r2 == pytest.approx(1 / 3)
        assert r4 == pytest.approx(2 / 3)

    def test_cohens_kappa(self):
        from app.utils.metrics import compute_cohens_kappa
        judge1 = [0, 0, 1, 1, 2]
        judge2 = [0, 0, 1, 1, 2]
        kappa = compute_cohens_kappa(judge1, judge2)
        assert kappa == 1.0

    def test_cv(self):
        from app.utils.metrics import compute_cv
        values = [10.0, 10.0, 10.0]
        assert compute_cv(values) == 0.0
        values2 = [5.0, 15.0]
        assert compute_cv(values2) > 0.0


class TestErrorAttributionTracker:
    def test_tracker_accuracy(self):
        from app.utils.metrics import ErrorAttributionTracker
        tracker = ErrorAttributionTracker()
        tracker.record_case("era", "era", {"rfs": 0.5, "ess": 0.8, "rcs": 0.9})
        tracker.record_case("tpa", "era", {"rfs": 0.9, "ess": 0.5, "rcs": 0.8})
        assert tracker.get_accuracy() == 0.5
        assert tracker.total_cases == 2

    def test_confidence_stats(self):
        from app.utils.metrics import ErrorAttributionTracker
        tracker = ErrorAttributionTracker()
        tracker.record_case("era", "era", {"rfs": 0.8, "ess": 0.6, "rcs": 0.7})
        tracker.record_case("era", "era", {"rfs": 0.9, "ess": 0.8, "rcs": 0.5})
        stats = tracker.get_confidence_stats()
        assert "rfs" in stats
        assert stats["rfs"]["mean"] == pytest.approx(0.85)

    def test_bottleneck_proportions(self):
        from app.utils.metrics import ErrorAttributionTracker
        tracker = ErrorAttributionTracker()
        tracker.record_case("era", "era", {"rfs": 0.9, "ess": 0.5, "rcs": 0.8})
        tracker.record_case("era", "era", {"rfs": 0.8, "ess": 0.9, "rcs": 0.3})
        props = tracker.get_bottleneck_proportions()
        assert props["ess"] == 0.5
        assert props["rcs"] == 0.5