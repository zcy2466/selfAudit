"""Evaluation metrics for SelfAudit experiments."""

import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support


def compute_accuracy(predictions: list[str], gold_labels: list[str]) -> float:
    """Compute exact-match accuracy."""
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    return correct / len(gold_labels) if gold_labels else 0.0


def compute_f1(predictions: list[str], gold_labels: list[str], average: str = "weighted") -> float:
    """Compute F1 score with specified averaging."""
    labels = sorted(set(gold_labels) | set(predictions))
    return f1_score(gold_labels, predictions, labels=labels, average=average, zero_division=0)


def compute_class_f1(predictions: list[str], gold_labels: list[str], class_label: str) -> float:
    """Compute per-class F1 score."""
    labels = sorted(set(gold_labels) | set(predictions))
    f1_scores = f1_score(gold_labels, predictions, labels=labels, average=None, zero_division=0)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    if class_label in label_to_idx:
        return f1_scores[label_to_idx[class_label]]
    return 0.0


def compute_weighted_f1(predictions: list[str], gold_labels: list[str]) -> float:
    """Compute support-weighted F1 (F1[W] as defined in the paper)."""
    return compute_f1(predictions, gold_labels, average="weighted")


def compute_precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(top_k)


def compute_recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Recall@k: fraction of all relevant items retrieved in top-k."""
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 1.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)


def compute_aupr(precision: list[float], recall: list[float]) -> float:
    """Area under the precision-recall curve via trapezoidal integration."""
    if len(precision) < 2:
        return 0.0
    recall = np.array(recall)
    precision = np.array(precision)
    sorted_idx = np.argsort(recall)
    recall = recall[sorted_idx]
    precision = precision[sorted_idx]
    return float(np.trapz(precision, recall))


def compute_cohens_kappa(judge1: list[int], judge2: list[int]) -> float:
    """Cohen's kappa for inter-rater agreement."""
    return cohen_kappa_score(judge1, judge2)


def compute_cv(values: list[float]) -> float:
    """Coefficient of variation (std / mean) as percentage."""
    arr = np.array(values)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean * 100)


def compute_token_f1(
    pred_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
    total_tokens: int,
) -> dict[str, float]:
    """Token-level F1 for span extraction tasks (CUAD)."""
    pred_tokens = set()
    gold_tokens = set()
    for start, end in pred_spans:
        pred_tokens.update(range(start, end))
    for start, end in gold_spans:
        gold_tokens.update(range(start, end))
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class ErrorAttributionTracker:
    """Track error attribution accuracy for dependency-aware experiments."""

    def __init__(self):
        self.total_cases = 0
        self.correct_attributions = 0
        self.earliest_node_correct = 0
        self.lowest_confidence_correct = 0
        self.confidence_log: list[dict] = []

    def record_case(
        self,
        true_source: str,
        attributed_source: str,
        confidence_3d: dict[str, float],
    ):
        self.total_cases += 1
        if attributed_source == true_source:
            self.correct_attributions += 1
        lowest_dim = min(confidence_3d, key=confidence_3d.get)
        if lowest_dim == true_source:
            self.lowest_confidence_correct += 1
        self.confidence_log.append({
            "true_source": true_source,
            "attributed": attributed_source,
            "rfs": confidence_3d.get("rfs", 0),
            "ess": confidence_3d.get("ess", 0),
            "rcs": confidence_3d.get("rcs", 0),
        })

    def get_accuracy(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.correct_attributions / self.total_cases

    def get_lowest_confidence_baseline(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.lowest_confidence_correct / self.total_cases

    def get_confidence_stats(self) -> dict:
        """Compute mean and std for each confidence dimension."""
        stats = defaultdict(list)
        for entry in self.confidence_log:
            for dim in ["rfs", "ess", "rcs"]:
                stats[dim].append(entry.get(dim, 0))
        result = {}
        for dim, values in stats.items():
            arr = np.array(values)
            result[dim] = {"mean": float(arr.mean()), "std": float(arr.std())}
        return result

    def get_bottleneck_proportions(self) -> dict[str, float]:
        """Proportion of cases where each dimension is the sole bottleneck (lowest).

        When there is a tie for the minimum, no dimension is credited,
        ensuring proportions sum to at most 1.0.
        """
        counts = {"rfs": 0, "ess": 0, "rcs": 0}
        total = 0
        for entry in self.confidence_log:
            rfs = entry.get("rfs", 0)
            ess = entry.get("ess", 0)
            rcs = entry.get("rcs", 0)
            min_val = min(rfs, ess, rcs)
            total += 1
            # Only count when there is a single unique minimum
            min_dims = [dim for dim, val in [("rfs", rfs), ("ess", ess), ("rcs", rcs)] if val == min_val]
            if len(min_dims) == 1:
                counts[min_dims[0]] += 1
        if total == 0:
            return {"rfs": 0.0, "ess": 0.0, "rcs": 0.0}
        return {k: v / total for k, v in counts.items()}