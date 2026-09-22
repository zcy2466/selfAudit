"""Utilities: logging, metrics, JSON parsing, dataset loading."""

from app.utils.logger import logger
from app.utils.value_parser import ValueParser
from app.utils.metrics import (
    compute_accuracy,
    compute_f1,
    compute_class_f1,
    compute_weighted_f1,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_aupr,
    compute_cohens_kappa,
    compute_cv,
    compute_token_f1,
    ErrorAttributionTracker,
)
from app.utils.dataset_utils import (
    load_contractnli,
    load_cuad,
    load_legalbench_rag,
    split_dataset,
)

__all__ = [
    "logger",
    "ValueParser",
    "compute_accuracy",
    "compute_f1",
    "compute_class_f1",
    "compute_weighted_f1",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_aupr",
    "compute_cohens_kappa",
    "compute_cv",
    "compute_token_f1",
    "ErrorAttributionTracker",
    "load_contractnli",
    "load_cuad",
    "load_legalbench_rag",
    "split_dataset",
]