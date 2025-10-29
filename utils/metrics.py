"""
Comprehensive evaluation metrics for credit risk models.
Includes ROC-AUC, PR-AUC, KS, Brier, ECE, and statistical significance tests.
"""
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from scipy import stats


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov (KS) statistic.

    KS measures the maximum separation between cumulative distributions
    of predicted probabilities for positive and negative classes.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class

    Returns:
        KS statistic (0-100 scale)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = (tpr - fpr).max() * 100
    return ks


def compute_expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual outcomes.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return ece


def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary of metric names and values
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'ks': compute_ks_statistic(y_true, y_proba),
        'brier': brier_score_loss(y_true, y_proba),
        'ece': compute_expected_calibration_error(y_true, y_proba),
    }

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) \
                    if (metrics['precision'] + metrics['recall']) > 0 else 0.0

    return metrics


def delong_test(y_true: np.ndarray, y_proba1: np.ndarray, y_proba2: np.ndarray) -> Tuple[float, float]:
    """
    DeLong test for comparing two ROC curves.

    Simplified implementation using bootstrap approximation.

    Args:
        y_true: True binary labels
        y_proba1: Predicted probabilities from model 1
        y_proba2: Predicted probabilities from model 2

    Returns:
        Tuple of (z_statistic, p_value)
    """
    auc1 = roc_auc_score(y_true, y_proba1)
    auc2 = roc_auc_score(y_true, y_proba2)

    # Bootstrap approximation (1000 samples)
    n_bootstrap = 1000
    auc_diffs = []

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            auc1_boot = roc_auc_score(y_true[indices], y_proba1[indices])
            auc2_boot = roc_auc_score(y_true[indices], y_proba2[indices])
            auc_diffs.append(auc1_boot - auc2_boot)
        except ValueError:
            continue

    auc_diffs = np.array(auc_diffs)
    se = auc_diffs.std()
    z_stat = (auc1 - auc2) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray, metric: str = 'roc_auc',
                 n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric name ('roc_auc', 'pr_auc', 'ks')
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    metric_func = {
        'roc_auc': roc_auc_score,
        'pr_auc': average_precision_score,
        'ks': compute_ks_statistic
    }.get(metric, roc_auc_score)

    point_estimate = metric_func(y_true, y_proba)

    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            score = metric_func(y_true[indices], y_proba[indices])
            scores.append(score)
        except ValueError:
            continue

    scores = np.array(scores)
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)

    return point_estimate, lower, upper


def compute_lift(y_true: np.ndarray, y_proba: np.ndarray, n_deciles: int = 10) -> np.ndarray:
    """
    Compute lift curve values across deciles.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_deciles: Number of deciles

    Returns:
        Array of lift values for each decile
    """
    sorted_indices = np.argsort(-y_proba)  # Sort descending
    y_sorted = y_true[sorted_indices]

    decile_size = len(y_true) // n_deciles
    lifts = []

    overall_rate = y_true.mean()

    for i in range(n_deciles):
        start = i * decile_size
        end = start + decile_size if i < n_deciles - 1 else len(y_true)
        decile_rate = y_sorted[start:end].mean()
        lift = decile_rate / overall_rate if overall_rate > 0 else 0.0
        lifts.append(lift)

    return np.array(lifts)
