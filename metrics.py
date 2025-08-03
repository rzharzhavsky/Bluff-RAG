"""
CALM-RAG Metrics Module
Implements all calibration and confidence metrics for RAG evaluation.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import re
from typing import List, Tuple, Dict, Any


# Hedge terms for H3 hypothesis
HEDGE_TERMS = [
    "likely", "probably", "possibly", "perhaps", "maybe", "might", "could",
    "seems", "appears", "suggests", "indicates", "presumably", "allegedly",
    "reportedly", "supposedly", "apparently", "potentially", "uncertain",
    "unclear", "ambiguous", "debatable", "questionable", "tentative"
]


def retrieval_recall(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """
    Calculate retrieval recall: fraction of relevant documents retrieved.
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
    
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_docs:
        return 1.0
    
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    return len(retrieved_set.intersection(relevant_set)) / len(relevant_set)


def recall_confidence_correlation(recalls: List[float], confidences: List[float]) -> float:
    """
    Calculate Pearson correlation between retrieval recall and confidence.
    
    Args:
        recalls: List of recall scores
        confidences: List of confidence scores
    
    Returns:
        Pearson correlation coefficient
    """
    if len(recalls) != len(confidences) or len(recalls) < 2:
        return 0.0
    
    correlation, _ = pearsonr(recalls, confidences)
    return correlation if not np.isnan(correlation) else 0.0


def overconfidence_index(confidences: List[float], accuracies: List[float], tau: float = 0.8) -> float:
    """
    Calculate Overconfidence Index (OCI) - fraction of high-confidence wrong answers.
    
    Args:
        confidences: List of confidence scores (0-1)
        accuracies: List of binary accuracy scores (0 or 1)
        tau: Confidence threshold for "high confidence"
    
    Returns:
        OCI score between 0 and 1
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidences and accuracies must have same length")
    
    high_conf_mask = np.array(confidences) >= tau
    high_conf_count = np.sum(high_conf_mask)
    
    if high_conf_count == 0:
        return 0.0
    
    high_conf_wrong = np.sum(high_conf_mask & (np.array(accuracies) == 0))
    return high_conf_wrong / high_conf_count


def expected_calibration_error(confidences: List[float], accuracies: List[float], n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: List of confidence scores (0-1)
        accuracies: List of binary accuracy scores (0 or 1)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidences and accuracies must have same length")
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def brier_score(confidences: List[float], accuracies: List[float]) -> float:
    """
    Calculate Brier Score for calibration assessment.
    
    Args:
        confidences: List of confidence scores (0-1)
        accuracies: List of binary accuracy scores (0 or 1)
    
    Returns:
        Brier score (lower is better)
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidences and accuracies must have same length")
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    return np.mean((confidences - accuracies) ** 2)


def contains_hedge(text: str) -> bool:
    """
    Check if text contains hedge terms.
    
    Args:
        text: Input text to analyze
    
    Returns:
        True if hedge terms are found
    """
    text_lower = text.lower()
    return any(hedge in text_lower for hedge in HEDGE_TERMS)


def hedge_precision_recall(predictions: List[str], true_uncertainties: List[bool]) -> Tuple[float, float]:
    """
    Calculate precision and recall for hedge detection.
    
    Args:
        predictions: List of prediction texts
        true_uncertainties: List of boolean uncertainty labels
    
    Returns:
        Tuple of (precision, recall)
    """
    if len(predictions) != len(true_uncertainties):
        raise ValueError("Predictions and uncertainties must have same length")
    
    hedge_predictions = [contains_hedge(pred) for pred in predictions]
    
    # True positives: predicted hedge and truly uncertain
    tp = sum(1 for pred, true in zip(hedge_predictions, true_uncertainties) if pred and true)
    
    # False positives: predicted hedge but not truly uncertain
    fp = sum(1 for pred, true in zip(hedge_predictions, true_uncertainties) if pred and not true)
    
    # False negatives: didn't predict hedge but truly uncertain
    fn = sum(1 for pred, true in zip(hedge_predictions, true_uncertainties) if not pred and true)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def confidence_accuracy_correlation(confidences: List[float], accuracies: List[float]) -> float:
    """
    Calculate correlation between confidence and accuracy (H4).
    
    Args:
        confidences: List of confidence scores
        accuracies: List of accuracy scores
    
    Returns:
        Pearson correlation coefficient
    """
    if len(confidences) != len(accuracies) or len(confidences) < 2:
        return 0.0
    
    correlation, _ = pearsonr(confidences, accuracies)
    return correlation if not np.isnan(correlation) else 0.0


def isotonic_calibration(confidences: List[float], accuracies: List[float]) -> Tuple[List[float], float]:
    """
    Apply isotonic regression for calibration and return calibrated confidences + ECE.
    
    Args:
        confidences: List of confidence scores
        accuracies: List of binary accuracy scores
    
    Returns:
        Tuple of (calibrated_confidences, ece_after_calibration)
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidences and accuracies must have same length")
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    calibrated_confidences = iso_reg.fit_transform(confidences, accuracies)
    
    ece_after = expected_calibration_error(calibrated_confidences.tolist(), accuracies)
    
    return calibrated_confidences.tolist(), ece_after


def lexical_overconfidence_index(texts: List[str], accuracies: List[float]) -> float:
    """
    Calculate lexical overconfidence: confident language in wrong answers.
    
    Args:
        texts: List of prediction texts
        accuracies: List of accuracy scores (0 or 1)
    
    Returns:
        Lexical overconfidence index
    """
    confident_terms = ["definitely", "certainly", "clearly", "obviously", "undoubtedly", "absolutely"]
    
    confident_wrong = 0
    total_confident = 0
    
    for text, accuracy in zip(texts, accuracies):
        text_lower = text.lower()
        is_confident = any(term in text_lower for term in confident_terms)
        
        if is_confident:
            total_confident += 1
            if accuracy == 0:  # Wrong answer
                confident_wrong += 1
    
    return confident_wrong / total_confident if total_confident > 0 else 0.0


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute all CALM-RAG metrics from a list of results.
    
    Args:
        results: List of result dictionaries with keys:
                - confidence, accuracy, prediction_text, retrieval_recall, etc.
    
    Returns:
        Dictionary of all computed metrics
    """
    confidences = [r.get('confidence', 0.0) for r in results]
    accuracies = [r.get('accuracy', 0.0) for r in results]
    predictions = [r.get('prediction_text', '') for r in results]
    recalls = [r.get('retrieval_recall', 0.0) for r in results]
    uncertainties = [r.get('is_uncertain', False) for r in results]
    
    metrics = {}
    
    # H1 metrics
    metrics['overconfidence_index'] = overconfidence_index(confidences, accuracies)
    metrics['recall_confidence_correlation'] = recall_confidence_correlation(recalls, confidences)
    
    # H2 metrics
    metrics['expected_calibration_error'] = expected_calibration_error(confidences, accuracies)
    metrics['brier_score'] = brier_score(confidences, accuracies)
    
    # H3 metrics
    hedge_prec, hedge_rec = hedge_precision_recall(predictions, uncertainties)
    metrics['hedge_precision'] = hedge_prec
    metrics['hedge_recall'] = hedge_rec
    metrics['lexical_overconfidence_index'] = lexical_overconfidence_index(predictions, accuracies)
    
    # H4 metrics
    metrics['confidence_accuracy_correlation'] = confidence_accuracy_correlation(confidences, accuracies)
    
    # Isotonic calibration
    _, ece_after_isotonic = isotonic_calibration(confidences, accuracies)
    metrics['ece_after_isotonic'] = ece_after_isotonic
    
    return metrics
