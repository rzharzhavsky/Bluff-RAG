"""
CALM-RAG Metrics Module
Implements all calibration and confidence metrics for RAG evaluation.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import re
from typing import List, Tuple, Dict, Any, Union, Optional
from urllib.parse import urlparse


# Hedge terms for H3 hypothesis
HEDGE_TERMS = [
    "likely", "probably", "possibly", "perhaps", "maybe", "might", "could",
    "seems", "appears", "suggests", "indicates", "presumably", "allegedly",
    "reportedly", "supposedly", "apparently", "potentially", "uncertain",
    "unclear", "ambiguous", "debatable", "questionable", "tentative"
]


def normalize_document_id(doc: Union[str, Dict[str, Any]]) -> str:
    """
    Normalize document to a consistent ID for comparison.
    
    Args:
        doc: Document as string (URL/ID) or dict with 'url', 'id', or 'domain' keys
    
    Returns:
        Normalized document identifier
    """
    if isinstance(doc, str):
        # If it's a URL, normalize it
        if doc.startswith('http'):
            parsed = urlparse(doc)
            return f"{parsed.netloc}{parsed.path.rstrip('/')}"
        return doc.lower().strip()
    
    elif isinstance(doc, dict):
        # Try different keys in order of preference
        for key in ['url', 'id', 'domain', 'source']:
            if key in doc and doc[key]:
                if key == 'url' and doc[key].startswith('http'):
                    parsed = urlparse(doc[key])
                    return f"{parsed.netloc}{parsed.path.rstrip('/')}"
                return str(doc[key]).lower().strip()
        
        # Fallback to string representation
        return str(doc).lower().strip()
    
    return str(doc).lower().strip()


def retrieval_recall(retrieved_docs: List[Union[str, Dict]], 
                    relevant_docs: List[Union[str, Dict]]) -> float:
    """
    Calculate retrieval recall: fraction of relevant documents retrieved.
    Handles both string IDs and document dictionaries.
    
    Args:
        retrieved_docs: List of retrieved documents (strings or dicts)
        relevant_docs: List of relevant documents (strings or dicts)
    
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_docs:
        return 1.0
    
    # Normalize all document IDs
    retrieved_ids = {normalize_document_id(doc) for doc in retrieved_docs}
    relevant_ids = {normalize_document_id(doc) for doc in relevant_docs}
    
    intersection = retrieved_ids.intersection(relevant_ids)
    recall = len(intersection) / len(relevant_ids)
    
    return recall


def retrieval_precision(retrieved_docs: List[Union[str, Dict]], 
                       relevant_docs: List[Union[str, Dict]]) -> float:
    """
    Calculate retrieval precision: fraction of retrieved documents that are relevant.
    
    Args:
        retrieved_docs: List of retrieved documents
        relevant_docs: List of relevant documents
    
    Returns:
        Precision score between 0 and 1
    """
    if not retrieved_docs:
        return 0.0
    
    retrieved_ids = {normalize_document_id(doc) for doc in retrieved_docs}
    relevant_ids = {normalize_document_id(doc) for doc in relevant_docs}
    
    intersection = retrieved_ids.intersection(relevant_ids)
    precision = len(intersection) / len(retrieved_ids)
    
    return precision


def retrieval_f1(retrieved_docs: List[Union[str, Dict]], 
                 relevant_docs: List[Union[str, Dict]]) -> float:
    """
    Calculate retrieval F1 score.
    
    Args:
        retrieved_docs: List of retrieved documents
        relevant_docs: List of relevant documents
    
    Returns:
        F1 score between 0 and 1
    """
    precision = retrieval_precision(retrieved_docs, relevant_docs)
    recall = retrieval_recall(retrieved_docs, relevant_docs)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def source_quality_score(retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Calculate source quality based on domain reliability categories.
    
    Args:
        retrieved_docs: List of document dicts with 'category' field
                       ('reliable', 'unreliable', 'unknown')
    
    Returns:
        Quality score between 0 and 1
    """
    if not retrieved_docs:
        return 0.0
    
    quality_weights = {
        'reliable': 1.0,
        'unknown': 0.5,
        'unreliable': 0.0
    }
    
    total_weight = 0.0
    for doc in retrieved_docs:
        category = doc.get('category', 'unknown').lower()
        total_weight += quality_weights.get(category, 0.5)
    
    return total_weight / len(retrieved_docs)


def retrieval_diversity(retrieved_docs: List[Union[str, Dict]]) -> float:
    """
    Calculate retrieval diversity based on unique domains.
    
    Args:
        retrieved_docs: List of retrieved documents
    
    Returns:
        Diversity score between 0 and 1
    """
    if not retrieved_docs:
        return 0.0
    
    domains = set()
    for doc in retrieved_docs:
        if isinstance(doc, dict) and 'domain' in doc:
            domains.add(doc['domain'].lower())
        elif isinstance(doc, dict) and 'url' in doc:
            parsed = urlparse(doc['url'])
            domains.add(parsed.netloc.lower())
        elif isinstance(doc, str) and doc.startswith('http'):
            parsed = urlparse(doc)
            domains.add(parsed.netloc.lower())
    
    # Diversity is ratio of unique domains to total documents
    return len(domains) / len(retrieved_docs)


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
    
    # Filter out any NaN or infinite values
    valid_pairs = [(r, c) for r, c in zip(recalls, confidences) 
                   if not (np.isnan(r) or np.isnan(c) or np.isinf(r) or np.isinf(c))]
    
    if len(valid_pairs) < 2:
        return 0.0
    
    valid_recalls, valid_confidences = zip(*valid_pairs)
    correlation, _ = pearsonr(valid_recalls, valid_confidences)
    return correlation if not np.isnan(correlation) else 0.0


def retrieval_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute comprehensive retrieval quality metrics.
    
    Args:
        results: List of result dicts with keys like:
                - retrieved_docs: List of retrieved documents
                - relevant_docs: List of relevant documents  
                - confidence: Confidence score
    
    Returns:
        Dictionary of retrieval quality metrics
    """
    metrics = {}
    
    # Individual retrieval metrics
    recalls = []
    precisions = []
    f1_scores = []
    quality_scores = []
    diversity_scores = []
    confidences = []
    
    for result in results:
        retrieved = result.get('retrieved_docs', [])
        relevant = result.get('relevant_docs', [])
        confidence = result.get('confidence', 0.0)
        
        # Basic retrieval metrics
        recall = retrieval_recall(retrieved, relevant)
        precision = retrieval_precision(retrieved, relevant)
        f1 = retrieval_f1(retrieved, relevant)
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        confidences.append(confidence)
        
        # Source quality (if available)
        if retrieved and isinstance(retrieved[0], dict):
            quality = source_quality_score(retrieved)
            diversity = retrieval_diversity(retrieved)
            quality_scores.append(quality)
            diversity_scores.append(diversity)
    
    # Aggregate metrics
    metrics['avg_retrieval_recall'] = np.mean(recalls) if recalls else 0.0
    metrics['avg_retrieval_precision'] = np.mean(precisions) if precisions else 0.0
    metrics['avg_retrieval_f1'] = np.mean(f1_scores) if f1_scores else 0.0
    
    # Confidence-recall correlation
    metrics['recall_confidence_correlation'] = recall_confidence_correlation(recalls, confidences)
    
    # Source quality metrics (if available)
    if quality_scores:
        metrics['avg_source_quality'] = np.mean(quality_scores)
        metrics['source_quality_std'] = np.std(quality_scores)
    
    if diversity_scores:
        metrics['avg_source_diversity'] = np.mean(diversity_scores)
        metrics['source_diversity_std'] = np.std(diversity_scores)
    
    return metrics


def overconfidence_index(confidences: List[float], accuracies: List[float], tau: float = 0.8) -> float:
    """
    Calculate Overconfidence Index (OCI) - fraction of high-confidence wrong answers.
    From CALM-RAG section 2.5.
    
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
    From CALM-RAG section 2.2.
    
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
    From CALM-RAG section 2.1.
    
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
    From CALM-RAG H3 hypothesis.
    
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
    From CALM-RAG H3 hypothesis.
    
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
    From CALM-RAG section 2.3.
    
    Args:
        confidences: List of confidence scores
        accuracies: List of accuracy scores
    
    Returns:
        Pearson correlation coefficient
    """
    if len(confidences) != len(accuracies) or len(confidences) < 2:
        return 0.0
    
    # Filter out any NaN or infinite values
    valid_pairs = [(c, a) for c, a in zip(confidences, accuracies) 
                   if not (np.isnan(c) or np.isnan(a) or np.isinf(c) or np.isinf(a))]
    
    if len(valid_pairs) < 2:
        return 0.0
    
    valid_confidences, valid_accuracies = zip(*valid_pairs)
    correlation, _ = pearsonr(valid_confidences, valid_accuracies)
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
    From CALM-RAG H3 hypothesis.
    
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


# =====================================================================
# CALM-RAG ALIGNED BENCHMARK FUNCTIONS
# =====================================================================

def get_retrieval_recall_benchmark(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    CALM-RAG aligned retrieval recall benchmark for H1: Overconfidence Under Sparse or Noisy Evidence.
    
    Computes retrieval recall as defined in CALM-RAG proposal section 2.4:
    R_i = (#of ground-truth facts retrieved) / (#of ground-truth facts)
    
    This measures how well the RAG system retrieves supporting evidence,
    which is critical for detecting overconfidence when evidence is sparse.
    
    Args:
        results: List of result dictionaries with CALM-RAG schema:
                - retrieved_docs: List of retrieved documents/passages
                - relevant_docs: List of ground-truth relevant documents
                - confidence: Model confidence score (for correlation analysis)
    
    Returns:
        Dictionary with retrieval recall metrics:
        - avg_retrieval_recall: Average recall across all queries
        - retrieval_recall_confidence_correlation: Correlation between recall and confidence (H1 metric)
    """
    if not results:
        return {'avg_retrieval_recall': 0.0, 'retrieval_recall_confidence_correlation': 0.0}
    
    recall_scores = []
    confidences = []
    
    for result in results:
        retrieved = result.get('retrieved_docs', [])
        relevant = result.get('relevant_docs', [])
        confidence = result.get('confidence', 0.0)
        
        # Calculate R_i as per CALM-RAG formula (section 2.4)
        if not relevant:
            # No ground-truth facts to retrieve
            recall_i = 1.0
        else:
            # Normalize document IDs for comparison
            retrieved_ids = {normalize_document_id(doc) for doc in retrieved}
            relevant_ids = {normalize_document_id(doc) for doc in relevant}
            
            # R_i = (#ground-truth facts retrieved) / (#ground-truth facts)
            intersection = retrieved_ids.intersection(relevant_ids)
            recall_i = len(intersection) / len(relevant_ids)
        
        recall_scores.append(recall_i)
        confidences.append(confidence)
    
    # Primary metric: average retrieval recall
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    # H1 metric: Retrieval Recall vs. Confidence Correlation
    # Negative correlation indicates overconfidence when evidence is sparse
    recall_confidence_corr = recall_confidence_correlation(recall_scores, confidences)
    
    return {
        'avg_retrieval_recall': avg_recall,
        'retrieval_recall_confidence_correlation': recall_confidence_corr
    }


def calm_rag_h1_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Complete H1 hypothesis metrics: Overconfidence Under Sparse or Noisy Evidence.
    
    Implements all metrics from CALM-RAG proposal section H1:
    - Retrieval Recall vs. Confidence correlation (negative expected)
    - Overconfidence Index for confident hallucinations
    - No-Answer vs. Wrong-Answer Rate analysis
    
    Args:
        results: List with CALM-RAG schema (section 4):
                - retrieved_docs: Retrieved passages/documents
                - relevant_docs: Ground-truth relevant documents
                - confidence: Model confidence score [0,1]
                - accuracy: Binary correctness [0,1] 
                - prediction_text: Model's answer text
    
    Returns:
        H1 metrics dictionary aligned with CALM-RAG proposal
    """
    if not results:
        return {
            'retrieval_recall_confidence_correlation': 0.0,
            'avg_retrieval_recall': 0.0,
            'overconfidence_index': 0.0,
            'wrong_answer_rate': 0.0,
            'refusal_rate': 0.0
        }
    
    recall_scores = []
    confidences = []
    accuracies = []
    wrong_answers = 0
    refusals = 0
    
    for result in results:
        # Retrieval recall calculation (R_i from section 2.4)
        retrieved = result.get('retrieved_docs', [])
        relevant = result.get('relevant_docs', [])
        confidence = result.get('confidence', 0.0)
        accuracy = result.get('accuracy', 0.0)
        answer_text = result.get('prediction_text', '').strip()
        
        if not relevant:
            recall_i = 1.0
        else:
            retrieved_ids = {normalize_document_id(doc) for doc in retrieved}
            relevant_ids = {normalize_document_id(doc) for doc in relevant}
            intersection = retrieved_ids.intersection(relevant_ids)
            recall_i = len(intersection) / len(relevant_ids)
        
        recall_scores.append(recall_i)
        confidences.append(confidence)
        accuracies.append(accuracy)
        
        # Count refusals and wrong answers (section 2.6)
        if not answer_text or answer_text.lower() in ["i don't know", "unclear", "unknown"]:
            refusals += 1
        elif accuracy == 0:  # Wrong non-empty answer
            wrong_answers += 1
    
    total_questions = len(results)
    
    return {
        # Core H1 metric: negative correlation indicates overconfidence when evidence is sparse
        'retrieval_recall_confidence_correlation': recall_confidence_correlation(recall_scores, confidences),
        'avg_retrieval_recall': sum(recall_scores) / len(recall_scores),
        
        # Overconfidence Index (section 2.5) - confident wrong answers
        'overconfidence_index': overconfidence_index(confidences, accuracies, tau=0.8),
        
        # No-Answer vs Wrong-Answer rates (section 2.6)
        'wrong_answer_rate': wrong_answers / total_questions,
        'refusal_rate': refusals / total_questions
    }


def calm_rag_h2_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Complete H2 hypothesis metrics: Calibration Difference with and without Retrieval.
    
    Implements metrics from CALM-RAG proposal section H2:
    - ECE comparison between retrieval and non-retrieval scenarios
    - Brier score comparison
    - Confidence-accuracy correlation difference
    
    Args:
        results: List with CALM-RAG schema:
                - retrieved_docs: Retrieved passages/documents
                - relevant_docs: Ground-truth relevant documents
                - confidence: Model confidence score [0,1]
                - accuracy: Binary correctness [0,1]
                - no_retrieval_confidence: Confidence without retrieval (if available)
                - no_retrieval_accuracy: Accuracy without retrieval (if available)
    
    Returns:
        H2 metrics dictionary aligned with CALM-RAG proposal
    """
    if not results:
        return {
            'ece_with_retrieval': 0.0,
            'ece_without_retrieval': 0.0,
            'ece_difference': 0.0,
            'brier_with_retrieval': 0.0,
            'brier_without_retrieval': 0.0,
            'brier_difference': 0.0,
            'confidence_accuracy_corr_with_retrieval': 0.0,
            'confidence_accuracy_corr_without_retrieval': 0.0,
            'correlation_difference': 0.0
        }
    
    # With retrieval metrics
    confidences_with = [r.get('confidence', 0.0) for r in results]
    accuracies_with = [r.get('accuracy', 0.0) for r in results]
    
    ece_with = expected_calibration_error(confidences_with, accuracies_with)
    brier_with = brier_score(confidences_with, accuracies_with)
    corr_with = confidence_accuracy_correlation(confidences_with, accuracies_with)
    
    # Without retrieval metrics (if available)
    confidences_without = [r.get('no_retrieval_confidence', 0.0) for r in results]
    accuracies_without = [r.get('no_retrieval_accuracy', 0.0) for r in results]
    
    # Check if we have no-retrieval data
    has_no_retrieval = any(c != 0.0 for c in confidences_without)
    
    if has_no_retrieval:
        ece_without = expected_calibration_error(confidences_without, accuracies_without)
        brier_without = brier_score(confidences_without, accuracies_without)
        corr_without = confidence_accuracy_correlation(confidences_without, accuracies_without)
        
        ece_diff = ece_with - ece_without
        brier_diff = brier_with - brier_without
        corr_diff = corr_with - corr_without
    else:
        # Use baseline values if no comparison data
        ece_without = ece_with  # Same as with retrieval
        brier_without = brier_with
        corr_without = corr_with
        ece_diff = 0.0
        brier_diff = 0.0
        corr_diff = 0.0
    
    return {
        'ece_with_retrieval': ece_with,
        'ece_without_retrieval': ece_without,
        'ece_difference': ece_diff,
        'brier_with_retrieval': brier_with,
        'brier_without_retrieval': brier_without,
        'brier_difference': brier_diff,
        'confidence_accuracy_corr_with_retrieval': corr_with,
        'confidence_accuracy_corr_without_retrieval': corr_without,
        'correlation_difference': corr_diff
    }


def calm_rag_h3_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Complete H3 hypothesis metrics: Hedging Language as a Signal of Uncertainty.
    
    Implements enhanced metrics from CALM-RAG proposal section H3:
    - Hedge term precision/recall
    - Lexical overconfidence detection
    - Uncertainty expression correlation
    - Hedge density analysis
    
    Args:
        results: List with CALM-RAG schema:
                - prediction_text: Model's answer text
                - is_uncertain: Boolean uncertainty label
                - confidence: Model confidence score
                - accuracy: Binary correctness
    
    Returns:
        H3 metrics dictionary aligned with CALM-RAG proposal
    """
    if not results:
        return {
            'hedge_precision': 0.0,
            'hedge_recall': 0.0,
            'hedge_f1': 0.0,
            'lexical_overconfidence_index': 0.0,
            'uncertainty_confidence_correlation': 0.0,
            'hedge_density': 0.0,
            'confident_wrong_rate': 0.0,
            'hedge_sophistication': 0.0,
            'enhanced_hedge_detection_rate': 0.0
        }
    
    predictions = [r.get('prediction_text', '') for r in results]
    uncertainties = [r.get('is_uncertain', False) for r in results]
    confidences = [r.get('confidence', 0.0) for r in results]
    accuracies = [r.get('accuracy', 0.0) for r in results]
    
    # Basic hedge metrics
    hedge_prec, hedge_rec = hedge_precision_recall(predictions, uncertainties)
    hedge_f1 = 2 * (hedge_prec * hedge_rec) / (hedge_prec + hedge_rec) if (hedge_prec + hedge_rec) > 0 else 0.0
    
    # Enhanced hedge detection
    enhanced_hedge_count = 0
    total_predictions = 0
    
    for prediction in predictions:
        if prediction:
            total_predictions += 1
            has_enhanced_hedge, _ = contains_enhanced_hedge(prediction)
            if has_enhanced_hedge:
                enhanced_hedge_count += 1
    
    enhanced_hedge_detection_rate = enhanced_hedge_count / total_predictions if total_predictions > 0 else 0.0
    
    # Lexical overconfidence
    lexical_oci = lexical_overconfidence_index(predictions, accuracies)
    
    # Uncertainty-confidence correlation
    uncertainty_confidence_corr = confidence_accuracy_correlation(confidences, uncertainties)
    
    # Hedge density analysis
    hedge_density = calculate_hedge_density(predictions)
    
    # Confident wrong answers rate
    confident_wrong_rate = calculate_confident_wrong_rate(confidences, accuracies)
    
    # Hedge sophistication
    hedge_soph_metrics = calculate_hedge_sophistication(predictions)
    hedge_sophistication = hedge_soph_metrics.get('sophisticated_hedge_rate', 0.0)
    
    return {
        'hedge_precision': hedge_prec,
        'hedge_recall': hedge_rec,
        'hedge_f1': hedge_f1,
        'lexical_overconfidence_index': lexical_oci,
        'uncertainty_confidence_correlation': uncertainty_confidence_corr,
        'hedge_density': hedge_density,
        'confident_wrong_rate': confident_wrong_rate,
        'hedge_sophistication': hedge_sophistication,
        'enhanced_hedge_detection_rate': enhanced_hedge_detection_rate
    }


def calm_rag_h4_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Complete H4 hypothesis metrics: Self-Assessment and Numeric Confidence Calibration.
    
    Implements comprehensive calibration metrics from CALM-RAG proposal section H4:
    - Multiple calibration error metrics
    - Isotonic regression calibration
    - Confidence distribution analysis
    - Calibration improvement metrics
    
    Args:
        results: List with CALM-RAG schema:
                - confidence: Model confidence score [0,1]
                - accuracy: Binary correctness [0,1]
                - human_confidence: Human confidence score (if available)
    
    Returns:
        H4 metrics dictionary aligned with CALM-RAG proposal
    """
    if not results:
        return {
            'expected_calibration_error': 0.0,
            'brier_score': 0.0,
            'confidence_accuracy_correlation': 0.0,
            'calibration_ece_after_isotonic': 0.0,
            'calibration_improvement': 0.0,
            'confidence_distribution_entropy': 0.0,
            'human_model_confidence_correlation': 0.0
        }
    
    confidences = [r.get('confidence', 0.0) for r in results]
    accuracies = [r.get('accuracy', 0.0) for r in results]
    human_confidences = [r.get('human_confidence', None) for r in results]
    
    # Basic calibration metrics
    ece = expected_calibration_error(confidences, accuracies)
    brier = brier_score(confidences, accuracies)
    corr = confidence_accuracy_correlation(confidences, accuracies)
    
    # Isotonic calibration
    calibrated_confidences, ece_after_isotonic = isotonic_calibration(confidences, accuracies)
    calibration_improvement = ece - ece_after_isotonic
    
    # Confidence distribution analysis
    confidence_entropy = calculate_confidence_entropy(confidences)
    
    # Human-model confidence correlation (if available)
    human_model_corr = 0.0
    if any(hc is not None for hc in human_confidences):
        valid_pairs = [(c, hc) for c, hc in zip(confidences, human_confidences) if hc is not None]
        if len(valid_pairs) >= 2:
            valid_confidences, valid_human_confidences = zip(*valid_pairs)
            human_model_corr = confidence_accuracy_correlation(valid_confidences, valid_human_confidences)
    
    return {
        'expected_calibration_error': ece,
        'brier_score': brier,
        'confidence_accuracy_correlation': corr,
        'calibration_ece_after_isotonic': ece_after_isotonic,
        'calibration_improvement': calibration_improvement,
        'confidence_distribution_entropy': confidence_entropy,
        'human_model_confidence_correlation': human_model_corr
    }


def calm_rag_h5_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Complete H5 hypothesis metrics: Source Quality Impact on Calibration.
    
    Implements source quality correlation metrics from CALM-RAG proposal section H5:
    - Source quality vs. confidence correlation
    - Source diversity vs. calibration correlation
    - Quality-weighted calibration metrics
    
    Args:
        results: List with CALM-RAG schema:
                - retrieved_docs: List of retrieved documents with quality info
                - confidence: Model confidence score
                - accuracy: Binary correctness
                - source_quality: Source quality score (if available)
    
    Returns:
        H5 metrics dictionary aligned with CALM-RAG proposal
    """
    if not results:
        return {
            'source_quality_confidence_correlation': 0.0,
            'source_diversity_calibration_correlation': 0.0,
            'quality_weighted_ece': 0.0,
            'high_quality_source_ece': 0.0,
            'low_quality_source_ece': 0.0,
            'quality_calibration_gap': 0.0
        }
    
    source_qualities = []
    source_diversities = []
    confidences = []
    accuracies = []
    
    for result in results:
        retrieved = result.get('retrieved_docs', [])
        confidence = result.get('confidence', 0.0)
        accuracy = result.get('accuracy', 0.0)
        
        if retrieved:
            # Calculate source quality
            quality = source_quality_score(retrieved)
            diversity = retrieval_diversity(retrieved)
            
            source_qualities.append(quality)
            source_diversities.append(diversity)
            confidences.append(confidence)
            accuracies.append(accuracy)
    
    if not source_qualities:
        return {
            'source_quality_confidence_correlation': 0.0,
            'source_diversity_calibration_correlation': 0.0,
            'quality_weighted_ece': 0.0,
            'high_quality_source_ece': 0.0,
            'low_quality_source_ece': 0.0,
            'quality_calibration_gap': 0.0
        }
    
    # Source quality vs. confidence correlation
    quality_confidence_corr = confidence_accuracy_correlation(source_qualities, confidences)
    
    # Source diversity vs. calibration correlation
    diversity_calibration_corr = confidence_accuracy_correlation(source_diversities, accuracies)
    
    # Quality-weighted calibration metrics
    quality_weighted_ece = calculate_quality_weighted_ece(source_qualities, confidences, accuracies)
    
    # High vs. low quality source calibration
    high_quality_ece, low_quality_ece = calculate_quality_separated_ece(source_qualities, confidences, accuracies)
    quality_calibration_gap = high_quality_ece - low_quality_ece
    
    return {
        'source_quality_confidence_correlation': quality_confidence_corr,
        'source_diversity_calibration_correlation': diversity_calibration_corr,
        'quality_weighted_ece': quality_weighted_ece,
        'high_quality_source_ece': high_quality_ece,
        'low_quality_source_ece': low_quality_ece,
        'quality_calibration_gap': quality_calibration_gap
    }


# Enhanced hedge detection functions
def calculate_hedge_density(texts: List[str]) -> float:
    """
    Calculate average hedge term density across texts.
    
    Args:
        texts: List of text strings to analyze
    
    Returns:
        Average hedge density (hedge terms per word)
    """
    if not texts:
        return 0.0
    
    total_hedge_count = 0
    total_word_count = 0
    
    for text in texts:
        if not text:
            continue
        
        words = text.lower().split()
        hedge_count = sum(1 for word in words if any(hedge in word for hedge in HEDGE_TERMS))
        
        total_hedge_count += hedge_count
        total_word_count += len(words)
    
    return total_hedge_count / total_word_count if total_word_count > 0 else 0.0


def calculate_confident_wrong_rate(confidences: List[float], accuracies: List[float], threshold: float = 0.8) -> float:
    """
    Calculate rate of confident but wrong answers.
    
    Args:
        confidences: List of confidence scores
        accuracies: List of accuracy scores
        threshold: Confidence threshold for "confident" answers
    
    Returns:
        Rate of confident wrong answers
    """
    if len(confidences) != len(accuracies):
        return 0.0
    
    confident_count = 0
    confident_wrong_count = 0
    
    for conf, acc in zip(confidences, accuracies):
        if conf >= threshold:
            confident_count += 1
            if acc == 0:  # Wrong answer
                confident_wrong_count += 1
    
    return confident_wrong_count / confident_count if confident_count > 0 else 0.0


def calculate_confidence_entropy(confidences: List[float], n_bins: int = 10) -> float:
    """
    Calculate entropy of confidence distribution.
    
    Args:
        confidences: List of confidence scores
        n_bins: Number of bins for histogram
    
    Returns:
        Entropy of confidence distribution
    """
    if not confidences:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(confidences, bins=n_bins, range=(0, 1))
    
    # Calculate entropy
    hist = hist[hist > 0]  # Remove zero bins
    if len(hist) == 0:
        return 0.0
    
    probabilities = hist / hist.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def calculate_quality_weighted_ece(source_qualities: List[float], confidences: List[float], accuracies: List[float]) -> float:
    """
    Calculate quality-weighted Expected Calibration Error.
    
    Args:
        source_qualities: List of source quality scores
        confidences: List of confidence scores
        accuracies: List of accuracy scores
    
    Returns:
        Quality-weighted ECE
    """
    if len(source_qualities) != len(confidences) or len(confidences) != len(accuracies):
        return 0.0
    
    # Normalize quality scores to sum to 1
    total_quality = sum(source_qualities)
    if total_quality == 0:
        return expected_calibration_error(confidences, accuracies)
    
    normalized_qualities = [q / total_quality for q in source_qualities]
    
    # Calculate weighted ECE
    weighted_ece = 0.0
    for quality, conf, acc in zip(normalized_qualities, confidences, accuracies):
        weighted_ece += quality * abs(conf - acc)
    
    return weighted_ece


def calculate_quality_separated_ece(source_qualities: List[float], confidences: List[float], accuracies: List[float]) -> Tuple[float, float]:
    """
    Calculate ECE separately for high and low quality sources.
    
    Args:
        source_qualities: List of source quality scores
        confidences: List of confidence scores
        accuracies: List of accuracy scores
    
    Returns:
        Tuple of (high_quality_ece, low_quality_ece)
    """
    if len(source_qualities) != len(confidences) or len(confidences) != len(accuracies):
        return 0.0, 0.0
    
    # Split into high and low quality groups
    median_quality = np.median(source_qualities)
    
    high_quality_data = [(c, a) for q, c, a in zip(source_qualities, confidences, accuracies) if q >= median_quality]
    low_quality_data = [(c, a) for q, c, a in zip(source_qualities, confidences, accuracies) if q < median_quality]
    
    # Calculate ECE for each group
    high_quality_ece = 0.0
    low_quality_ece = 0.0
    
    if high_quality_data:
        high_conf, high_acc = zip(*high_quality_data)
        high_quality_ece = expected_calibration_error(list(high_conf), list(high_acc))
    
    if low_quality_data:
        low_conf, low_acc = zip(*low_quality_data)
        low_quality_ece = expected_calibration_error(list(low_conf), list(low_acc))
    
    return high_quality_ece, low_quality_ece


# Enhanced retrieval quality metrics
def retrieval_quality_metrics_enhanced(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Enhanced retrieval quality metrics with source quality analysis.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary of enhanced retrieval quality metrics
    """
    metrics = {}
    
    # Basic retrieval metrics
    recalls = []
    precisions = []
    f1_scores = []
    quality_scores = []
    diversity_scores = []
    confidences = []
    accuracies = []
    
    for result in results:
        retrieved = result.get('retrieved_docs', [])
        relevant = result.get('relevant_docs', [])
        confidence = result.get('confidence', 0.0)
        accuracy = result.get('accuracy', 0.0)
        
        # Basic retrieval metrics
        recall = retrieval_recall(retrieved, relevant)
        precision = retrieval_precision(retrieved, relevant)
        f1 = retrieval_f1(retrieved, relevant)
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        confidences.append(confidence)
        accuracies.append(accuracy)
        
        # Source quality (if available)
        if retrieved and isinstance(retrieved[0], dict):
            quality = source_quality_score(retrieved)
            diversity = retrieval_diversity(retrieved)
            quality_scores.append(quality)
            diversity_scores.append(diversity)
    
    # Aggregate metrics
    metrics['avg_retrieval_recall'] = np.mean(recalls) if recalls else 0.0
    metrics['avg_retrieval_precision'] = np.mean(precisions) if precisions else 0.0
    metrics['avg_retrieval_f1'] = np.mean(f1_scores) if f1_scores else 0.0
    
    # Confidence correlations
    metrics['recall_confidence_correlation'] = recall_confidence_correlation(recalls, confidences)
    metrics['accuracy_confidence_correlation'] = confidence_accuracy_correlation(confidences, accuracies)
    
    # Source quality metrics (if available)
    if quality_scores:
        metrics['avg_source_quality'] = np.mean(quality_scores)
        metrics['source_quality_std'] = np.std(quality_scores)
        metrics['source_quality_confidence_correlation'] = confidence_accuracy_correlation(quality_scores, confidences)
    
    if diversity_scores:
        metrics['avg_source_diversity'] = np.mean(diversity_scores)
        metrics['source_diversity_std'] = np.std(diversity_scores)
        metrics['source_diversity_accuracy_correlation'] = confidence_accuracy_correlation(diversity_scores, accuracies)
    
    return metrics


def compute_all_calm_rag_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute all CALM-RAG metrics from a list of results.
    Aligned with the complete CALM-RAG proposal hypotheses H1-H5.
    
    Args:
        results: List of result dictionaries with CALM-RAG schema keys:
                - confidence, accuracy, prediction_text, retrieved_docs, relevant_docs, 
                - is_uncertain, etc.
    
    Returns:
        Dictionary of all computed CALM-RAG metrics organized by hypothesis
    """
    metrics = {}
    
    # H1: Overconfidence Under Sparse or Noisy Evidence
    h1_metrics = calm_rag_h1_metrics(results)
    for key, value in h1_metrics.items():
        metrics[f'h1_{key}'] = value
    
    # H2: Calibration Difference with and without Retrieval
    h2_metrics = calm_rag_h2_metrics(results)
    for key, value in h2_metrics.items():
        metrics[f'h2_{key}'] = value
    
    # H3: Hedging Language as a Signal of Uncertainty
    h3_metrics = calm_rag_h3_metrics(results)
    for key, value in h3_metrics.items():
        metrics[f'h3_{key}'] = value
    
    # H4: Self-Assessment and Numeric Confidence Calibration
    h4_metrics = calm_rag_h4_metrics(results)
    for key, value in h4_metrics.items():
        metrics[f'h4_{key}'] = value
    
    # H5: Source Quality Impact on Calibration
    h5_metrics = calm_rag_h5_metrics(results)
    for key, value in h5_metrics.items():
        metrics[f'h5_{key}'] = value
    
    # Enhanced retrieval quality metrics
    retrieval_metrics = retrieval_quality_metrics_enhanced(results)
    for key, value in retrieval_metrics.items():
        metrics[f'retrieval_{key}'] = value
    
    return metrics


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Legacy function - kept for backwards compatibility.
    Use compute_all_calm_rag_metrics() for CALM-RAG aligned evaluation.
    """
    return compute_all_calm_rag_metrics(results)

"""
======================================================================================
Testing Metrics
======================================================================================
"""
# Example 1: Basic retrieval recall benchmark data
basic_results = [
    {
        # Question 1: Good retrieval, high confidence (well-calibrated)
        "retrieved_docs": [
            "https://nejm.org/drug-x-trial",
            "https://pubmed.gov/12345",
            "https://clinicaltrials.gov/ct2/show/NCT123"
        ],
        "relevant_docs": [
            "https://nejm.org/drug-x-trial",
            "https://pubmed.gov/12345"
        ],
        "confidence": 0.85,
        "accuracy": 1.0,  # Correct answer
        "prediction_text": "The Phase 3 trial showed 45% remission rate."
    },
    {
        # Question 2: Poor retrieval, high confidence (overconfident - RED FLAG)
        "retrieved_docs": [
            "https://healthblog.com/random-post",
            "https://wikipedia.org/some-page"
        ],
        "relevant_docs": [
            "https://nejm.org/authoritative-study",
            "https://jama.org/clinical-data"
        ],
        "confidence": 0.92,  # Very confident but wrong retrieval
        "accuracy": 0.0,     # Wrong answer
        "prediction_text": "Drug Y cures 95% of patients."
    },
    {
        # Question 3: Good retrieval, appropriate hedging
        "retrieved_docs": [
            "https://cochrane.org/systematic-review",
            "https://nejm.org/conflicting-study"
        ],
        "relevant_docs": [
            "https://cochrane.org/systematic-review",
            "https://nejm.org/conflicting-study",
            "https://jama.org/missing-study"  # This one wasn't retrieved
        ],
        "confidence": 0.60,  # Appropriately uncertain
        "accuracy": 1.0,
        "prediction_text": "The evidence suggests it might be effective, but results are mixed.",
        "is_uncertain": True
    }
]

# Example 2: CALM-RAG schema format (from proposal section 4)
calm_rag_results = [
    {
        "id": 17,
        "domain": "medicine",
        "question": "What was the remission rate in the Phase 3 trial of Drug X?",
        
        # Retrieved documents (what RAG system found)
        "retrieved_docs": [
            {
                "title": "NEJM 2021 Trial Results",
                "url": "https://doi.org/10.xxxx",
                "date": "2021-05-10",
                "text": "The Phase 3 trial of Drug X reported a 45% complete remission rate..."
            },
            {
                "title": "HealthNews on Drug X", 
                "url": "https://healthnews.example.com/...",
                "date": "2022-05-05",
                "text": "Some experts note that the true efficacy of Drug X is uncertain..."
            }
        ],
        
        # Ground truth relevant documents (gold standard)
        "relevant_docs": [
            {
                "title": "NEJM 2021 Trial Results",
                "url": "https://doi.org/10.xxxx",
                "date": "2021-05-10"
            }
            # Note: Only 1 out of 2 retrieved docs is actually relevant
        ],
        
        "confidence": 0.85,
        "accuracy": 1.0,
        "prediction_text": "The Phase 3 trial showed a 45% complete remission rate.",
        "gold_answer": "45% (complete remission rate)",
        "human_confidence": 0.6,
        "human_hedge_label": "Likely",
        "is_uncertain": False
    },
    
    {
        "id": 18,
        "domain": "climate",
        "question": "What is the predicted sea level rise by 2100?",
        
        "retrieved_docs": [
            "https://ipcc.ch/report-ar6",
            "https://climate-blog.net/speculation"  # Low quality source
        ],
        
        "relevant_docs": [
            "https://ipcc.ch/report-ar6",
            "https://nature.com/climate-study",  # This authoritative source was missed
            "https://science.org/sea-level-data"  # This one too
        ],
        
        "confidence": 0.95,  # Overconfident despite poor retrieval
        "accuracy": 0.0,     # Wrong answer
        "prediction_text": "Sea levels will definitely rise 2.5 meters by 2100.",
        "gold_answer": "0.43-0.84 meters (IPCC AR6 estimate)",
        "is_uncertain": False
    }
]

# Example 3: Simple string-based documents (easier format)
string_based_results = [
    {
        "retrieved_docs": ["doc1", "doc2", "doc3"],
        "relevant_docs": ["doc1", "doc4"],  # Only doc1 was correctly retrieved
        "confidence": 0.7,
        "accuracy": 1.0,
        "prediction_text": "The answer is probably X based on the evidence."
    },
    {
        "retrieved_docs": ["irrelevant_doc1", "irrelevant_doc2"],
        "relevant_docs": ["good_doc1", "good_doc2", "good_doc3"],  # 0/3 recall!
        "confidence": 0.9,  # Overconfident
        "accuracy": 0.0,
        "prediction_text": "I'm certain the answer is Z."
    }
]

# Example 4: Mixed document formats (handles both)
mixed_results = [
    {
        "retrieved_docs": [
            {"url": "https://example.com/doc1", "title": "Study 1"},
            "simple_doc_id_2"
        ],
        "relevant_docs": [
            "https://example.com/doc1",  # Will match the URL
            {"id": "doc3", "domain": "medicine"}
        ],
        "confidence": 0.8,
        "accuracy": 1.0,
        "prediction_text": "Based on current research, the treatment appears effective."
    }
]

print("=== BASIC RETRIEVAL RECALL BENCHMARK ===")
benchmark_results = get_retrieval_recall_benchmark(calm_rag_results)
print(f"Average Retrieval Recall: {benchmark_results['avg_retrieval_recall']:.3f}")

# Additional utility functions for comprehensive evaluation
def calculate_retrieval_confidence_gap(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate the gap between retrieval quality and confidence.
    This helps identify overconfidence when retrieval is poor.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with confidence gap metrics
    """
    if not results:
        return {
            'avg_confidence_gap': 0.0,
            'overconfidence_rate': 0.0,
            'underconfidence_rate': 0.0
        }
    
    confidence_gaps = []
    overconfident_count = 0
    underconfident_count = 0
    
    for result in results:
        retrieved = result.get('retrieved_docs', [])
        relevant = result.get('relevant_docs', [])
        confidence = result.get('confidence', 0.0)
        
        if retrieved and relevant:
            recall = retrieval_recall(retrieved, relevant)
            # Gap: confidence - recall (positive = overconfident, negative = underconfident)
            gap = confidence - recall
            confidence_gaps.append(gap)
            
            if gap > 0.2:  # Overconfident threshold
                overconfident_count += 1
            elif gap < -0.2:  # Underconfident threshold
                underconfident_count += 1
    
    total = len(confidence_gaps)
    
    return {
        'avg_confidence_gap': np.mean(confidence_gaps) if confidence_gaps else 0.0,
        'overconfidence_rate': overconfident_count / total if total > 0 else 0.0,
        'underconfidence_rate': underconfident_count / total if total > 0 else 0.0
    }


def calculate_source_quality_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze the distribution of source quality scores.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with source quality distribution metrics
    """
    if not results:
        return {
            'reliable_source_rate': 0.0,
            'unreliable_source_rate': 0.0,
            'unknown_source_rate': 0.0,
            'avg_source_quality': 0.0,
            'source_quality_variance': 0.0
        }
    
    quality_scores = []
    source_counts = {'reliable': 0, 'unreliable': 0, 'unknown': 0}
    total_sources = 0
    
    for result in results:
        retrieved = result.get('retrieved_docs', [])
        if retrieved:
            for doc in retrieved:
                if isinstance(doc, dict):
                    category = doc.get('category', 'unknown').lower()
                    source_counts[category] = source_counts.get(category, 0) + 1
                    total_sources += 1
                    
                    # Calculate quality score for this document
                    quality = source_quality_score([doc])
                    quality_scores.append(quality)
    
    if total_sources == 0:
        return {
            'reliable_source_rate': 0.0,
            'unreliable_source_rate': 0.0,
            'unknown_source_rate': 0.0,
            'avg_source_quality': 0.0,
            'source_quality_variance': 0.0
        }
    
    return {
        'reliable_source_rate': source_counts['reliable'] / total_sources,
        'unreliable_source_rate': source_counts['unreliable'] / total_sources,
        'unknown_source_rate': source_counts['unknown'] / total_sources,
        'avg_source_quality': np.mean(quality_scores) if quality_scores else 0.0,
        'source_quality_variance': np.var(quality_scores) if quality_scores else 0.0
    }


def calculate_hedge_effectiveness(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate how effective hedging language is at indicating uncertainty.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with hedge effectiveness metrics
    """
    if not results:
        return {
            'hedge_uncertainty_correlation': 0.0,
            'hedge_accuracy_correlation': 0.0,
            'hedge_confidence_correlation': 0.0,
            'effective_hedge_rate': 0.0
        }
    
    hedge_predictions = []
    uncertainties = []
    accuracies = []
    confidences = []
    
    for result in results:
        prediction = result.get('prediction_text', '')
        uncertainty = result.get('is_uncertain', False)
        accuracy = result.get('accuracy', 0.0)
        confidence = result.get('confidence', 0.0)
        
        has_hedge = contains_hedge(prediction)
        
        hedge_predictions.append(has_hedge)
        uncertainties.append(uncertainty)
        accuracies.append(accuracy)
        confidences.append(confidence)
    
    # Calculate correlations
    hedge_uncertainty_corr = confidence_accuracy_correlation(hedge_predictions, uncertainties)
    hedge_accuracy_corr = confidence_accuracy_correlation(hedge_predictions, accuracies)
    hedge_confidence_corr = confidence_accuracy_correlation(hedge_predictions, confidences)
    
    # Calculate effective hedge rate (hedges that correctly indicate uncertainty)
    effective_hedges = 0
    total_hedges = 0
    
    for hedge, uncertain in zip(hedge_predictions, uncertainties):
        if hedge:
            total_hedges += 1
            if uncertain:
                effective_hedges += 1
    
    effective_hedge_rate = effective_hedges / total_hedges if total_hedges > 0 else 0.0
    
    return {
        'hedge_uncertainty_correlation': hedge_uncertainty_corr,
        'hedge_accuracy_correlation': hedge_accuracy_corr,
        'hedge_confidence_correlation': hedge_confidence_corr,
        'effective_hedge_rate': effective_hedge_rate
    }


def calculate_calibration_improvement_potential(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate the potential for calibration improvement.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with calibration improvement metrics
    """
    if not results:
        return {
            'calibration_improvement_potential': 0.0,
            'overconfidence_severity': 0.0,
            'underconfidence_severity': 0.0,
            'calibration_consistency': 0.0
        }
    
    confidences = [r.get('confidence', 0.0) for r in results]
    accuracies = [r.get('accuracy', 0.0) for r in results]
    
    # Calculate current ECE
    current_ece = expected_calibration_error(confidences, accuracies)
    
    # Calculate isotonic calibration improvement
    _, ece_after_isotonic = isotonic_calibration(confidences, accuracies)
    improvement_potential = current_ece - ece_after_isotonic
    
    # Calculate overconfidence and underconfidence severity
    overconfident_errors = []
    underconfident_errors = []
    
    for conf, acc in zip(confidences, accuracies):
        error = abs(conf - acc)
        if conf > acc:  # Overconfident
            overconfident_errors.append(error)
        elif conf < acc:  # Underconfident
            underconfident_errors.append(error)
    
    overconfidence_severity = np.mean(overconfident_errors) if overconfident_errors else 0.0
    underconfidence_severity = np.mean(underconfident_errors) if underconfident_errors else 0.0
    
    # Calculate calibration consistency (lower variance = more consistent)
    calibration_errors = [abs(c - a) for c, a in zip(confidences, accuracies)]
    calibration_consistency = 1.0 - np.std(calibration_errors)  # Higher = more consistent
    
    return {
        'calibration_improvement_potential': improvement_potential,
        'overconfidence_severity': overconfidence_severity,
        'underconfidence_severity': underconfidence_severity,
        'calibration_consistency': max(0.0, calibration_consistency)
    }


def calculate_comprehensive_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate comprehensive retrieval metrics including quality, diversity, and calibration.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with comprehensive retrieval metrics
    """
    if not results:
        return {}
    
    # Basic retrieval metrics
    basic_metrics = retrieval_quality_metrics_enhanced(results)
    
    # Confidence gap metrics
    confidence_gap_metrics = calculate_retrieval_confidence_gap(results)
    
    # Source quality distribution
    quality_dist_metrics = calculate_source_quality_distribution(results)
    
    # Combine all metrics
    comprehensive_metrics = {}
    comprehensive_metrics.update(basic_metrics)
    comprehensive_metrics.update(confidence_gap_metrics)
    comprehensive_metrics.update(quality_dist_metrics)
    
    return comprehensive_metrics


def calculate_all_enhanced_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate all enhanced CALM-RAG metrics including utility functions.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with all enhanced metrics
    """
    if not results:
        return {}
    
    # Core CALM-RAG metrics
    core_metrics = compute_all_calm_rag_metrics(results)
    
    # Additional utility metrics
    utility_metrics = {}
    
    # Hedge effectiveness
    hedge_metrics = calculate_hedge_effectiveness(results)
    for key, value in hedge_metrics.items():
        utility_metrics[f'hedge_{key}'] = value
    
    # Calibration improvement potential
    calibration_metrics = calculate_calibration_improvement_potential(results)
    for key, value in calibration_metrics.items():
        utility_metrics[f'calibration_{key}'] = value
    
    # Comprehensive retrieval metrics
    retrieval_metrics = calculate_comprehensive_retrieval_metrics(results)
    for key, value in retrieval_metrics.items():
        if key not in core_metrics:  # Avoid duplicates
            utility_metrics[f'retrieval_{key}'] = value
    
    # Combine all metrics
    all_metrics = {}
    all_metrics.update(core_metrics)
    all_metrics.update(utility_metrics)
    
    return all_metrics


# Enhanced hedge detection with more sophisticated patterns
def contains_enhanced_hedge(text: str) -> Tuple[bool, List[str]]:
    """
    Enhanced hedge detection with pattern matching and confidence scoring.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Tuple of (has_hedge, list_of_hedge_terms_found)
    """
    if not text:
        return False, []
    
    text_lower = text.lower()
    found_hedges = []
    
    # Basic hedge terms
    for hedge in HEDGE_TERMS:
        if hedge in text_lower:
            found_hedges.append(hedge)
    
    # Enhanced patterns
    enhanced_patterns = [
        r'\b(?:it\s+(?:appears|seems|looks))\b',
        r'\b(?:based\s+on\s+(?:the\s+)?(?:evidence|data|research))\b',
        r'\b(?:according\s+to\s+(?:some|recent|preliminary))\b',
        r'\b(?:may\s+(?:or\s+may\s+not|be|have))\b',
        r'\b(?:could\s+(?:potentially|possibly|theoretically))\b',
        r'\b(?:suggests?\s+(?:that|a|an))\b',
        r'\b(?:indicates?\s+(?:that|a|an))\b',
        r'\b(?:evidence\s+(?:suggests|indicates|points\s+to))\b',
        r'\b(?:studies?\s+(?:suggest|indicate|show))\b',
        r'\b(?:research\s+(?:suggests|indicates|demonstrates))\b'
    ]
    
    import re
    for pattern in enhanced_patterns:
        if re.search(pattern, text_lower):
            found_hedges.append(f"pattern_{pattern[:20]}...")
    
    has_hedge = len(found_hedges) > 0
    return has_hedge, found_hedges


def calculate_hedge_sophistication(texts: List[str]) -> Dict[str, float]:
    """
    Calculate hedge sophistication metrics.
    
    Args:
        texts: List of text strings to analyze
    
    Returns:
        Dictionary with hedge sophistication metrics
    """
    if not texts:
        return {
            'avg_hedge_count': 0.0,
            'hedge_variety': 0.0,
            'hedge_density': 0.0,
            'sophisticated_hedge_rate': 0.0
        }
    
    total_hedge_count = 0
    all_hedge_types = set()
    sophisticated_hedge_count = 0
    
    for text in texts:
        if not text:
            continue
        
        has_hedge, hedge_terms = contains_enhanced_hedge(text)
        if has_hedge:
            total_hedge_count += len(hedge_terms)
            all_hedge_types.update(hedge_terms)
            
            # Count sophisticated hedges (multi-word patterns)
            sophisticated_count = sum(1 for h in hedge_terms if 'pattern_' in h or ' ' in h)
            sophisticated_hedge_count += sophisticated_count
    
    total_texts = len([t for t in texts if t])
    
    return {
        'avg_hedge_count': total_hedge_count / total_texts if total_texts > 0 else 0.0,
        'hedge_variety': len(all_hedge_types),
        'hedge_density': total_hedge_count / sum(len(t.split()) for t in texts if t) if any(t for t in texts) else 0.0,
        'sophisticated_hedge_rate': sophisticated_hedge_count / total_hedge_count if total_hedge_count > 0 else 0.0
    }


# Enhanced testing examples with all new metrics
def test_all_metrics():
    """
    Comprehensive test of all CALM-RAG metrics with realistic examples.
    """
    print("=== COMPREHENSIVE CALM-RAG METRICS TESTING ===\n")
    
    # Test data with all required fields for comprehensive evaluation
    comprehensive_test_data = [
        {
            # Example 1: Well-calibrated with good retrieval
            "id": 1,
            "question": "What is the efficacy of Drug X?",
            "retrieved_docs": [
                {"url": "https://nejm.org/study1", "category": "reliable", "domain": "nejm.org"},
                {"url": "https://pubmed.gov/study2", "category": "reliable", "domain": "pubmed.gov"}
            ],
            "relevant_docs": [
                {"url": "https://nejm.org/study1", "category": "reliable"},
                {"url": "https://pubmed.gov/study2", "category": "reliable"}
            ],
            "confidence": 0.85,
            "accuracy": 1.0,
            "prediction_text": "Based on the evidence, Drug X shows approximately 75% efficacy in clinical trials.",
            "is_uncertain": False,
            "human_confidence": 0.8
        },
        {
            # Example 2: Overconfident with poor retrieval
            "id": 2,
            "question": "What are the side effects of Drug Y?",
            "retrieved_docs": [
                {"url": "https://blog.com/random", "category": "unreliable", "domain": "blog.com"},
                {"url": "https://wiki.org/page", "category": "unknown", "domain": "wiki.org"}
            ],
            "relevant_docs": [
                {"url": "https://nejm.org/study3", "category": "reliable"},
                {"url": "https://jama.org/study4", "category": "reliable"}
            ],
            "confidence": 0.95,
            "accuracy": 0.0,
            "prediction_text": "Drug Y definitely causes severe side effects in 90% of patients.",
            "is_uncertain": False,
            "human_confidence": 0.3
        },
        {
            # Example 3: Appropriately uncertain with hedging
            "id": 3,
            "question": "What is the long-term safety profile?",
            "retrieved_docs": [
                {"url": "https://cochrane.org/review", "category": "reliable", "domain": "cochrane.org"},
                {"url": "https://nejm.org/conflicting", "category": "reliable", "domain": "nejm.org"}
            ],
            "relevant_docs": [
                {"url": "https://cochrane.org/review", "category": "reliable"},
                {"url": "https://nejm.org/conflicting", "category": "reliable"}
            ],
            "confidence": 0.60,
            "accuracy": 1.0,
            "prediction_text": "The evidence suggests that the long-term safety profile may be acceptable, though results are somewhat mixed across studies.",
            "is_uncertain": True,
            "human_confidence": 0.6
        },
        {
            # Example 4: No retrieval data available
            "id": 4,
            "question": "What is the mechanism of action?",
            "retrieved_docs": [],
            "relevant_docs": [],
            "confidence": 0.70,
            "accuracy": 0.0,
            "prediction_text": "The mechanism involves receptor binding and signal transduction pathways.",
            "is_uncertain": False,
            "human_confidence": 0.4
        }
    ]
    
    print("1. Testing Core CALM-RAG Metrics (H1-H5)...")
    core_metrics = compute_all_calm_rag_metrics(comprehensive_test_data)
    
    # Print key metrics by hypothesis
    print("\nH1 - Overconfidence Under Sparse/Noisy Evidence:")
    for key, value in core_metrics.items():
        if key.startswith('h1_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nH2 - Calibration Difference with/without Retrieval:")
    for key, value in core_metrics.items():
        if key.startswith('h2_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nH3 - Hedging Language as Uncertainty Signal:")
    for key, value in core_metrics.items():
        if key.startswith('h3_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nH4 - Self-Assessment Calibration:")
    for key, value in core_metrics.items():
        if key.startswith('h4_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nH5 - Source Quality Impact:")
    for key, value in core_metrics.items():
        if key.startswith('h5_'):
            print(f"  {key}: {value:.3f}")
    
    print("\n2. Testing Enhanced Utility Metrics...")
    enhanced_metrics = calculate_all_enhanced_metrics(comprehensive_test_data)
    
    print("\nHedge Effectiveness:")
    for key, value in enhanced_metrics.items():
        if key.startswith('hedge_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nCalibration Improvement Potential:")
    for key, value in enhanced_metrics.items():
        if key.startswith('calibration_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nComprehensive Retrieval Metrics:")
    for key, value in enhanced_metrics.items():
        if key.startswith('retrieval_') and key not in core_metrics:
            print(f"  {key}: {value:.3f}")
    
    print("\n3. Testing Individual Metric Functions...")
    
    # Test hedge detection
    test_texts = [
        "This is definitely correct.",
        "It appears to be true based on the evidence.",
        "The research suggests this might be accurate.",
        "This is absolutely certain."
    ]
    
    print(f"\nHedge Detection Test:")
    for text in test_texts:
        has_hedge = contains_hedge(text)
        has_enhanced, hedge_terms = contains_enhanced_hedge(text)
        print(f"  Text: '{text[:50]}...'")
        print(f"    Basic hedge: {has_hedge}")
        print(f"    Enhanced hedge: {has_enhanced} (terms: {hedge_terms})")
    
    # Test source quality analysis
    print(f"\nSource Quality Analysis:")
    quality_dist = calculate_source_quality_distribution(comprehensive_test_data)
    for key, value in quality_dist.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n=== METRICS TESTING COMPLETE ===")
    return core_metrics, enhanced_metrics


def get_metrics_summary() -> Dict[str, List[str]]:
    """
    Get a comprehensive summary of all available CALM-RAG metrics.
    
    Returns:
        Dictionary organizing metrics by category
    """
    return {
        "H1 - Overconfidence Under Sparse/Noisy Evidence": [
            "retrieval_recall_confidence_correlation",
            "avg_retrieval_recall", 
            "overconfidence_index",
            "wrong_answer_rate",
            "refusal_rate"
        ],
        "H2 - Calibration Difference with/without Retrieval": [
            "ece_with_retrieval",
            "ece_without_retrieval", 
            "ece_difference",
            "brier_with_retrieval",
            "brier_without_retrieval",
            "brier_difference",
            "confidence_accuracy_corr_with_retrieval",
            "confidence_accuracy_corr_without_retrieval",
            "correlation_difference"
        ],
        "H3 - Hedging Language as Uncertainty Signal": [
            "hedge_precision",
            "hedge_recall",
            "hedge_f1",
            "lexical_overconfidence_index",
            "uncertainty_confidence_correlation",
            "hedge_density",
            "confident_wrong_rate",
            "hedge_sophistication",
            "enhanced_hedge_detection_rate"
        ],
        "H4 - Self-Assessment and Numeric Confidence Calibration": [
            "expected_calibration_error",
            "brier_score",
            "confidence_accuracy_correlation",
            "calibration_ece_after_isotonic",
            "calibration_improvement",
            "confidence_distribution_entropy",
            "human_model_confidence_correlation"
        ],
        "H5 - Source Quality Impact on Calibration": [
            "source_quality_confidence_correlation",
            "source_diversity_calibration_correlation",
            "quality_weighted_ece",
            "high_quality_source_ece",
            "low_quality_source_ece",
            "quality_calibration_gap"
        ],
        "Enhanced Retrieval Metrics": [
            "avg_retrieval_recall",
            "avg_retrieval_precision", 
            "avg_retrieval_f1",
            "recall_confidence_correlation",
            "accuracy_confidence_correlation",
            "avg_source_quality",
            "source_quality_std",
            "source_quality_confidence_correlation",
            "avg_source_diversity",
            "source_diversity_std",
            "source_diversity_accuracy_correlation"
        ],
        "Utility Metrics": [
            "hedge_hedge_uncertainty_correlation",
            "hedge_hedge_accuracy_correlation",
            "hedge_hedge_confidence_correlation", 
            "hedge_effective_hedge_rate",
            "calibration_calibration_improvement_potential",
            "calibration_overconfidence_severity",
            "calibration_underconfidence_severity",
            "calibration_calibration_consistency",
            "retrieval_avg_confidence_gap",
            "retrieval_overconfidence_rate",
            "retrieval_underconfidence_rate",
            "reliable_source_rate",
            "unreliable_source_rate",
            "unknown_source_rate",
            "avg_source_quality",
            "source_quality_variance"
        ]
    }


def print_metrics_summary():
    """
    Print a comprehensive summary of all available CALM-RAG metrics.
    """
    summary = get_metrics_summary()
    
    print("=== CALM-RAG METRICS COMPREHENSIVE SUMMARY ===\n")
    print("This implementation provides complete coverage of all CALM-RAG hypotheses (H1-H5)\n")
    
    total_metrics = 0
    for category, metrics in summary.items():
        print(f"{category}:")
        for metric in metrics:
            print(f"  - {metric}")
        print(f"  Total: {len(metrics)} metrics\n")
        total_metrics += len(metrics)
    
    print(f"TOTAL METRICS IMPLEMENTED: {total_metrics}")
    print("\n=== USAGE EXAMPLES ===")
    print("1. Core CALM-RAG metrics: compute_all_calm_rag_metrics(results)")
    print("2. Enhanced metrics: calculate_all_enhanced_metrics(results)")
    print("3. Individual hypothesis metrics: calm_rag_h1_metrics(results)")
    print("4. Utility functions: calculate_hedge_effectiveness(results)")
    print("\n=== DATA SCHEMA REQUIRED ===")
    print("Each result should contain:")
    print("  - retrieved_docs: List of retrieved documents")
    print("  - relevant_docs: List of ground-truth relevant documents")
    print("  - confidence: Model confidence score [0,1]")
    print("  - accuracy: Binary correctness [0,1]")
    print("  - prediction_text: Model's answer text")
    print("  - is_uncertain: Boolean uncertainty label")
    print("  - human_confidence: Human confidence score (optional)")
    print("  - no_retrieval_confidence: Confidence without retrieval (optional)")
    print("  - no_retrieval_accuracy: Accuracy without retrieval (optional)")


if __name__ == "__main__":
    # Print comprehensive summary
    print_metrics_summary()
    
    print("\n" + "="*60 + "\n")
    
    # Run comprehensive testing
    core_results, enhanced_results = test_all_metrics()
    
    print(f"\nTotal Core Metrics: {len(core_results)}")
    print(f"Total Enhanced Metrics: {len(enhanced_results)}")
    
    # Example usage of specific metrics
    print(f"\nExample - H1 Retrieval Recall: {core_results.get('h1_avg_retrieval_recall', 0.0):.3f}")
    print(f"Example - H3 Hedge Precision: {core_results.get('h3_hedge_precision', 0.0):.3f}")
    print(f"Example - Calibration Improvement: {enhanced_results.get('calibration_calibration_improvement_potential', 0.0):.3f}")
    
    print("\n" + "="*60)
    print("CALM-RAG METRICS IMPLEMENTATION COMPLETE!")
    print("All hypotheses H1-H5 are now fully implemented with comprehensive metrics.")
    print("="*60)