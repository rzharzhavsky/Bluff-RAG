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
import string


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
    
    # Check if arrays are constant (all values the same)
    if len(set(valid_recalls)) == 1 or len(set(valid_confidences)) == 1:
        return 0.0
    
    correlation, _ = pearsonr(valid_recalls, valid_confidences)
    return correlation if not np.isnan(correlation) else 0.0


def retrieval_quality_metrics(results: List[Dict[str, Any]], include_correlations: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive retrieval quality metrics with optional correlation features.
    
    Args:
        results: List of result dicts with CALM-RAG schema
        include_correlations: Whether to include correlation metrics
    
    Returns:
        Dictionary of retrieval quality metrics
    """
    if not results:
        return {}
    
    # Extract data using utility functions
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results) if include_correlations else []
    retrieved_docs_list, relevant_docs_list = extract_retrieval_data(results)
    
    # Calculate basic retrieval metrics
    recalls = []
    precisions = []
    f1_scores = []
    quality_scores = []
    diversity_scores = []
    
    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        # Basic retrieval metrics
        recall = retrieval_recall(retrieved, relevant)
        precision = retrieval_precision(retrieved, relevant)
        f1 = retrieval_f1(retrieved, relevant)
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        
        # Source quality (if available)
        if retrieved and isinstance(retrieved[0], dict):
            quality = source_quality_score(retrieved)
            diversity = retrieval_diversity(retrieved)
            quality_scores.append(quality)
            diversity_scores.append(diversity)
    
    # Build metrics dictionary
    metrics = {
        'avg_retrieval_recall': np.mean(recalls) if recalls else 0.0,
        'avg_retrieval_precision': np.mean(precisions) if precisions else 0.0,
        'avg_retrieval_f1': np.mean(f1_scores) if f1_scores else 0.0,
    }
    
    # Add confidence-recall correlation
    metrics['recall_confidence_correlation'] = recall_confidence_correlation(recalls, confidences)
    
    # Add correlation metrics if requested
    if include_correlations and accuracies:
        metrics['accuracy_confidence_correlation'] = confidence_accuracy_correlation(confidences, accuracies)
    
    # Add source quality metrics (if available)
    if quality_scores:
        metrics['avg_source_quality'] = np.mean(quality_scores)
        metrics['source_quality_std'] = np.std(quality_scores)
        if include_correlations:
            metrics['source_quality_confidence_correlation'] = confidence_accuracy_correlation(quality_scores, confidences)
    
    if diversity_scores:
        metrics['avg_source_diversity'] = np.mean(diversity_scores)
        metrics['source_diversity_std'] = np.std(diversity_scores)
        if include_correlations:
            metrics['source_diversity_accuracy_correlation'] = confidence_accuracy_correlation(diversity_scores, accuracies)
    
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
         Proportional OCI between 0 and 1, where:
        - 0 = no overconfidence
        - 1 = maximum overconfidence (all high-confidence predictions are completely wrong)
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidences and accuracies must have same length")
    
    # Find high-confidence predictions
    high_conf_mask = np.array(confidences) >= tau
    high_conf_count = np.sum(high_conf_mask)
    
    if high_conf_count == 0:
        return 0.0
    
    # Calculate overconfidence for each high-confidence prediction
    overconfidence_scores = []
    
    for i in range(len(confidences)):
        if high_conf_mask[i]:
            conf = confidences[i]
            acc = accuracies[i]
            
            if conf > acc:  # Overconfident
                # Calculate proportional overconfidence
                # If confidence=0.9 and accuracy=0.3, overconfidence = (0.9-0.3)/0.9 = 0.67
                overconfidence = (conf - acc) / conf
                overconfidence_scores.append(overconfidence)
            else:  # Not overconfident (confidence <= accuracy)
                overconfidence_scores.append(0.0)
    
    # Return average proportional overconfidence
    return np.mean(overconfidence_scores)


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


def contains_hedge(text: str) -> int:
    """
    Check if text contains hedge terms.
    From CALM-RAG H3 hypothesis.
    
    Args:
        text: Input text to analyze
    
    Returns:
        count of hedge terms found
    """
    if not text:
        return 0
    
    text_lower = text.lower()
    hedge_count = 0
    
    for hedge in HEDGE_TERMS:
        hedge_count += text_lower.count(hedge)
    
    return hedge_count


def hedge_precision_recall(predictions: List[str], true_uncertainties: List[float]) -> Tuple[float, float]:
    """
    Calculate precision and recall for hedge detection using continuous hedge counts and uncertainty scores.
    From CALM-RAG H3 hypothesis.
    
    Args:
        predictions: List of prediction texts
        true_uncertainties: List of uncertainty scores (0-1, where 1 = most it ought to beuncertain)
    
    Returns:
        Tuple of (precision, recall)
    """
    if len(predictions) != len(true_uncertainties):
        raise ValueError("Predictions and uncertainties must have same length")
    
    # Get hedge counts for each prediction (continuous values)
    hedge_counts = [contains_hedge(pred) for pred in predictions]
    
    # Calculate weighted precision and recall using continuous values
    weighted_tp = 0.0  # Weighted true positives (hedge when should be uncertain)
    weighted_fp = 0.0  # Weighted false positives (hedge when should be confident)
    weighted_fn = 0.0  # Weighted false negatives (no hedge when should be uncertain)
    weighted_tn = 0.0  # Weighted true negatives (no hedge when should be confident)
    
    for hedge_count, uncertainty in zip(hedge_counts, true_uncertainties):
        if hedge_count > 0:  # System used hedging
            if uncertainty > 0.5:  # Should be uncertain
                weighted_tp += uncertainty * hedge_count  # Weight by both uncertainty and hedge intensity
            else:  # Should be confident
                weighted_fp += (1.0 - uncertainty) * hedge_count  # Weight by confidence and hedge intensity
        else:  # System didn't use hedging
            if uncertainty > 0.5:  # Should be uncertain
                weighted_fn += uncertainty  # Weight by uncertainty level
            else:  # Should be confident
                weighted_tn += (1.0 - uncertainty)  # Weight by confidence level
    
    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    
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
    
    # Check if arrays are constant (all values the same)
    if len(set(valid_confidences)) == 1 or len(set(valid_accuracies)) == 1:
        return 0.0
    
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
    confident_terms = ["definitely", "certainly", "clearly", "obviously", "undoubtedly", "absolutely", "For sure", "No doubt"]
    
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
    
    # Extract data using utility functions
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results)
    prediction_texts = extract_prediction_texts(results)
    retrieved_docs_list, relevant_docs_list = extract_retrieval_data(results)
    
    # Calculate retrieval recall scores
    recall_scores = []
    wrong_answers = 0
    refusals = 0
    
    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        if not relevant:
            recall_i = 1.0
        else:
            retrieved_ids = {normalize_document_id(doc) for doc in retrieved}
            relevant_ids = {normalize_document_id(doc) for doc in relevant}
            intersection = retrieved_ids.intersection(relevant_ids)
            recall_i = len(intersection) / len(relevant_ids)
        recall_scores.append(recall_i)
    
    # Count refusals and wrong answers
    for answer_text, accuracy in zip(prediction_texts, accuracies):
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
    
    # Extract data using utility functions
    confidences_with = extract_confidence_scores(results)
    accuracies_with = extract_accuracy_scores(results)
    
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
    
    Implements advanced metrics from CALM-RAG proposal section H3:
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
            'advanced_hedge_detection_rate': 0.0
        }
    
    # Extract data using utility functions
    predictions = extract_prediction_texts(results)
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results)
    # Use continuous uncertainty instead of binary
    uncertainties = [r.get('continuous_uncertainty', 0.5) for r in results]
    
    # Basic hedge metrics
    hedge_prec, hedge_rec = hedge_precision_recall(predictions, uncertainties)
    hedge_f1 = 2 * (hedge_prec * hedge_rec) / (hedge_prec + hedge_rec) if (hedge_prec + hedge_rec) > 0 else 0.0
    
    # Advanced hedge detection
    advanced_hedge_count = 0
    total_predictions = 0
    
    for prediction in predictions:
        if prediction:
            total_predictions += 1
            has_advanced_hedge, _ = contains_advanced_hedge(prediction)
            if has_advanced_hedge:
                advanced_hedge_count += 1
    
    advanced_hedge_detection_rate = advanced_hedge_count / total_predictions if total_predictions > 0 else 0.0
    
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
        'advanced_hedge_detection_rate': advanced_hedge_detection_rate
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
    
    # Extract data using utility functions
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results)
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
    
    # Extract data using utility functions
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results)
    retrieved_docs_list, _ = extract_retrieval_data(results)
    
    source_qualities = []
    source_diversities = []
    
    for retrieved in retrieved_docs_list:
        if retrieved:
            # Calculate source quality
            quality = source_quality_score(retrieved)
            diversity = retrieval_diversity(retrieved)
            
            source_qualities.append(quality)
            source_diversities.append(diversity)
    
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


# Advanced hedge detection functions
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


# Retrieval quality metrics
# This function has been consolidated into retrieval_quality_metrics() with include_correlations=True


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
    
    # H6: Faithfulness and Grounding Metrics
    faithfulness_metrics = calm_rag_faithfulness_metrics(results)
    for key, value in faithfulness_metrics.items():
        metrics[f'h6_{key}'] = value
    
    # Retrieval quality metrics (avoid duplicates with hypothesis metrics)
    retrieval_metrics = retrieval_quality_metrics(results, include_correlations=True)
    for key, value in retrieval_metrics.items():
        # Skip metrics that are already included in hypothesis metrics
        if key not in ['avg_retrieval_recall', 'recall_confidence_correlation']:
            metrics[f'retrieval_{key}'] = value
    
    return metrics


# Legacy function removed - use compute_all_calm_rag_metrics() directly


# =============================================================================
# FAITHFULNESS METRICS
# =============================================================================

def calculate_answer_source_overlap(prediction: str, retrieved_docs: List[Dict[str, Any]], 
                                  method: str = "token") -> float:
    """
    Calculate the overlap between the prediction and retrieved source documents.
    
    Args:
        prediction: The model's generated answer
        retrieved_docs: List of retrieved document dictionaries
        method: Overlap calculation method ("token", "ngram", "semantic")
    
    Returns:
        Overlap score between 0 and 1
    """
    if not prediction or not retrieved_docs:
        return 0.0
    
    # Extract text from retrieved documents
    source_texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '')
        else:
            text = str(doc)
        if text:
            source_texts.append(text)
    
    if not source_texts:
        return 0.0
    
    # Combine all source text
    combined_sources = " ".join(source_texts)
    
    if method == "token":
        return _calculate_token_overlap(prediction, combined_sources)
    elif method == "ngram":
        return _calculate_ngram_overlap(prediction, combined_sources)
    elif method == "semantic":
        return _calculate_semantic_overlap(prediction, combined_sources)
    else:
        raise ValueError(f"Unknown overlap method: {method}")


def _calculate_token_overlap(prediction: str, source_text: str) -> float:
    """Calculate token-level overlap between prediction and source."""
    pred_tokens = set(normalize_text(prediction).split())
    source_tokens = set(normalize_text(source_text).split())
    
    if not pred_tokens:
        return 0.0
    
    intersection = pred_tokens.intersection(source_tokens)
    return len(intersection) / len(pred_tokens)


def _calculate_ngram_overlap(prediction: str, source_text: str, n: int = 3) -> float:
    """Calculate n-gram overlap between prediction and source."""
    def get_ngrams(text: str, n: int) -> set:
        tokens = normalize_text(text).split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    pred_ngrams = get_ngrams(prediction, n)
    source_ngrams = get_ngrams(source_text, n)
    
    if not pred_ngrams:
        return 0.0
    
    intersection = pred_ngrams.intersection(source_ngrams)
    return len(intersection) / len(pred_ngrams)


def _calculate_semantic_overlap(prediction: str, source_text: str) -> float:
    """Calculate semantic overlap using simple word embeddings approach."""
    # Simple implementation using word overlap with synonyms
    # In practice, you might want to use more sophisticated embeddings
    pred_tokens = set(normalize_text(prediction).split())
    source_tokens = set(normalize_text(source_text).split())
    
    if not pred_tokens:
        return 0.0
    
    # Basic semantic similarity using word overlap
    intersection = pred_tokens.intersection(source_tokens)
    return len(intersection) / len(pred_tokens)


def calculate_attribution_accuracy(prediction: str, retrieved_docs: List[Dict[str, Any]], 
                                 gold_answer: str = None) -> Dict[str, float]:
    """
    Calculate attribution accuracy - how well claims in the prediction can be attributed to sources.
    
    Args:
        prediction: The model's generated answer
        retrieved_docs: List of retrieved document dictionaries
        gold_answer: Ground truth answer for comparison
    
    Returns:
        Dictionary with attribution metrics
    """
    if not prediction or not retrieved_docs:
        return {
            'attribution_coverage': 0.0,
            'source_utilization': 0.0,
            'claim_attribution_rate': 0.0,
            'attribution_precision': 0.0
        }
    
    # Extract claims from prediction (simple sentence splitting)
    prediction_sentences = [s.strip() for s in prediction.split('.') if s.strip()]
    
    # Extract text from sources
    source_texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '')
        else:
            text = str(doc)
        if text:
            source_texts.append(text)
    
    if not source_texts:
        return {
            'attribution_coverage': 0.0,
            'source_utilization': 0.0,
            'claim_attribution_rate': 0.0,
            'attribution_precision': 0.0
        }
    
    # Calculate attribution metrics
    attributed_claims = 0
    total_claims = len(prediction_sentences)
    
    for claim in prediction_sentences:
        if _can_claim_be_attributed(claim, source_texts):
            attributed_claims += 1
    
    # Source utilization - how many sources are actually used
    used_sources = 0
    for source_text in source_texts:
        if _is_source_used_in_prediction(prediction, source_text):
            used_sources += 1
    
    return {
        'attribution_coverage': attributed_claims / total_claims if total_claims > 0 else 0.0,
        'source_utilization': used_sources / len(source_texts) if source_texts else 0.0,
        'claim_attribution_rate': attributed_claims / total_claims if total_claims > 0 else 0.0,
        'attribution_precision': attributed_claims / total_claims if total_claims > 0 else 0.0
    }


def _can_claim_be_attributed(claim: str, source_texts: List[str]) -> bool:
    """Check if a claim can be attributed to any source text."""
    claim_tokens = set(normalize_text(claim).split())
    
    for source_text in source_texts:
        source_tokens = set(normalize_text(source_text).split())
        # Check if significant portion of claim tokens appear in source
        overlap = len(claim_tokens.intersection(source_tokens))
        if overlap >= max(1, len(claim_tokens) * 0.3):  # At least 30% overlap
            return True
    
    return False


def _is_source_used_in_prediction(prediction: str, source_text: str) -> bool:
    """Check if a source text is used in the prediction."""
    pred_tokens = set(normalize_text(prediction).split())
    source_tokens = set(normalize_text(source_text).split())
    
    # Check for significant overlap
    overlap = len(pred_tokens.intersection(source_tokens))
    return overlap >= max(3, len(source_tokens) * 0.1)  # At least 10% overlap or 3 tokens


def calculate_hallucination_detection(prediction: str, retrieved_docs: List[Dict[str, Any]], 
                                    gold_answer: str = None) -> Dict[str, float]:
    """
    Detect potential hallucinations in the prediction.
    
    Args:
        prediction: The model's generated answer
        retrieved_docs: List of retrieved document dictionaries
        gold_answer: Ground truth answer for comparison
    
    Returns:
        Dictionary with hallucination detection metrics
    """
    if not prediction:
        return {
            'hallucination_rate': 0.0,
            'unsupported_claims_rate': 0.0,
            'hallucination_severity': 0.0,
            'factual_consistency': 0.0
        }
    
    # Extract text from sources
    source_texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '')
        else:
            text = str(doc)
        if text:
            source_texts.append(text)
    
    # Split prediction into claims
    prediction_sentences = [s.strip() for s in prediction.split('.') if s.strip()]
    
    hallucinated_claims = 0
    unsupported_claims = 0
    total_claims = len(prediction_sentences)
    
    for claim in prediction_sentences:
        if not _can_claim_be_attributed(claim, source_texts):
            unsupported_claims += 1
            # Check if it's a clear hallucination (contains specific facts not in sources)
            if _is_potential_hallucination(claim, source_texts):
                hallucinated_claims += 1
    
    # Calculate factual consistency with gold answer if available
    factual_consistency = 1.0
    if gold_answer:
        factual_consistency = _calculate_factual_consistency(prediction, gold_answer)
    
    return {
        'hallucination_rate': hallucinated_claims / total_claims if total_claims > 0 else 0.0,
        'unsupported_claims_rate': unsupported_claims / total_claims if total_claims > 0 else 0.0,
        'hallucination_severity': hallucinated_claims / total_claims if total_claims > 0 else 0.0,
        'factual_consistency': factual_consistency
    }


def _is_potential_hallucination(claim: str, source_texts: List[str]) -> bool:
    """Check if a claim is likely a hallucination."""
    # Look for specific factual claims that should be in sources
    factual_indicators = [
        r'\d+%',  # percentages
        r'\d+\.\d+',  # decimal numbers
        r'\d{4}',  # years
        r'\$\d+',  # monetary amounts
        r'\d+\s+(years?|months?|days?)',  # time periods
    ]
    
    for pattern in factual_indicators:
        if re.search(pattern, claim):
            # If it contains specific facts but can't be attributed, likely hallucination
            return True
    
    return False


def _calculate_factual_consistency(prediction: str, gold_answer: str) -> float:
    """Calculate factual consistency between prediction and gold answer."""
    if not gold_answer:
        return 1.0
    
    # Simple token overlap as a proxy for factual consistency
    pred_tokens = set(normalize_text(prediction).split())
    gold_tokens = set(normalize_text(gold_answer).split())
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    intersection = pred_tokens.intersection(gold_tokens)
    union = pred_tokens.union(gold_tokens)
    
    return len(intersection) / len(union) if union else 0.0


def calculate_source_grounding_metrics(prediction: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate how well the prediction is grounded in the retrieved sources.
    
    Args:
        prediction: The model's generated answer
        retrieved_docs: List of retrieved document dictionaries
    
    Returns:
        Dictionary with grounding metrics
    """
    if not prediction or not retrieved_docs:
        return {
            'grounding_score': 0.0,
            'source_coverage': 0.0,
            'grounding_consistency': 0.0,
            'source_relevance': 0.0
        }
    
    # Extract text from sources
    source_texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '')
        else:
            text = str(doc)
        if text:
            source_texts.append(text)
    
    if not source_texts:
        return {
            'grounding_score': 0.0,
            'source_coverage': 0.0,
            'grounding_consistency': 0.0,
            'source_relevance': 0.0
        }
    
    # Calculate grounding score (average overlap with all sources)
    overlap_scores = []
    for source_text in source_texts:
        overlap = _calculate_token_overlap(prediction, source_text)
        overlap_scores.append(overlap)
    
    grounding_score = np.mean(overlap_scores) if overlap_scores else 0.0
    
    # Source coverage - how many sources are used
    used_sources = sum(1 for score in overlap_scores if score > 0.1)
    source_coverage = used_sources / len(source_texts) if source_texts else 0.0
    
    # Grounding consistency - variance in overlap scores
    grounding_consistency = 1.0 - np.var(overlap_scores) if len(overlap_scores) > 1 else 1.0
    
    # Source relevance - average relevance of used sources
    source_relevance = np.mean([score for score in overlap_scores if score > 0.1]) if any(score > 0.1 for score in overlap_scores) else 0.0
    
    return {
        'grounding_score': grounding_score,
        'source_coverage': source_coverage,
        'grounding_consistency': grounding_consistency,
        'source_relevance': source_relevance
    }


def calm_rag_faithfulness_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate comprehensive faithfulness metrics for CALM-RAG evaluation.
    
    Args:
        results: List of result dictionaries with CALM-RAG schema
    
    Returns:
        Dictionary of faithfulness metrics
    """
    if not results:
        return {}
    
    # Initialize metric accumulators
    overlap_scores = []
    attribution_metrics = {
        'attribution_coverage': [],
        'source_utilization': [],
        'claim_attribution_rate': [],
        'attribution_precision': []
    }
    hallucination_metrics = {
        'hallucination_rate': [],
        'unsupported_claims_rate': [],
        'hallucination_severity': [],
        'factual_consistency': []
    }
    grounding_metrics = {
        'grounding_score': [],
        'source_coverage': [],
        'grounding_consistency': [],
        'source_relevance': []
    }
    
    # Process each result
    for result in results:
        prediction = result.get('prediction_text', '')
        retrieved_docs = result.get('retrieved_docs', [])
        gold_answer = result.get('gold_answer', '')
        
        # Calculate answer-source overlap
        overlap = calculate_answer_source_overlap(prediction, retrieved_docs, method="token")
        overlap_scores.append(overlap)
        
        # Calculate attribution accuracy
        attribution = calculate_attribution_accuracy(prediction, retrieved_docs, gold_answer)
        for key, value in attribution.items():
            attribution_metrics[key].append(value)
        
        # Calculate hallucination detection
        hallucination = calculate_hallucination_detection(prediction, retrieved_docs, gold_answer)
        for key, value in hallucination.items():
            hallucination_metrics[key].append(value)
        
        # Calculate source grounding
        grounding = calculate_source_grounding_metrics(prediction, retrieved_docs)
        for key, value in grounding.items():
            grounding_metrics[key].append(value)
    
    # Calculate average metrics
    metrics = {}
    
    # Answer-source overlap metrics
    metrics['answer_source_overlap'] = np.mean(overlap_scores) if overlap_scores else 0.0
    metrics['answer_source_overlap_std'] = np.std(overlap_scores) if overlap_scores else 0.0
    
    # Attribution metrics
    for key, values in attribution_metrics.items():
        metrics[f'attribution_{key}'] = np.mean(values) if values else 0.0
        metrics[f'attribution_{key}_std'] = np.std(values) if values else 0.0
    
    # Hallucination metrics
    for key, values in hallucination_metrics.items():
        metrics[f'hallucination_{key}'] = np.mean(values) if values else 0.0
        metrics[f'hallucination_{key}_std'] = np.std(values) if values else 0.0
    
    # Grounding metrics
    for key, values in grounding_metrics.items():
        metrics[f'grounding_{key}'] = np.mean(values) if values else 0.0
        metrics[f'grounding_{key}_std'] = np.std(values) if values else 0.0
    
    # Overall faithfulness score (composite metric)
    faithfulness_components = [
        metrics['answer_source_overlap'],
        1.0 - metrics['hallucination_hallucination_rate'],  # Invert hallucination rate
        metrics['attribution_attribution_coverage'],
        metrics['grounding_grounding_score']
    ]
    metrics['overall_faithfulness'] = np.mean(faithfulness_components)
    
    return metrics


# =============================================================================
# AMBIGUITY SENSITIVITY INDEX (ASI) METRICS
# =============================================================================

def calculate_ambiguity_sensitivity_index(clear_entry: Dict[str, Any], ambiguous_entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the Ambiguity Sensitivity Index (ASI) for a question with both clear and ambiguous evidence.
    
    ASI measures whether the model adapts appropriately when evidence is ambiguous, conflicting, or misleading
    compared to when evidence is clear and credible.
    
    Args:
        clear_entry: Dictionary containing clear set results with keys:
                    - prediction_text: model's answer
                    - confidence: scalar probability 0-1
                    - prediction_explanation: model reasoning (optional)
                    - verbal_uncertainty_flags: list of hedging words
                    - faithfulness_score: 0, 0.5, or 1
                    - abstain_flag: true if model explicitly abstained
                    - set_type: "clear"
        ambiguous_entry: Dictionary containing ambiguous set results with same keys plus:
                        - ambiguity_type: "unanswerable" or "conflicting"
    
    Returns:
        Dictionary containing ASI components and final score:
        - cdr: Confidence Dampening Ratio (0-1)
        - vhl: Verbal Hedging Lift (0-1)
        - ha: Hallucination Avoidance (0-1)
        - aq: Abstention Quality (0-1)
        - asi: Final ASI score (0-1)
        - components: Detailed breakdown for debugging
    """
    
    # Extract confidence scores with fallback handling
    conf_c = clear_entry.get('confidence', 0.0)
    conf_a = ambiguous_entry.get('confidence', 0.0)
    
    # Ensure confidence values are valid floats in [0,1] range
    conf_c = max(0.0, min(1.0, float(conf_c) if conf_c is not None else 0.0))
    conf_a = max(0.0, min(1.0, float(conf_a) if conf_a is not None else 0.0))
    
    # 1. CONFIDENCE DAMPENING RATIO (CDR)
    # Expect confidence to drop on ambiguous set compared to clear set
    if conf_c == 0:
        cdr = 1.0  # Perfect score when clear confidence is 0
    else:
        # Clip the ratio to [0, 1] range
        raw_cdr = (conf_c - conf_a) / conf_c
        cdr = max(0.0, min(1.0, raw_cdr))
    
    # 2. VERBAL HEDGING LIFT (VHL)
    # Count hedging words in both sets
    hedging_c = len(clear_entry.get('verbal_uncertainty_flags', []))
    hedging_a = len(ambiguous_entry.get('verbal_uncertainty_flags', []))
    
    if hedging_a > hedging_c:
        vhl = 1.0  # More hedging on ambiguous set (good)
    elif hedging_a == hedging_c:
        vhl = 0.5  # Same hedging (neutral)
    else:
        vhl = 0.0  # Less hedging on ambiguous set (bad)
    
    # 3. HALLUCINATION AVOIDANCE (HA)
    # Faithfulness score on ambiguous set only
    ha = ambiguous_entry.get('faithfulness_score', 0.0)
    # Ensure faithfulness_score is valid
    if ha not in [0.0, 0.5, 1.0]:
        ha = 0.0  # Default to worst case if invalid
    
    # 4. ABSTENTION QUALITY (AQ)
    # Uses ambiguity_type to determine appropriate behavior
    ambiguity_type = ambiguous_entry.get('ambiguity_type', 'conflicting')
    abstain_flag = ambiguous_entry.get('abstain_flag', False)
    
    if ambiguity_type == 'unanswerable':
        # For unanswerable questions: reward abstention, penalize overconfident answers
        if abstain_flag:
            aq = 1.0  # Perfect abstention
        elif conf_a > 0.7:
            aq = 0.0  # Overconfident on unanswerable (bad)
        else:
            aq = 0.5  # Moderate confidence on unanswerable (neutral)
    
    elif ambiguity_type == 'conflicting':
        # For conflicting evidence: reward hedged, faithful answers; penalize overconfident, unfaithful
        if abstain_flag:
            aq = 0.8  # Good abstention on conflicting evidence
        elif ha >= 0.5 and hedging_a > 0:
            aq = 1.0  # Faithful and hedged (excellent)
        elif ha >= 0.5:
            aq = 0.7  # Faithful but not hedged (good)
        elif hedging_a > 0:
            aq = 0.5  # Hedged but not faithful (neutral)
        else:
            aq = 0.0  # Neither faithful nor hedged (bad)
    
    else:
        # Unknown ambiguity type - default to neutral
        aq = 0.5
    
    # COMBINE COMPONENTS WITH WEIGHTS
    # Weights: CDR=0.30, VHL=0.20, HA=0.30, AQ=0.20
    asi = 0.30 * cdr + 0.20 * vhl + 0.30 * ha + 0.20 * aq
    
    # Ensure final ASI score is in [0, 1] range
    asi = max(0.0, min(1.0, asi))
    
    return {
        'cdr': cdr,
        'vhl': vhl,
        'ha': ha,
        'aq': aq,
        'asi': asi,
        'components': {
            'confidence_clear': conf_c,
            'confidence_ambiguous': conf_a,
            'hedging_clear': hedging_c,
            'hedging_ambiguous': hedging_a,
            'faithfulness_ambiguous': ha,
            'abstain_flag': abstain_flag,
            'ambiguity_type': ambiguity_type
        }
    }


def calculate_batch_asi(question_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate ASI metrics for a batch of questions, each with clear and ambiguous entries.
    
    Args:
        question_results: List of dictionaries, each containing:
                         - question_id: unique identifier
                         - clear_entry: results from clear evidence set
                         - ambiguous_entry: results from ambiguous evidence set
    
    Returns:
        Dictionary containing batch ASI statistics:
        - mean_asi: Average ASI score across all questions
        - std_asi: Standard deviation of ASI scores
        - mean_cdr: Average Confidence Dampening Ratio
        - mean_vhl: Average Verbal Hedging Lift
        - mean_ha: Average Hallucination Avoidance
        - mean_aq: Average Abstention Quality
        - individual_scores: List of individual ASI scores for each question
    """
    
    if not question_results:
        return {
            'mean_asi': 0.0,
            'std_asi': 0.0,
            'mean_cdr': 0.0,
            'mean_vhl': 0.0,
            'mean_ha': 0.0,
            'mean_aq': 0.0,
            'individual_scores': []
        }
    
    individual_scores = []
    cdr_scores = []
    vhl_scores = []
    ha_scores = []
    aq_scores = []
    
    for result in question_results:
        try:
            clear_entry = result.get('clear_entry', {})
            ambiguous_entry = result.get('ambiguous_entry', {})
            
            if not clear_entry or not ambiguous_entry:
                # Skip incomplete entries
                continue
            
            asi_result = calculate_ambiguity_sensitivity_index(clear_entry, ambiguous_entry)
            
            individual_scores.append(asi_result['asi'])
            cdr_scores.append(asi_result['cdr'])
            vhl_scores.append(asi_result['vhl'])
            ha_scores.append(asi_result['ha'])
            aq_scores.append(asi_result['aq'])
            
        except Exception as e:
            # Skip problematic entries and continue
            print(f"Warning: Skipping ASI calculation for question {result.get('question_id', 'unknown')}: {e}")
            continue
    
    if not individual_scores:
        return {
            'mean_asi': 0.0,
            'std_asi': 0.0,
            'mean_cdr': 0.0,
            'mean_vhl': 0.0,
            'mean_ha': 0.0,
            'mean_aq': 0.0,
            'individual_scores': []
        }
    
    return {
        'mean_asi': np.mean(individual_scores),
        'std_asi': np.std(individual_scores),
        'mean_cdr': np.mean(cdr_scores),
        'mean_vhl': np.mean(vhl_scores),
        'mean_ha': np.mean(ha_scores),
        'mean_aq': np.mean(aq_scores),
        'individual_scores': individual_scores
    }


def test_asi_metrics():
    """Test the ASI metrics with sample data."""
    print("Testing Ambiguity Sensitivity Index (ASI) metrics...")
    
    # Sample data: Clear evidence case
    clear_entry = {
        'prediction_text': 'The capital of France is Paris.',
        'confidence': 0.9,
        'prediction_explanation': 'Based on the clear source material.',
        'verbal_uncertainty_flags': [],  # No hedging needed for clear evidence
        'faithfulness_score': 1.0,
        'abstain_flag': False,
        'set_type': 'clear'
    }
    
    # Sample data: Ambiguous evidence case (conflicting)
    ambiguous_entry_conflicting = {
        'prediction_text': 'The sources suggest Paris might be the capital, though there is some conflicting information.',
        'confidence': 0.6,  # Lower confidence due to ambiguity
        'prediction_explanation': 'The sources present conflicting information about the capital.',
        'verbal_uncertainty_flags': ['suggest', 'might', 'conflicting'],  # More hedging
        'faithfulness_score': 0.5,  # Some faithfulness issues
        'abstain_flag': False,
        'set_type': 'ambiguous',
        'ambiguity_type': 'conflicting'
    }
    
    # Sample data: Ambiguous evidence case (unanswerable)
    ambiguous_entry_unanswerable = {
        'prediction_text': 'I cannot determine the capital from the available sources.',
        'confidence': 0.2,  # Very low confidence
        'prediction_explanation': 'The sources do not contain sufficient information.',
        'verbal_uncertainty_flags': ['cannot', 'determine'],  # Hedging language
        'faithfulness_score': 1.0,  # Faithful to sources
        'abstain_flag': True,  # Explicit abstention
        'set_type': 'ambiguous',
        'ambiguity_type': 'unanswerable'
    }
    
    # Test conflicting case
    print("\n--- Testing Conflicting Evidence Case ---")
    asi_conflicting = calculate_ambiguity_sensitivity_index(clear_entry, ambiguous_entry_conflicting)
    print(f"ASI Score: {asi_conflicting['asi']:.3f}")
    print(f"  CDR (Confidence Dampening): {asi_conflicting['cdr']:.3f}")
    print(f"  VHL (Verbal Hedging Lift): {asi_conflicting['vhl']:.3f}")
    print(f"  HA (Hallucination Avoidance): {asi_conflicting['ha']:.3f}")
    print(f"  AQ (Abstention Quality): {asi_conflicting['aq']:.3f}")
    
    # Test unanswerable case
    print("\n--- Testing Unanswerable Evidence Case ---")
    asi_unanswerable = calculate_ambiguity_sensitivity_index(clear_entry, ambiguous_entry_unanswerable)
    print(f"ASI Score: {asi_unanswerable['asi']:.3f}")
    print(f"  CDR (Confidence Dampening): {asi_unanswerable['cdr']:.3f}")
    print(f"  VHL (Verbal Hedging Lift): {asi_unanswerable['vhl']:.3f}")
    print(f"  HA (Hallucination Avoidance): {asi_unanswerable['ha']:.3f}")
    print(f"  AQ (Abstention Quality): {asi_unanswerable['aq']:.3f}")
    
    # Test batch calculation
    print("\n--- Testing Batch ASI Calculation ---")
    batch_results = [
        {
            'question_id': 'q1',
            'clear_entry': clear_entry,
            'ambiguous_entry': ambiguous_entry_conflicting
        },
        {
            'question_id': 'q2',
            'clear_entry': clear_entry,
            'ambiguous_entry': ambiguous_entry_unanswerable
        }
    ]
    
    batch_asi = calculate_batch_asi(batch_results)
    print(f"Mean ASI: {batch_asi['mean_asi']:.3f}")
    print(f"Std ASI: {batch_asi['std_asi']:.3f}")
    print(f"Mean CDR: {batch_asi['mean_cdr']:.3f}")
    print(f"Mean VHL: {batch_asi['mean_vhl']:.3f}")
    print(f"Mean HA: {batch_asi['mean_ha']:.3f}")
    print(f"Mean AQ: {batch_asi['mean_aq']:.3f}")
    
    print("ASI metrics test completed!")


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
    
    # Extract data using utility functions
    confidences = extract_confidence_scores(results)
    retrieved_docs_list, relevant_docs_list = extract_retrieval_data(results)
    
    confidence_gaps = []
    overconfident_count = 0
    underconfident_count = 0
    
    for retrieved, relevant, confidence in zip(retrieved_docs_list, relevant_docs_list, confidences):
        
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
    
    retrieved_docs_list, _ = extract_retrieval_data(results)
    
    for retrieved in retrieved_docs_list:
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
    
    # Extract data using utility functions
    predictions = extract_prediction_texts(results)
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results)
    
    hedge_predictions = []
    uncertainties = []
    
    for prediction in predictions:
        has_hedge = contains_hedge(prediction)
        hedge_predictions.append(has_hedge)
    
    for result in results:
        uncertainty = result.get('is_uncertain', False)
        uncertainties.append(uncertainty)
    
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
    
    # Extract data using utility functions
    confidences = extract_confidence_scores(results)
    accuracies = extract_accuracy_scores(results)
    
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
    basic_metrics = retrieval_quality_metrics(results, include_correlations=True)
    
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


def calculate_all_utility_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate all CALM-RAG metrics including utility functions.
    
    Args:
        results: List of result dicts with CALM-RAG schema
    
    Returns:
        Dictionary with all utility metrics
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


# Advanced hedge detection with more sophisticated patterns
def contains_advanced_hedge(text: str) -> Tuple[bool, List[str]]:
    """
    Advanced hedge detection with pattern matching and confidence scoring.
    
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
    
    # Advanced patterns
    advanced_patterns = [
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
    for pattern in advanced_patterns:
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
        
        has_hedge, hedge_terms = contains_advanced_hedge(text)
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


# Advanced testing examples with all new metrics
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
    
    print("\n2. Testing Utility Metrics...")
    utility_metrics = calculate_all_utility_metrics(comprehensive_test_data)
    
    print("\nHedge Effectiveness:")
    for key, value in utility_metrics.items():
        if key.startswith('hedge_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nCalibration Improvement Potential:")
    for key, value in utility_metrics.items():
        if key.startswith('calibration_'):
            print(f"  {key}: {value:.3f}")
    
    print("\nComprehensive Retrieval Metrics:")
    for key, value in utility_metrics.items():
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
        has_advanced, hedge_terms = contains_advanced_hedge(text)
        print(f"  Text: '{text[:50]}...'")
        print(f"    Basic hedge: {has_hedge}")
        print(f"    Advanced hedge: {has_advanced} (terms: {hedge_terms})")
    
    # Test source quality analysis
    print(f"\nSource Quality Analysis:")
    quality_dist = calculate_source_quality_distribution(comprehensive_test_data)
    for key, value in quality_dist.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n=== METRICS TESTING COMPLETE ===")
    return core_metrics, utility_metrics


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
            "advanced_hedge_detection_rate"
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
        "Retrieval Metrics": [
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
    print("2. Utility metrics: calculate_all_utility_metrics(results)")
    print("3. Individual hypothesis metrics: calm_rag_h1_metrics(results)")
    print("4. Utility functions: calculate_hedge_effectiveness(results)")
    print("5. Soft accuracy: calculate_soft_accuracy(prediction, gold_answers)")
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


# =====================================================================
# UTILITY FUNCTIONS FOR DATA EXTRACTION
# =====================================================================

def extract_confidence_scores(results: List[Dict[str, Any]]) -> List[float]:
    """Extract confidence scores from results list."""
    return [r.get('confidence', 0.0) for r in results]


def extract_accuracy_scores(results: List[Dict[str, Any]]) -> List[float]:
    """Extract accuracy scores from results list."""
    return [r.get('accuracy', 0.0) for r in results]


def extract_retrieval_data(results: List[Dict[str, Any]]) -> Tuple[List[List], List[List]]:
    """Extract retrieved and relevant documents from results list."""
    retrieved_docs = [r.get('retrieved_docs', []) for r in results]
    relevant_docs = [r.get('relevant_docs', []) for r in results]
    return retrieved_docs, relevant_docs


def extract_prediction_texts(results: List[Dict[str, Any]]) -> List[str]:
    """Extract prediction texts from results list."""
    return [r.get('prediction_text', '') for r in results]


def calculate_correlation_cached(x_values: List[float], y_values: List[float], 
                               cache: Dict[str, float] = None) -> float:
    """Calculate correlation with optional caching to avoid repeated computation."""
    if cache is None:
        cache = {}
    
    # Create cache key from sorted values
    cache_key = f"{hash(tuple(sorted(x_values)))}_{hash(tuple(sorted(y_values)))}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    # Calculate correlation
    correlation = confidence_accuracy_correlation(x_values, y_values)
    cache[cache_key] = correlation
    return correlation


# =====================================================================
# LLM-BASED GRADING FUNCTIONS
# =====================================================================

import openai
import json
import time
from typing import Optional, Dict, Any, List

class LLMGrader:
    """LLM-based grading system for evaluating answer accuracy."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize LLM grader.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for grading (default: gpt-4o)
            temperature: Temperature for grading (default: 0.0 for consistency)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.grading_cache = {}  # Cache for grading results
        
    def create_grading_prompt(self, question: str, prediction: str, gold_answer: str, 
                            context: Optional[str] = None) -> str:
        """
        Create a grading prompt for the LLM.
        
        Args:
            question: The original question
            prediction: The model's prediction
            gold_answer: The correct answer
            context: Optional context information
            
        Returns:
            Formatted grading prompt
        """
        prompt = f"""You are an expert evaluator tasked with grading the accuracy of AI model responses to questions. Your job is to provide a numerical accuracy score between 0.0 and 1.0, where:
- 1.0 = Perfectly correct answer
- 0.8-0.9 = Mostly correct with minor issues
- 0.6-0.7 = Partially correct with some errors
- 0.4-0.5 = Somewhat correct but significant issues
- 0.2-0.3 = Mostly incorrect but some relevant content
- 0.0-0.1 = Completely incorrect or irrelevant

Question: {question}

Correct Answer: {gold_answer}

Model's Answer: {prediction}"""

        if context:
            prompt += f"\n\nAdditional Context: {context}"
            
        prompt += """

Please evaluate the model's answer and provide:
1. An accuracy score between 0.0 and 1.0
2. A brief explanation of your reasoning

IMPORTANT: Respond ONLY with valid JSON in the following format (no markdown, no code blocks, no additional text):
{
    "accuracy_score": 0.85,
    "explanation": "The answer is mostly correct but includes some additional irrelevant information."
}"""
        
        return prompt
    
    def grade_answer(self, question: str, prediction: str, gold_answer: str, 
                    context: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Grade an answer using LLM.
        
        Args:
            question: The original question
            prediction: The model's prediction
            gold_answer: The correct answer
            context: Optional context information
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with accuracy score and explanation
        """
        # Create cache key
        cache_key = f"{hash(question)}_{hash(prediction)}_{hash(gold_answer)}"
        
        if use_cache and cache_key in self.grading_cache:
            return self.grading_cache[cache_key]
        
        try:
            # Create grading prompt
            prompt = self.create_grading_prompt(question, prediction, gold_answer, context)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response text - remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```
            
            response_text = response_text.strip()
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                accuracy_score = float(result.get('accuracy_score', 0.0))
                explanation = result.get('explanation', 'No explanation provided')
                
                # Validate accuracy score
                accuracy_score = max(0.0, min(1.0, accuracy_score))
                
                grading_result = {
                    'accuracy_score': accuracy_score,
                    'explanation': explanation,
                    'grading_method': 'llm',
                    'model_used': self.model,
                    'cached': False
                }
                
            except json.JSONDecodeError as e:
                # Try to extract accuracy score from the response text even if JSON parsing fails
                accuracy_score = 0.5  # Default fallback
                explanation = f'Failed to parse LLM response: {response_text[:200]}...'
                
                # Try to extract accuracy score using regex
                import re
                score_match = re.search(r'"accuracy_score":\s*([0-9.]+)', response_text)
                if score_match:
                    try:
                        accuracy_score = float(score_match.group(1))
                        accuracy_score = max(0.0, min(1.0, accuracy_score))
                        explanation = f'Extracted score from malformed JSON: {response_text[:200]}...'
                    except ValueError:
                        pass
                
                grading_result = {
                    'accuracy_score': accuracy_score,
                    'explanation': explanation,
                    'grading_method': 'llm_fallback',
                    'model_used': self.model,
                    'cached': False,
                    'json_error': str(e)
                }
            
            # Cache the result
            if use_cache:
                self.grading_cache[cache_key] = grading_result
                grading_result['cached'] = True
            
            return grading_result
            
        except Exception as e:
            # Fallback to soft accuracy if LLM call fails
            soft_score = calculate_soft_accuracy(prediction, [gold_answer])
            
            return {
                'accuracy_score': soft_score,
                'explanation': f'LLM grading failed, used soft accuracy: {str(e)}',
                'grading_method': 'soft_accuracy_fallback',
                'model_used': 'soft_accuracy',
                'cached': False,
                'error': str(e)
            }
    
    def grade_batch(self, questions: List[str], predictions: List[str], 
                   gold_answers: List[str], contexts: Optional[List[str]] = None,
                   delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Grade multiple answers in batch.
        
        Args:
            questions: List of questions
            predictions: List of predictions
            gold_answers: List of gold answers
            contexts: Optional list of contexts
            delay: Delay between API calls (seconds)
            
        Returns:
            List of grading results
        """
        if len(questions) != len(predictions) or len(predictions) != len(gold_answers):
            raise ValueError("All input lists must have the same length")
        
        if contexts and len(contexts) != len(questions):
            raise ValueError("Contexts list must have same length as other lists")
        
        results = []
        for i, (question, prediction, gold_answer) in enumerate(zip(questions, predictions, gold_answers)):
            context = contexts[i] if contexts else None
            
            result = self.grade_answer(question, prediction, gold_answer, context)
            results.append(result)
            
            # Rate limiting
            if i < len(questions) - 1:  # Don't delay after last call
                time.sleep(delay)
        
        return results
    
    def clear_cache(self):
        """Clear the grading cache."""
        self.grading_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.grading_cache),
            'cache_keys': list(self.grading_cache.keys())[:10]  # First 10 keys
        }


def create_llm_grader(api_key: str, model: str = "gpt-4o", temperature: float = 0.0) -> LLMGrader:
    """
    Create an LLM grader instance.
    
    Args:
        api_key: OpenAI API key
        model: Model to use for grading
        temperature: Temperature for grading
        
    Returns:
        LLMGrader instance
    """
    return LLMGrader(api_key, model, temperature)


def calculate_llm_accuracy(question: str, prediction: str, gold_answer: str, 
                          grader: LLMGrader, context: Optional[str] = None) -> float:
    """
    Calculate accuracy using LLM grading.
    
    Args:
        question: The original question
        prediction: The model's prediction
        gold_answer: The correct answer
        grader: LLMGrader instance
        context: Optional context information
        
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    result = grader.grade_answer(question, prediction, gold_answer, context)
    return result['accuracy_score']


# =====================================================================
# SOFT ACCURACY FUNCTIONS
# =====================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, strip punctuation, trim whitespace.
    
    Args:
        text: Input text to normalize
    
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Trim whitespace and normalize spaces
    text = ' '.join(text.split())
    
    return text


def calculate_token_overlap_f1(prediction_tokens: set, gold_tokens: set) -> float:
    """
    Calculate token overlap F1 score between prediction and gold answer tokens.
    
    Args:
        prediction_tokens: Set of tokens in prediction
        gold_tokens: Set of tokens in gold answer
    
    Returns:
        F1 score between 0 and 1
    """
    if not prediction_tokens and not gold_tokens:
        return 1.0  # Both empty = perfect match
    
    if not prediction_tokens or not gold_tokens:
        return 0.0  # One empty, one not = no overlap
    
    # Calculate intersection
    intersection = prediction_tokens.intersection(gold_tokens)
    
    # Calculate precision and recall
    precision = len(intersection) / len(prediction_tokens)
    recall = len(intersection) / len(gold_tokens)
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_edit_distance_similarity(text1: str, text2: str) -> float:
    """
    Calculate edit distance similarity: 1 - (Levenshtein distance / max length).
    
    Args:
        text1: First text string
        text2: Second text string
    
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 and not text2:
        return 1.0  # Both empty = perfect match
    
    if not text1 or not text2:
        return 0.0  # One empty, one not = no similarity
    
    # Calculate Levenshtein distance
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(text1, text2)
    max_length = max(len(text1), len(text2))
    
    # Calculate similarity: 1 - (distance / max_length)
    similarity = 1.0 - (distance / max_length)
    return max(0.0, similarity)  # Ensure non-negative


def calculate_soft_accuracy(prediction: str, gold_answers: List[str], 
                          token_weight: float = 0.8, edit_weight: float = 0.2) -> float:
    """
    Calculate non-binary answer accuracy (soft accuracy) for open-ended QA.
    
    This function computes a score between 0 and 1 representing how correct 
    the prediction is compared to the gold answers, with partial credit for 
    partially correct answers.
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of acceptable gold answer strings
        token_weight: Weight for token overlap F1 score (default: 0.8)
        edit_weight: Weight for edit distance similarity (default: 0.2)
    
    Returns:
        Soft accuracy score between 0 and 1
    """
    if not prediction:
        return 0.0
    
    if not gold_answers:
        return 0.0
    
    # Normalize prediction
    normalized_prediction = normalize_text(prediction)
    prediction_tokens = set(normalized_prediction.split())
    
    # Calculate similarity against each gold answer
    gold_scores = []
    
    for gold_answer in gold_answers:
        if not gold_answer:
            continue
            
        # Normalize gold answer
        normalized_gold = normalize_text(gold_answer)
        gold_tokens = set(normalized_gold.split())
        
        # Calculate token overlap F1
        token_f1 = calculate_token_overlap_f1(prediction_tokens, gold_tokens)
        
        # Calculate edit distance similarity
        edit_similarity = calculate_edit_distance_similarity(normalized_prediction, normalized_gold)
        
        # Combine with weighted average
        combined_score = (token_weight * token_f1) + (edit_weight * edit_similarity)
        gold_scores.append(combined_score)
    
    # Return the maximum score across all gold answers
    return max(gold_scores) if gold_scores else 0.0


def calculate_soft_accuracy_batch(predictions: List[str], gold_answer_sets: List[List[str]], 
                                 token_weight: float = 0.8, edit_weight: float = 0.2) -> List[float]:
    """
    Calculate soft accuracy for multiple predictions and gold answer sets (batch mode).
    
    Args:
        predictions: List of predicted answer strings
        gold_answer_sets: List of lists of acceptable gold answer strings
        token_weight: Weight for token overlap F1 score (default: 0.8)
        edit_weight: Weight for edit distance similarity (default: 0.2)
    
    Returns:
        List of soft accuracy scores between 0 and 1
    """
    if len(predictions) != len(gold_answer_sets):
        raise ValueError("Number of predictions must match number of gold answer sets")
    
    scores = []
    for prediction, gold_answers in zip(predictions, gold_answer_sets):
        score = calculate_soft_accuracy(prediction, gold_answers, token_weight, edit_weight)
        scores.append(score)
    
    return scores


def calculate_soft_accuracy_with_breakdown(prediction: str, gold_answers: List[str], 
                                         token_weight: float = 0.8, edit_weight: float = 0.2) -> Dict[str, Any]:
    """
    Calculate soft accuracy with detailed breakdown of components.
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of acceptable gold answer strings
        token_weight: Weight for token overlap F1 score (default: 0.8)
        edit_weight: Weight for edit distance similarity (default: 0.2)
    
    Returns:
        Dictionary with detailed breakdown including:
        - soft_accuracy: Final soft accuracy score
        - best_gold_match: Index of best matching gold answer
        - token_f1_scores: F1 scores for each gold answer
        - edit_similarities: Edit similarities for each gold answer
        - combined_scores: Combined scores for each gold answer
        - normalized_prediction: Normalized prediction text
        - normalized_gold_answers: List of normalized gold answer texts
    """
    if not prediction:
        return {
            'soft_accuracy': 0.0,
            'best_gold_match': -1,
            'token_f1_scores': [],
            'edit_similarities': [],
            'combined_scores': [],
            'normalized_prediction': '',
            'normalized_gold_answers': []
        }
    
    if not gold_answers:
        return {
            'soft_accuracy': 0.0,
            'best_gold_match': -1,
            'token_f1_scores': [],
            'edit_similarities': [],
            'combined_scores': [],
            'normalized_prediction': '',
            'normalized_gold_answers': []
        }
    
    # Normalize prediction
    normalized_prediction = normalize_text(prediction)
    prediction_tokens = set(normalized_prediction.split())
    
    # Calculate scores for each gold answer
    token_f1_scores = []
    edit_similarities = []
    combined_scores = []
    normalized_gold_answers = []
    
    for gold_answer in gold_answers:
        if not gold_answer:
            token_f1_scores.append(0.0)
            edit_similarities.append(0.0)
            combined_scores.append(0.0)
            normalized_gold_answers.append('')
            continue
            
        # Normalize gold answer
        normalized_gold = normalize_text(gold_answer)
        normalized_gold_answers.append(normalized_gold)
        gold_tokens = set(normalized_gold.split())
        
        # Calculate token overlap F1
        token_f1 = calculate_token_overlap_f1(prediction_tokens, gold_tokens)
        token_f1_scores.append(token_f1)
        
        # Calculate edit distance similarity
        edit_similarity = calculate_edit_distance_similarity(normalized_prediction, normalized_gold)
        edit_similarities.append(edit_similarity)
        
        # Combine with weighted average
        combined_score = (token_weight * token_f1) + (edit_weight * edit_similarity)
        combined_scores.append(combined_score)
    
    # Find best match
    best_score = max(combined_scores) if combined_scores else 0.0
    best_gold_match = combined_scores.index(best_score) if combined_scores else -1
    
    return {
        'soft_accuracy': best_score,
        'best_gold_match': best_gold_match,
        'token_f1_scores': token_f1_scores,
        'edit_similarities': edit_similarities,
        'combined_scores': combined_scores,
        'normalized_prediction': normalized_prediction,
        'normalized_gold_answers': normalized_gold_answers
    }


def calculate_continuous_uncertainty(entry: Dict[str, Any], retrieved_docs: List[Dict], 
                                   question: str) -> float:
    """
    Calculate a continuous uncertainty score (0-1) based on multiple factors.
    Higher score = more uncertain the model should be.
    
    Args:
        entry: Dataset entry with gold answer and source information
        retrieved_docs: List of retrieved documents
        question: The question being asked
    
    Returns:
        Uncertainty score between 0 and 1
    """
    uncertainty_factors = []
    
    # Factor 1: Retrieval Quality (0-1)
    # How well did the model retrieve relevant sources?
    relevant_docs = entry.get('source_sets', {}).get('clear', [])
    retrieved_urls = {doc.get('url', '') for doc in retrieved_docs}
    relevant_urls = {doc.get('url', '') for doc in relevant_docs}
    
    if relevant_urls:
        retrieval_recall = len(retrieved_urls.intersection(relevant_urls)) / len(relevant_urls)
        retrieval_uncertainty = 1.0 - retrieval_recall  # Higher recall = lower uncertainty
    else:
        retrieval_uncertainty = 0.5  # No relevant docs to compare against
    
    uncertainty_factors.append(retrieval_uncertainty)
    
    # Factor 2: Source Quality (0-1)
    # How reliable are the retrieved sources?
    quality_scores = []
    for doc in retrieved_docs:
        domain = doc.get('domain', '').lower()
        category = doc.get('category', '').lower()
        
        # Academic/educational domains are high quality
        if any(edu in domain for edu in ['.edu', 'academic', 'university', 'college']):
            quality_scores.append(0.1)  # Low uncertainty
        # Government domains are reliable
        elif any(gov in domain for gov in ['.gov', 'government', 'official']):
            quality_scores.append(0.2)  # Low uncertainty
        # News sites vary in quality
        elif any(news in domain for news in ['.com', 'news', 'media']):
            quality_scores.append(0.5)  # Medium uncertainty
        # Social media/blogs are less reliable
        elif any(social in domain for social in ['blog', 'social', 'forum']):
            quality_scores.append(0.8)  # High uncertainty
        else:
            quality_scores.append(0.6)  # Default medium uncertainty
    
    source_quality_uncertainty = np.mean(quality_scores) if quality_scores else 0.5
    uncertainty_factors.append(source_quality_uncertainty)
    
    # Factor 3: Question Complexity (0-1)
    # How complex/difficult is the question?
    question_lower = question.lower()
    
    # Complex question indicators
    complexity_indicators = [
        'compare', 'contrast', 'analyze', 'evaluate', 'explain why',
        'what are the implications', 'how does', 'what factors',
        'multiple', 'several', 'various', 'different', 'relationship'
    ]
    
    # Simple question indicators
    simplicity_indicators = [
        'what is', 'who is', 'when', 'where', 'how many',
        'define', 'name', 'list', 'single', 'one'
    ]
    
    complexity_score = 0.5  # Default medium complexity
    
    # Count complexity indicators
    complex_count = sum(1 for indicator in complexity_indicators if indicator in question_lower)
    simple_count = sum(1 for indicator in simplicity_indicators if indicator in question_lower)
    
    if complex_count > simple_count:
        complexity_score = 0.3 + (complex_count * 0.1)  # More complex = lower uncertainty (model should be more confident)
    elif simple_count > complex_count:
        complexity_score = 0.7 + (simple_count * 0.05)  # Simpler = higher uncertainty (model might be overconfident)
    
    complexity_score = min(1.0, max(0.0, complexity_score))
    #uncertainty_factors.append(complexity_score)
    # Not using complexity score for now
    
    
    # Factor 4: Source Coverage (0-1)
    # How many sources were retrieved vs. how many are available?
    total_available_sources = len(entry.get('source_sets', {}).get('clear', [])) + \
                             len(entry.get('source_sets', {}).get('ambiguous', []))
    
    if total_available_sources > 0:
        coverage_ratio = len(retrieved_docs) / total_available_sources
        # Too few sources = high uncertainty, too many sources = medium uncertainty
        if coverage_ratio < 0.3:
            coverage_uncertainty = 0.8  # Very few sources
        elif coverage_ratio < 0.6:
            coverage_uncertainty = 0.6  # Moderate coverage
        else:
            coverage_uncertainty = 0.4  # Good coverage
    else:
        coverage_uncertainty = 0.5
    
    uncertainty_factors.append(coverage_uncertainty)
    
    # Factor 6: Domain Expertise Required (0-1)
    # Some domains require more expertise
    domain_keywords = {
        'medical': ['health', 'medical', 'disease', 'treatment', 'symptoms', 'diagnosis'],
        'legal': ['law', 'legal', 'court', 'case', 'rights', 'regulation'],
        'technical': ['technology', 'software', 'algorithm', 'code', 'system'],
        'scientific': ['research', 'study', 'experiment', 'data', 'analysis'],
        'financial': ['finance', 'economic', 'market', 'investment', 'stock']
    }
    
    question_domain_uncertainty = 0.5  # Default
    for domain, keywords in domain_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            question_domain_uncertainty = 0.7  # Higher uncertainty for specialized domains
            break
    
    #uncertainty_factors.append(question_domain_uncertainty)
    # Not using question domain uncertainty for now
    
    # Factor 7: Temporal Relevance (0-1)
    # How recent/current is the information needed?
    temporal_indicators = ['recent', 'latest', 'current', 'now', 'today', '2024', '2023']
    if any(indicator in question_lower for indicator in temporal_indicators):
        temporal_uncertainty = 0.8  # High uncertainty for current events
    else:
        temporal_uncertainty = 0.4  # Lower uncertainty for historical/fact-based questions
    
    uncertainty_factors.append(temporal_uncertainty)
    
    # Calculate weighted average of all factors
    # Give more weight to retrieval quality and source quality
    weights = [0.40, 0.40, 0.10, 0.10]  # Sum to 1.0 (removed factors 3, 5, and 6)
    
    # Ensure we have the right number of weights
    if len(weights) != len(uncertainty_factors):
        weights = weights[:len(uncertainty_factors)]
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    final_uncertainty = sum(factor * weight for factor, weight in zip(uncertainty_factors, weights))
    
    # Ensure the result is between 0 and 1
    return max(0.0, min(1.0, final_uncertainty))


def test_soft_accuracy_functions():
    """
    Comprehensive test of soft accuracy functions with realistic examples.
    """
    print("=== SOFT ACCURACY FUNCTIONS TESTING ===\n")
    
    # Test Case 1: Example from user specification
    print("1. Testing Example from User Specification:")
    prediction = "Alexander Fleming in 1928"
    gold_answers = ["Alexander Fleming", "Fleming"]
    
    breakdown = calculate_soft_accuracy_with_breakdown(prediction, gold_answers)
    soft_score = calculate_soft_accuracy(prediction, gold_answers)
    
    print(f"   Prediction: '{prediction}'")
    print(f"   Gold answers: {gold_answers}")
    print(f"   Normalized prediction: '{breakdown['normalized_prediction']}'")
    print(f"   Normalized gold answers: {breakdown['normalized_gold_answers']}")
    print(f"   Token F1 scores: {[f'{score:.3f}' for score in breakdown['token_f1_scores']]}")
    print(f"   Edit similarities: {[f'{score:.3f}' for score in breakdown['edit_similarities']]}")
    print(f"   Combined scores: {[f'{score:.3f}' for score in breakdown['combined_scores']]}")
    print(f"   Best gold match: {breakdown['best_gold_match']}")
    print(f"   Final soft accuracy: {soft_score:.3f}")
    print()
    
    # Test Case 2: Perfect match
    print("2. Testing Perfect Match:")
    prediction2 = "Alexander Fleming"
    gold_answers2 = ["Alexander Fleming", "Fleming"]
    
    score2 = calculate_soft_accuracy(prediction2, gold_answers2)
    print(f"   Prediction: '{prediction2}'")
    print(f"   Gold answers: {gold_answers2}")
    print(f"   Soft accuracy: {score2:.3f}")
    print()
    
    # Test Case 3: Partial match
    print("3. Testing Partial Match:")
    prediction3 = "Fleming discovered penicillin"
    gold_answers3 = ["Alexander Fleming", "Fleming"]
    
    score3 = calculate_soft_accuracy(prediction3, gold_answers3)
    print(f"   Prediction: '{prediction3}'")
    print(f"   Gold answers: {gold_answers3}")
    print(f"   Soft accuracy: {score3:.3f}")
    print()
    
    # Test Case 4: No match
    print("4. Testing No Match:")
    prediction4 = "Louis Pasteur"
    gold_answers4 = ["Alexander Fleming", "Fleming"]
    
    score4 = calculate_soft_accuracy(prediction4, gold_answers4)
    print(f"   Prediction: '{prediction4}'")
    print(f"   Gold answers: {gold_answers4}")
    print(f"   Soft accuracy: {score4:.3f}")
    print()
    
    # Test Case 5: Empty inputs
    print("5. Testing Edge Cases:")
    
    # Empty prediction
    score5a = calculate_soft_accuracy("", gold_answers)
    print(f"   Empty prediction: {score5a:.3f}")
    
    # Empty gold answers
    score5b = calculate_soft_accuracy(prediction, [])
    print(f"   Empty gold answers: {score5b:.3f}")
    
    # Both empty
    score5c = calculate_soft_accuracy("", [])
    print(f"   Both empty: {score5c:.3f}")
    print()
    
    # Test Case 6: Batch processing
    print("6. Testing Batch Processing:")
    predictions_batch = [
        "Alexander Fleming in 1928",
        "Alexander Fleming",
        "Fleming discovered penicillin",
        "Louis Pasteur"
    ]
    gold_answer_sets_batch = [
        ["Alexander Fleming", "Fleming"],
        ["Alexander Fleming", "Fleming"],
        ["Alexander Fleming", "Fleming"],
        ["Alexander Fleming", "Fleming"]
    ]
    
    batch_scores = calculate_soft_accuracy_batch(predictions_batch, gold_answer_sets_batch)
    print(f"   Batch predictions: {len(predictions_batch)}")
    print(f"   Batch scores: {[f'{score:.3f}' for score in batch_scores]}")
    print()
    
    # Test Case 7: Different weights
    print("7. Testing Different Weight Configurations:")
    weights_configs = [
        (0.8, 0.2),  # Default
        (0.6, 0.4),  # More edit distance weight
        (1.0, 0.0),  # Only token overlap
        (0.0, 1.0)   # Only edit distance
    ]
    
    for token_weight, edit_weight in weights_configs:
        score = calculate_soft_accuracy(prediction, gold_answers, token_weight, edit_weight)
        print(f"   Weights (token={token_weight}, edit={edit_weight}): {score:.3f}")
    print()
    
    # Test Case 8: Complex medical example
    print("8. Testing Complex Medical Example:")
    medical_prediction = "The patient was diagnosed with acute myocardial infarction and received thrombolytic therapy"
    medical_gold_answers = [
        "acute myocardial infarction",
        "heart attack",
        "AMI",
        "myocardial infarction"
    ]
    
    medical_breakdown = calculate_soft_accuracy_with_breakdown(medical_prediction, medical_gold_answers)
    medical_score = calculate_soft_accuracy(medical_prediction, medical_gold_answers)
    
    print(f"   Prediction: '{medical_prediction}'")
    print(f"   Gold answers: {medical_gold_answers}")
    print(f"   Best match index: {medical_breakdown['best_gold_match']}")
    print(f"   Soft accuracy: {medical_score:.3f}")
    print()
    
    print("=== SOFT ACCURACY TESTING COMPLETE ===")
    return {
        'example_score': soft_score,
        'perfect_score': score2,
        'partial_score': score3,
        'no_match_score': score4,
        'batch_scores': batch_scores,
        'medical_score': medical_score
    }

def test_faithfulness_metrics():
    """Test the faithfulness metrics with sample data."""
    print("Testing faithfulness metrics...")
    
    # Sample data
    sample_results = [
        {
            'prediction_text': 'The capital of France is Paris, which is located in northern France.',
            'retrieved_docs': [
                {'text': 'Paris is the capital and largest city of France. It is located in northern France.'},
                {'text': 'France is a country in Western Europe with Paris as its capital.'}
            ],
            'gold_answer': 'Paris',
            'confidence': 0.9,
            'accuracy': 1.0
        },
        {
            'prediction_text': 'The capital of France is London, which is a major financial center.',
            'retrieved_docs': [
                {'text': 'Paris is the capital and largest city of France.'},
                {'text': 'London is the capital of the United Kingdom.'}
            ],
            'gold_answer': 'Paris',
            'confidence': 0.8,
            'accuracy': 0.0
        }
    ]
    
    # Test faithfulness metrics
    faithfulness_metrics = calm_rag_faithfulness_metrics(sample_results)
    
    print("Faithfulness metrics:")
    for key, value in faithfulness_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print("Faithfulness metrics test completed!")

if __name__ == "__main__":
    # Print comprehensive summary
    print_metrics_summary()
    
    print("\n" + "="*60 + "\n")
    
    # Run comprehensive testing
    core_results, utility_results = test_all_metrics()
    
    print(f"\nTotal Core Metrics: {len(core_results)}")
    print(f"Total Utility Metrics: {len(utility_results)}")
    
    # Example usage of specific metrics
    print(f"\nExample - H1 Retrieval Recall: {core_results.get('h1_avg_retrieval_recall', 0.0):.3f}")
    print(f"Example - H3 Hedge Precision: {core_results.get('h3_hedge_precision', 0.0):.3f}")
    print(f"Example - Calibration Improvement: {utility_results.get('calibration_calibration_improvement_potential', 0.0):.3f}")
    
    print("\n" + "="*60 + "\n")
    
    # Test soft accuracy functions
    soft_accuracy_results = test_soft_accuracy_functions()
    
    # Test faithfulness metrics
    test_faithfulness_metrics()
    
    # Test ASI metrics
    test_asi_metrics()
    
    print("\n" + "="*60)
    print("CALM-RAG METRICS IMPLEMENTATION COMPLETE!")
    print("All hypotheses H1-H6 are now fully implemented with comprehensive metrics.")
    print("Soft accuracy functions are now available for non-binary answer evaluation.")
    print("Faithfulness metrics are now available for grounding evaluation.")
    print("Ambiguity Sensitivity Index (ASI) is now available for ambiguity-aware evaluation.")
    print("="*60)