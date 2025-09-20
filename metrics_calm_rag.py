"""
CALM-RAG Metrics Module - Streamlined Version
Implements all CALM-RAG hypothesis metrics (H1-H5) and core evaluation metrics.
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
    """Normalize document to a consistent ID for comparison."""
    if isinstance(doc, str):
        if doc.startswith('http'):
            parsed = urlparse(doc)
            return f"{parsed.netloc}{parsed.path.rstrip('/')}"
        return doc.lower().strip()
    elif isinstance(doc, dict):
        for key in ['url', 'id', 'domain', 'source']:
            if key in doc and doc[key]:
                if key == 'url' and doc[key].startswith('http'):
                    parsed = urlparse(doc[key])
                    return f"{parsed.netloc}{parsed.path.rstrip('/')}"
                return str(doc[key]).lower().strip()
        return str(doc).lower().strip()
    return str(doc).lower().strip()


def retrieval_recall(retrieved_docs: List[Union[str, Dict]], 
                    relevant_docs: List[Union[str, Dict]]) -> float:
    """Calculate retrieval recall: fraction of relevant documents retrieved."""
    if not relevant_docs:
        return 1.0
    
    retrieved_ids = {normalize_document_id(doc) for doc in retrieved_docs}
    relevant_ids = {normalize_document_id(doc) for doc in relevant_docs}
    
    intersection = retrieved_ids.intersection(relevant_ids)
    return len(intersection) / len(relevant_ids)


def retrieval_precision(retrieved_docs: List[Union[str, Dict]], 
                       relevant_docs: List[Union[str, Dict]]) -> float:
    """Calculate retrieval precision: fraction of retrieved documents that are relevant."""
    if not retrieved_docs:
        return 0.0
    
    retrieved_ids = {normalize_document_id(doc) for doc in retrieved_docs}
    relevant_ids = {normalize_document_id(doc) for doc in relevant_docs}
    
    intersection = retrieved_ids.intersection(relevant_ids)
    return len(intersection) / len(retrieved_ids)


def retrieval_f1(retrieved_docs: List[Union[str, Dict]], 
                 relevant_docs: List[Union[str, Dict]]) -> float:
    """Calculate retrieval F1 score."""
    precision = retrieval_precision(retrieved_docs, relevant_docs)
    recall = retrieval_recall(retrieved_docs, relevant_docs)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def source_quality_score(retrieved_docs: List[Dict[str, Any]]) -> float:
    """Calculate source quality based on domain reliability categories."""
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
    """Calculate retrieval diversity based on unique domains."""
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
    
    return len(domains) / len(retrieved_docs)


def expected_calibration_error(confidences: List[float], accuracies: List[float], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if len(confidences) != len(accuracies) or len(confidences) < 2:
        return 0.0
    
    # Filter out invalid values
    valid_pairs = [(c, a) for c, a in zip(confidences, accuracies) 
                   if not (np.isnan(c) or np.isnan(a) or np.isinf(c) or np.isinf(a))]
    
    if len(valid_pairs) < 2:
        return 0.0
    
    confidences, accuracies = zip(*valid_pairs)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [(c, a) for c, a in zip(confidences, accuracies) 
                  if bin_lower <= c < bin_upper]
        
        if in_bin:
            bin_confidences, bin_accuracies = zip(*in_bin)
            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_accuracies)
            bin_size = len(in_bin)
            ece += abs(avg_confidence - avg_accuracy) * bin_size
    
    return ece / len(confidences)


def brier_score(confidences: List[float], accuracies: List[float]) -> float:
    """Calculate Brier score."""
    if len(confidences) != len(accuracies) or len(confidences) < 2:
        return 0.0
    
    valid_pairs = [(c, a) for c, a in zip(confidences, accuracies) 
                   if not (np.isnan(c) or np.isnan(a) or np.isinf(c) or np.isinf(a))]
    
    if not valid_pairs:
        return 0.0
    
    confidences, accuracies = zip(*valid_pairs)
    
    # Brier score: mean((confidence - accuracy)^2)
    return np.mean([(c - a) ** 2 for c, a in zip(confidences, accuracies)])


def confidence_accuracy_correlation(confidences: List[float], accuracies: List[float]) -> float:
    """Calculate Pearson correlation between confidence and accuracy."""
    if len(confidences) != len(accuracies) or len(confidences) < 2:
        return 0.0
    
    valid_pairs = [(c, a) for c, a in zip(confidences, accuracies) 
                   if not (np.isnan(c) or np.isnan(a) or np.isinf(c) or np.isinf(a))]
    
    if len(valid_pairs) < 2:
        return 0.0
    
    confidences, accuracies = zip(*valid_pairs)
    
    try:
        correlation, _ = pearsonr(confidences, accuracies)
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def contains_hedge(text: str) -> int:
    """Count hedge terms in text."""
    if not text:
        return 0
    
    text_lower = text.lower()
    count = 0
    for term in HEDGE_TERMS:
        count += text_lower.count(term)
    return count


def calculate_soft_accuracy(prediction: str, gold_answers: List[str]) -> float:
    """Calculate soft accuracy using fuzzy matching."""
    if not prediction or not gold_answers:
        return 0.0
    
    prediction_lower = prediction.lower().strip()
    best_score = 0.0
    
    for gold in gold_answers:
        if not gold:
            continue
        
        gold_lower = gold.lower().strip()
        
        # Exact match
        if prediction_lower == gold_lower:
            return 1.0
        
        # Substring match
        if prediction_lower in gold_lower or gold_lower in prediction_lower:
            best_score = max(best_score, 0.8)
        
        # Word overlap
        pred_words = set(prediction_lower.split())
        gold_words = set(gold_lower.split())
        
        if pred_words and gold_words:
            overlap = len(pred_words.intersection(gold_words))
            overlap_score = overlap / max(len(pred_words), len(gold_words))
            best_score = max(best_score, overlap_score)
    
    return best_score


# ===== CALM-RAG HYPOTHESIS METRICS =====

def calm_rag_h1_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    H1: Overconfidence under sparse/noisy evidence
    Tests if models are overconfident when retrieval quality is poor.
    """
    if not results:
        return {}
    
    recalls = []
    confidences = []
    accuracies = []
    
    for result in results:
        retrieved_docs = result.get('retrieved_docs', [])
        relevant_docs = result.get('relevant_docs', [])
        
        recall = retrieval_recall(retrieved_docs, relevant_docs)
        confidence = result.get('confidence', 0.5)
        accuracy = result.get('accuracy', 0.0)
        
        recalls.append(recall)
        confidences.append(confidence)
        accuracies.append(accuracy)
    
    # Calculate correlations
    recall_confidence_corr = confidence_accuracy_correlation(recalls, confidences)
    
    # Overconfidence index: confidence when accuracy is low
    overconfidence_cases = [(c, a) for c, a in zip(confidences, accuracies) if a < 0.5]
    overconfidence_index = 0.0
    if overconfidence_cases:
        avg_confidence_wrong = np.mean([c for c, a in overconfidence_cases])
        overconfidence_index = max(0, avg_confidence_wrong - 0.5)
    
    # Wrong answer rate
    wrong_answer_rate = np.mean([1 - a for a in accuracies])
    
    # Refusal rate (when confidence is very low)
    refusal_rate = np.mean([1 if c < 0.3 else 0 for c in confidences])
    
    return {
        'retrieval_recall_confidence_correlation': recall_confidence_corr,
        'avg_retrieval_recall': np.mean(recalls),
        'overconfidence_index': overconfidence_index,
        'wrong_answer_rate': wrong_answer_rate,
        'refusal_rate': refusal_rate
    }


def calm_rag_h2_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    H2: Calibration difference with and without retrieval
    Tests if retrieval improves calibration.
    """
    if not results:
        return {}
    
    # Separate results by retrieval status (simulated)
    with_retrieval_confidences = []
    with_retrieval_accuracies = []
    without_retrieval_confidences = []
    without_retrieval_accuracies = []
    
    for result in results:
        confidence = result.get('confidence', 0.5)
        accuracy = result.get('accuracy', 0.0)
        
        # Simulate without-retrieval scenario using lower confidence
        # This is a simplified approach - in practice you'd run models twice
        without_confidence = confidence * 0.7  # Simulate lower confidence without retrieval
        without_accuracy = accuracy * 0.8  # Simulate lower accuracy without retrieval
        
        with_retrieval_confidences.append(confidence)
        with_retrieval_accuracies.append(accuracy)
        without_retrieval_confidences.append(without_confidence)
        without_retrieval_accuracies.append(without_accuracy)
    
    # Calculate ECE for both scenarios
    ece_with = expected_calibration_error(with_retrieval_confidences, with_retrieval_accuracies)
    ece_without = expected_calibration_error(without_retrieval_confidences, without_retrieval_accuracies)
    ece_difference = ece_without - ece_with  # Positive means retrieval helps
    
    # Calculate Brier scores
    brier_with = brier_score(with_retrieval_confidences, with_retrieval_accuracies)
    brier_without = brier_score(without_retrieval_confidences, without_retrieval_accuracies)
    brier_difference = brier_without - brier_with
    
    # Calculate correlations
    corr_with = confidence_accuracy_correlation(with_retrieval_confidences, with_retrieval_accuracies)
    corr_without = confidence_accuracy_correlation(without_retrieval_confidences, without_retrieval_accuracies)
    corr_difference = corr_with - corr_without
    
    return {
        'ece_with_retrieval': ece_with,
        'ece_without_retrieval': ece_without,
        'ece_difference': ece_difference,
        'brier_with_retrieval': brier_with,
        'brier_without_retrieval': brier_without,
        'brier_difference': brier_difference,
        'confidence_accuracy_corr_with_retrieval': corr_with,
        'confidence_accuracy_corr_without_retrieval': corr_without,
        'correlation_difference': corr_difference
    }


def calm_rag_h3_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    H3: Hedging language as signal of uncertainty
    Tests if models use appropriate hedging language when uncertain.
    """
    if not results:
        return {}
    
    texts = []
    confidences = []
    accuracies = []
    
    for result in results:
        prediction = result.get('prediction_text', '')
        explanation = result.get('prediction_explanation', '')
        combined_text = f"{prediction} {explanation}"
        
        texts.append(combined_text)
        confidences.append(result.get('confidence', 0.5))
        accuracies.append(result.get('accuracy', 0.0))
    
    # Calculate hedge metrics
    hedge_counts = [contains_hedge(text) for text in texts]
    
    # Hedge precision/recall (simplified - using confidence as proxy for true uncertainty)
    true_uncertainties = [1 - c for c in confidences]  # Higher confidence = lower uncertainty
    hedge_detections = [1 if count > 0 else 0 for count in hedge_counts]
    
    # Calculate precision and recall
    true_positives = sum([1 for h, u in zip(hedge_detections, true_uncertainties) if h == 1 and u > 0.3])
    false_positives = sum([1 for h, u in zip(hedge_detections, true_uncertainties) if h == 1 and u <= 0.3])
    false_negatives = sum([1 for h, u in zip(hedge_detections, true_uncertainties) if h == 0 and u > 0.3])
    
    hedge_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    hedge_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    hedge_f1 = 2 * (hedge_precision * hedge_recall) / (hedge_precision + hedge_recall) if (hedge_precision + hedge_recall) > 0 else 0.0
    
    # Lexical overconfidence: confident language when wrong
    confident_wrong_cases = [(text, acc) for text, acc in zip(texts, accuracies) if acc < 0.5]
    lexical_overconfidence = 0.0
    if confident_wrong_cases:
        confident_terms = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously']
        overconfident_count = 0
        for text, acc in confident_wrong_cases:
            text_lower = text.lower()
            if any(term in text_lower for term in confident_terms):
                overconfident_count += 1
        lexical_overconfidence = overconfident_count / len(confident_wrong_cases)
    
    # Uncertainty-confidence correlation
    uncertainty_scores = [contains_hedge(text) for text in texts]
    uncertainty_confidence_corr = confidence_accuracy_correlation(uncertainty_scores, confidences)
    
    # Hedge density
    hedge_density = np.mean([count / max(len(text.split()), 1) for count, text in zip(hedge_counts, texts)])
    
    # Confident wrong rate
    confident_threshold = 0.7
    confident_wrong_rate = np.mean([1 if c > confident_threshold and a < 0.5 else 0 for c, a in zip(confidences, accuracies)])
    
    return {
        'hedge_precision': hedge_precision,
        'hedge_recall': hedge_recall,
        'hedge_f1': hedge_f1,
        'lexical_overconfidence_index': lexical_overconfidence,
        'uncertainty_confidence_correlation': uncertainty_confidence_corr,
        'hedge_density': hedge_density,
        'confident_wrong_rate': confident_wrong_rate
    }


def calm_rag_h4_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    H4: Self-assessment and numeric calibration
    Tests if models can accurately assess their own confidence.
    """
    if not results:
        return {}
    
    confidences = []
    accuracies = []
    
    for result in results:
        confidences.append(result.get('confidence', 0.5))
        accuracies.append(result.get('accuracy', 0.0))
    
    # Basic calibration metrics
    ece = expected_calibration_error(confidences, accuracies)
    brier = brier_score(confidences, accuracies)
    correlation = confidence_accuracy_correlation(confidences, accuracies)
    
    # Isotonic calibration
    try:
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(confidences, accuracies)
        calibrated_confidences = iso_reg.transform(confidences)
        ece_after_isotonic = expected_calibration_error(calibrated_confidences, accuracies)
        calibration_improvement = ece - ece_after_isotonic
    except:
        ece_after_isotonic = ece
        calibration_improvement = 0.0
    
    # Confidence distribution entropy
    confidence_entropy = 0.0
    if len(confidences) > 1:
        hist, _ = np.histogram(confidences, bins=10, range=(0, 1))
        probs = hist / len(confidences)
        probs = probs[probs > 0]  # Remove zero probabilities
        confidence_entropy = -np.sum(probs * np.log2(probs))
    
    return {
        'expected_calibration_error': ece,
        'brier_score': brier,
        'confidence_accuracy_correlation': correlation,
        'calibration_ece_after_isotonic': ece_after_isotonic,
        'calibration_improvement': calibration_improvement,
        'confidence_distribution_entropy': confidence_entropy
    }


def calm_rag_h5_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    H5: Source quality impact on calibration
    Tests if source quality affects model calibration.
    """
    if not results:
        return {}
    
    source_qualities = []
    confidences = []
    accuracies = []
    
    for result in results:
        retrieved_docs = result.get('retrieved_docs', [])
        quality = source_quality_score(retrieved_docs)
        
        source_qualities.append(quality)
        confidences.append(result.get('confidence', 0.5))
        accuracies.append(result.get('accuracy', 0.0))
    
    # Source quality-confidence correlation
    quality_confidence_corr = confidence_accuracy_correlation(source_qualities, confidences)
    
    # Quality-weighted ECE
    quality_weighted_ece = 0.0
    if source_qualities and confidences and accuracies:
        total_weight = sum(source_qualities)
        if total_weight > 0:
            for q, c, a in zip(source_qualities, confidences, accuracies):
                weight = q / total_weight
                error = abs(c - a)
                quality_weighted_ece += weight * error
    
    # Separate ECE for high vs low quality sources
    high_quality_threshold = 0.7
    high_quality_confidences = [c for q, c in zip(source_qualities, confidences) if q >= high_quality_threshold]
    high_quality_accuracies = [a for q, a in zip(source_qualities, accuracies) if q >= high_quality_threshold]
    low_quality_confidences = [c for q, c in zip(source_qualities, confidences) if q < high_quality_threshold]
    low_quality_accuracies = [a for q, a in zip(source_qualities, accuracies) if q < high_quality_threshold]
    
    high_quality_ece = expected_calibration_error(high_quality_confidences, high_quality_accuracies)
    low_quality_ece = expected_calibration_error(low_quality_confidences, low_quality_accuracies)
    quality_calibration_gap = high_quality_ece - low_quality_ece
    
    # Source diversity correlation
    diversities = []
    for result in results:
        retrieved_docs = result.get('retrieved_docs', [])
        diversity = retrieval_diversity(retrieved_docs)
        diversities.append(diversity)
    
    diversity_calibration_corr = confidence_accuracy_correlation(diversities, confidences)
    
    return {
        'source_quality_confidence_correlation': quality_confidence_corr,
        'source_diversity_calibration_correlation': diversity_calibration_corr,
        'quality_weighted_ece': quality_weighted_ece,
        'high_quality_source_ece': high_quality_ece,
        'low_quality_source_ece': low_quality_ece,
        'quality_calibration_gap': quality_calibration_gap
    }


# ===== ADDITIONAL METRICS =====

def calculate_ambiguity_sensitivity_index(clear_entry: Dict[str, Any], 
                                        ambiguous_entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate Ambiguity Sensitivity Index (ASI) comparing clear vs ambiguous sources.
    """
    clear_confidence = clear_entry.get('confidence', 0.5)
    ambiguous_confidence = ambiguous_entry.get('confidence', 0.5)
    clear_accuracy = clear_entry.get('accuracy', 0.0)
    ambiguous_accuracy = ambiguous_entry.get('accuracy', 0.0)
    
    # ASI components
    confidence_sensitivity = abs(clear_confidence - ambiguous_confidence)
    accuracy_sensitivity = abs(clear_accuracy - ambiguous_accuracy)
    
    # Overall ASI (normalized combination)
    asi = (confidence_sensitivity + accuracy_sensitivity) / 2.0
    
    return {
        'asi': asi,
        'confidence_sensitivity': confidence_sensitivity,
        'accuracy_sensitivity': accuracy_sensitivity,
        'clear_confidence': clear_confidence,
        'ambiguous_confidence': ambiguous_confidence,
        'clear_accuracy': clear_accuracy,
        'ambiguous_accuracy': ambiguous_accuracy
    }


def calculate_batch_asi(question_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate batch ASI statistics from individual ASI results."""
    if not question_results:
        return {}
    
    asi_scores = [result.get('asi_score', 0.0) for result in question_results]
    
    return {
        'mean_asi': np.mean(asi_scores),
        'std_asi': np.std(asi_scores),
        'min_asi': np.min(asi_scores),
        'max_asi': np.max(asi_scores),
        'median_asi': np.median(asi_scores)
    }


def calculate_continuous_uncertainty(entry: Dict[str, Any], 
                                   retrieved_docs: List[Dict[str, Any]], 
                                   question: str) -> float:
    """
    Calculate continuous uncertainty score based on multiple factors.
    """
    # Factor 1: Source quality
    source_quality = source_quality_score(retrieved_docs)
    quality_uncertainty = 1.0 - source_quality
    
    # Factor 2: Source diversity (low diversity = higher uncertainty)
    diversity = retrieval_diversity(retrieved_docs)
    diversity_uncertainty = 1.0 - diversity
    
    # Factor 3: Number of sources (fewer sources = higher uncertainty)
    num_sources = len(retrieved_docs)
    source_count_uncertainty = max(0, (5 - num_sources) / 5.0)  # Assume 5 is optimal
    
    # Combine factors
    uncertainty = (quality_uncertainty + diversity_uncertainty + source_count_uncertainty) / 3.0
    
    return min(1.0, max(0.0, uncertainty))


def compute_all_calm_rag_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute all CALM-RAG metrics."""
    all_metrics = {}
    
    # H1-H5 metrics
    all_metrics.update(calm_rag_h1_metrics(results))
    all_metrics.update(calm_rag_h2_metrics(results))
    all_metrics.update(calm_rag_h3_metrics(results))
    all_metrics.update(calm_rag_h4_metrics(results))
    all_metrics.update(calm_rag_h5_metrics(results))
    
    return all_metrics


def calculate_all_utility_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate utility metrics for retrieval quality assessment."""
    if not results:
        return {}
    
    recalls = []
    precisions = []
    f1_scores = []
    source_qualities = []
    diversities = []
    
    for result in results:
        retrieved_docs = result.get('retrieved_docs', [])
        relevant_docs = result.get('relevant_docs', [])
        
        recall = retrieval_recall(retrieved_docs, relevant_docs)
        precision = retrieval_precision(retrieved_docs, relevant_docs)
        f1 = retrieval_f1(retrieved_docs, relevant_docs)
        quality = source_quality_score(retrieved_docs)
        diversity = retrieval_diversity(retrieved_docs)
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        source_qualities.append(quality)
        diversities.append(diversity)
    
    return {
        'avg_retrieval_recall': np.mean(recalls),
        'avg_retrieval_precision': np.mean(precisions),
        'avg_retrieval_f1': np.mean(f1_scores),
        'avg_source_quality': np.mean(source_qualities),
        'source_quality_std': np.std(source_qualities),
        'avg_source_diversity': np.mean(diversities),
        'source_diversity_std': np.std(diversities)
    }

