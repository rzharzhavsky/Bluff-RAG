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
    confidences = [r.get('confidence', 0.0) for r in results]
    accuracies = [r.get('accuracy', 0.0) for r in results]
    predictions = [r.get('prediction_text', '') for r in results]
    uncertainties = [r.get('is_uncertain', False) for r in results]
    
    metrics = {}
    
    # H1: Overconfidence Under Sparse or Noisy Evidence
    h1_metrics = calm_rag_h1_metrics(results)
    for key, value in h1_metrics.items():
        metrics[f'h1_{key}'] = value
    
    # H2: Calibration Difference with and without Retrieval
    metrics['h2_expected_calibration_error'] = expected_calibration_error(confidences, accuracies)
    metrics['h2_brier_score'] = brier_score(confidences, accuracies)
    metrics['h2_confidence_accuracy_correlation'] = confidence_accuracy_correlation(confidences, accuracies)
    
    # H3: Hedging Language as a Signal of Uncertainty
    hedge_prec, hedge_rec = hedge_precision_recall(predictions, uncertainties)
    metrics['h3_hedge_precision'] = hedge_prec
    metrics['h3_hedge_recall'] = hedge_rec
    metrics['h3_lexical_overconfidence_index'] = lexical_overconfidence_index(predictions, accuracies)
    
    # H4: Self-Assessment and Numeric Confidence Calibration
    metrics['h4_confidence_accuracy_correlation'] = confidence_accuracy_correlation(confidences, accuracies)
    metrics['h4_brier_score'] = brier_score(confidences, accuracies)
    
    # Isotonic calibration (post-hoc calibration)
    _, ece_after_isotonic = isotonic_calibration(confidences, accuracies)
    metrics['calibration_ece_after_isotonic'] = ece_after_isotonic
    
    # Additional retrieval quality metrics
    retrieval_metrics = retrieval_quality_metrics(results)
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