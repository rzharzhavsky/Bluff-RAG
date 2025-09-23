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


def normalize_text(text: str) -> str:
    """Normalize text for comparison by converting to lowercase and removing punctuation."""
    if not text:
        return ""
    # Convert to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    return ' '.join(text.split())


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


def calculate_llm_accuracy(prediction: str, gold_answer: str, question: str = "", 
                          openai_client=None) -> float:
    """
    Calculate accuracy using LLM grading for semantic similarity.
    
    Args:
        prediction: Model's predicted answer
        gold_answer: Ground truth answer
        question: Original question (optional, for context)
        openai_client: OpenAI client instance
    
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if not prediction or not gold_answer:
        return 0.0
    
    if not openai_client:
        # Fallback to soft accuracy if no OpenAI client
        return calculate_soft_accuracy(prediction, [gold_answer])
    
    try:
        # Create grading prompt
        grading_prompt = f"""You are an expert evaluator assessing the accuracy of AI-generated answers. 

Question: {question}

Gold Answer: {gold_answer}

AI Answer: {prediction}

Please evaluate how accurate the AI answer is compared to the gold answer. Consider:
1. Semantic similarity and meaning
2. Factual correctness
3. Completeness of information
4. Relevance to the question

Rate the accuracy on a scale from 0.0 to 1.0 where:
- 1.0 = Perfect match, completely accurate
- 0.8-0.9 = Very accurate with minor differences
- 0.6-0.7 = Mostly accurate with some differences
- 0.4-0.5 = Partially accurate, some correct information
- 0.2-0.3 = Mostly inaccurate, few correct elements
- 0.0-0.1 = Completely inaccurate or irrelevant

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.75")."""

        # Call OpenAI API for grading
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Respond with only a number between 0.0 and 1.0."},
                {"role": "user", "content": grading_prompt}
            ],
            max_tokens=10,
            temperature=0.0 
        )
        
        # Extract score
        score_text = response.choices[0].message.content.strip()
        
        # Parse score
        try:
            score = float(score_text)
            # Clamp to valid range
            return max(0.0, min(1.0, score))
        except ValueError:
            # If parsing fails, fall back to soft accuracy
            print(f"LLM grading failed: {e}. Falling back to soft accuracy.")
            return calculate_soft_accuracy(prediction, [gold_answer])
            
    except Exception as e:
        print(f"LLM grading failed: {e}. Falling back to soft accuracy.")
        return calculate_soft_accuracy(prediction, [gold_answer])


def calculate_soft_accuracy(prediction: str, gold_answers: List[str]) -> float:
    """Calculate soft accuracy using fuzzy matching (fallback method)."""
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
    # Adjust thresholds for calibrated confidence scale
    # Calibrated confidences typically range from 0.04 to 1.0, so use adaptive thresholds
    confidence_median = np.median(confidences)
    confidence_75th = np.percentile(confidences, 75)
    
    # For calibrated confidence, use 75th percentile as "high confidence" threshold
    # This identifies cases where model is confident but wrong
    high_confidence_threshold = confidence_75th
    
    overconfidence_cases = [(c, a) for c, a in zip(confidences, accuracies) 
                          if a < 0.5 and c > high_confidence_threshold]
    overconfidence_index = 0.0
    if overconfidence_cases:
        avg_confidence_wrong = np.mean([c for c, a in overconfidence_cases])
        # Scale by confidence range to make it interpretable
        confidence_range = max(confidences) - min(confidences)
        overconfidence_index = (avg_confidence_wrong - high_confidence_threshold) / max(confidence_range, 0.01)
    
    # Wrong answer rate
    wrong_answer_rate = np.mean([1 - a for a in accuracies])
    
    # Refusal rate (when confidence is very low) - adjust for calibrated scale
    # Use 25th percentile instead of hardcoded 0.3 for calibrated confidence
    low_confidence_threshold = np.percentile(confidences, 25)  # Bottom 25% of confidences
    refusal_rate = np.mean([1 if c < low_confidence_threshold else 0 for c in confidences])
    
    # Calculate H1 composite score
    # Normalize metrics to 0-1 scale where higher is better
    normalized_avg_recall = np.mean(recalls)  # Already 0-1, higher is better
    normalized_overconfidence = max(0, 1 - overconfidence_index)  # Invert overconfidence (lower is better)
    normalized_wrong_rate = max(0, 1 - wrong_answer_rate)  # Invert wrong rate (lower is better)
    
    h1_composite_score = (normalized_avg_recall + normalized_overconfidence + normalized_wrong_rate) / 3
    
    return {
        'retrieval_recall_confidence_correlation': recall_confidence_corr,
        'avg_retrieval_recall': np.mean(recalls),
        'overconfidence_index': overconfidence_index,
        'wrong_answer_rate': wrong_answer_rate,
        'refusal_rate': refusal_rate,
        'h1_composite_score': h1_composite_score,
        # Diagnostic info for calibrated confidence thresholds
        'calibrated_confidence_stats': {
            'median': confidence_median,
            '25th_percentile': low_confidence_threshold,
            '75th_percentile': high_confidence_threshold,
            'high_confidence_threshold': high_confidence_threshold,
            'min': min(confidences) if confidences else 0,
            'max': max(confidences) if confidences else 0,
            'overconfidence_cases_count': len(overconfidence_cases),
            'total_wrong_answers': len([a for a in accuracies if a < 0.5])
        }
    }




def calculate_question_difficulty(result: Dict[str, Any]) -> float:
    """
    Calculate true uncertainty based on question difficulty factors.
    Returns a score from 0 (easy/certain) to 1 (difficult/uncertain).
    """
    difficulty_factors = []
    
    # Factor 1: Retrieval quality (low recall = high difficulty)
    retrieved_docs = result.get('retrieved_docs', [])
    relevant_docs = result.get('relevant_docs', [])
    retrieval_recall_score = retrieval_recall(retrieved_docs, relevant_docs)
    difficulty_factors.append(1 - retrieval_recall_score)  # Low recall = high difficulty
    
    # Factor 2: Source quality (low quality = high difficulty)
    source_quality = source_quality_score(retrieved_docs)
    difficulty_factors.append(1 - source_quality)  # Low quality = high difficulty
    
    # Factor 3: Source diversity (high diversity might indicate conflicting info)
    if len(retrieved_docs) > 1:
        # Simple diversity measure: unique domains
        domains = set()
        for doc in retrieved_docs:
            if isinstance(doc, dict) and 'url' in doc:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(doc['url']).netloc
                    domains.add(domain)
                except:
                    pass
        diversity = len(domains) / len(retrieved_docs) if retrieved_docs else 0
        # High diversity can indicate uncertainty (conflicting sources)
        difficulty_factors.append(min(diversity * 2, 1.0))  # Scale and cap at 1.0
    else:
        difficulty_factors.append(0.5)  # Neutral if single/no source
    
    # Factor 4: Question ambiguity (based on source set type)
    source_set_type = result.get('source_set_type', 'clear')
    if source_set_type == 'ambiguous':
        difficulty_factors.append(0.8)  # Ambiguous questions are inherently difficult
    else:
        difficulty_factors.append(0.2)  # Clear questions are easier
    
    # Return weighted average of difficulty factors
    return np.mean(difficulty_factors)


def calm_rag_h3_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    H3: Hedging language as signal of uncertainty
    Tests if models use appropriate hedging language when uncertain.
    Improved with true difficulty-based uncertainty and non-binary hedging.
    """
    if not results:
        return {}
    
    texts = []
    confidences = []
    accuracies = []
    true_uncertainties = []
    
    for result in results:
        prediction = result.get('prediction_text', '')
        explanation = result.get('prediction_explanation', '')
        combined_text = f"{prediction} {explanation}"
        
        texts.append(combined_text)
        confidences.append(result.get('confidence', 0.5))
        accuracies.append(result.get('accuracy', 0.0))
        
        # Calculate true uncertainty based on question difficulty
        difficulty = calculate_question_difficulty(result)
        true_uncertainties.append(difficulty)
    
    # Calculate hedge density (non-binary)
    hedge_counts = [contains_hedge(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    hedge_densities = [count / max(words, 1) for count, words in zip(hedge_counts, word_counts)]
    
    # Define thresholds for classification
    uncertainty_threshold = np.percentile(true_uncertainties, 75)  # Top 25% most uncertain
    certainty_threshold = np.percentile(true_uncertainties, 25)   # Bottom 25% most certain
    hedge_threshold = np.percentile(hedge_densities, 50)          # Median hedge density
    
    # Calculate confusion matrix elements with true negatives
    true_positives = sum([1 for h, u in zip(hedge_densities, true_uncertainties) 
                         if h > hedge_threshold and u > uncertainty_threshold])
    false_positives = sum([1 for h, u in zip(hedge_densities, true_uncertainties) 
                          if h > hedge_threshold and u < certainty_threshold])
    false_negatives = sum([1 for h, u in zip(hedge_densities, true_uncertainties) 
                          if h <= hedge_threshold and u > uncertainty_threshold])
    true_negatives = sum([1 for h, u in zip(hedge_densities, true_uncertainties) 
                         if h <= hedge_threshold and u < certainty_threshold])
    
    # Calculate comprehensive metrics
    hedge_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    hedge_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    hedge_specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    hedge_f1 = 2 * (hedge_precision * hedge_recall) / (hedge_precision + hedge_recall) if (hedge_precision + hedge_recall) > 0 else 0.0
    hedge_accuracy = (true_positives + true_negatives) / len(results) if len(results) > 0 else 0.0
    
    # Lexical overconfidence: confident language when wrong (improved)
    confident_terms = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously', 'undoubtedly', 'without doubt', 'for sure']
    confident_densities = []
    for text, acc in zip(texts, accuracies):
        word_count = max(len(text.split()), 1)
        confident_count = sum([text.lower().count(term) for term in confident_terms])
        confident_density = confident_count / word_count
        confident_densities.append(confident_density)
    
    # Lexical overconfidence: average confident language density in wrong answers
    wrong_answers_indices = [i for i, acc in enumerate(accuracies) if acc < 0.5]
    lexical_overconfidence = np.mean([confident_densities[i] for i in wrong_answers_indices]) if wrong_answers_indices else 0.0
    
    # Uncertainty-confidence correlation (using true difficulty-based uncertainty)
    uncertainty_confidence_corr = confidence_accuracy_correlation(true_uncertainties, confidences)
    
    # Overall hedge density
    hedge_density = np.mean(hedge_densities)
    
    # Confident wrong rate - adjust for calibrated scale
    confident_threshold = np.percentile(confidences, 75)  # Top 25% of confidences
    confident_wrong_rate = np.mean([1 if c > confident_threshold and a < 0.5 else 0 for c, a in zip(confidences, accuracies)])
    
    return {
        'hedge_precision': hedge_precision,
        'hedge_recall': hedge_recall,
        'hedge_specificity': hedge_specificity,
        'hedge_f1': hedge_f1,
        'hedge_accuracy': hedge_accuracy,
        'lexical_overconfidence_index': lexical_overconfidence,
        'uncertainty_confidence_correlation': uncertainty_confidence_corr,
        'hedge_density': hedge_density,
        'confident_wrong_rate': confident_wrong_rate,
        # Diagnostic information
        'hedging_diagnostics': {
            'uncertainty_threshold': uncertainty_threshold,
            'certainty_threshold': certainty_threshold,
            'hedge_threshold': hedge_threshold,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'avg_true_uncertainty': np.mean(true_uncertainties),
            'avg_hedge_density': hedge_density
        }
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
    
    # Calculate H5 score as average of 3 key metrics
    # Normalize metrics to 0-1 scale where higher is better
    normalized_quality_corr = max(0, quality_confidence_corr)  # Already 0-1, higher is better
    normalized_weighted_ece = max(0, 1 - quality_weighted_ece)  # Invert ECE (lower is better)
    normalized_calibration_gap = max(0, quality_calibration_gap)  # Higher gap is better (shows source discrimination)
    
    h5_score = (normalized_quality_corr + normalized_weighted_ece + normalized_calibration_gap) / 3
    
    return {
        'source_quality_confidence_correlation': quality_confidence_corr,
        'source_diversity_calibration_correlation': diversity_calibration_corr,
        'quality_weighted_ece': quality_weighted_ece,
        'high_quality_source_ece': high_quality_ece,
        'low_quality_source_ece': low_quality_ece,
        'quality_calibration_gap': quality_calibration_gap,
        'h5_source_quality_score': h5_score
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
    """Compute all CALM-RAG metrics (excluding faithfulness, which is calculated separately)."""
    all_metrics = {}
    
    # H1, H3-H5 metrics
    all_metrics.update(calm_rag_h1_metrics(results))
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
            # Try multiple possible text fields
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '') or doc.get('title', '')
        else:
            text = str(doc)
        if text and len(text.strip()) > 10:  # Only include substantial text
            source_texts.append(text.strip())
    
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
    if not prediction or not source_text:
        return 0.0
    
    pred_tokens = set(normalize_text(prediction).split())
    source_tokens = set(normalize_text(source_text).split())
    
    if not pred_tokens:
        return 0.0
    
    # Remove very common words that don't add meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    pred_tokens = pred_tokens - stop_words
    source_tokens = source_tokens - stop_words
    
    if not pred_tokens:
        return 0.0
    
    intersection = pred_tokens.intersection(source_tokens)
    overlap_ratio = len(intersection) / len(pred_tokens)
    
    # Boost score if there's substantial overlap
    if overlap_ratio > 0.3:
        return min(1.0, overlap_ratio * 1.2)
    
    return overlap_ratio


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
    
    # Extract claims from prediction (improved sentence splitting)
    prediction_sentences = []
    for sentence in prediction.split('.'):
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Only substantial sentences
            prediction_sentences.append(sentence)
    
    # Extract text from sources
    source_texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '') or doc.get('title', '')
        else:
            text = str(doc)
        if text and len(text.strip()) > 10:  # Only substantial text
            source_texts.append(text.strip())
    
    if not source_texts or not prediction_sentences:
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
    
    # Calculate metrics
    attribution_coverage = attributed_claims / total_claims if total_claims > 0 else 0.0
    source_utilization = used_sources / len(source_texts) if source_texts else 0.0
    
    return {
        'attribution_coverage': attribution_coverage,
        'source_utilization': source_utilization,
        'claim_attribution_rate': attribution_coverage,
        'attribution_precision': attribution_coverage
    }


def _can_claim_be_attributed(claim: str, source_texts: List[str]) -> bool:
    """Check if a claim can be attributed to any source text."""
    if not claim or not source_texts:
        return False
    
    claim_tokens = set(normalize_text(claim).split())
    
    # Remove stop words for better matching
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    claim_tokens = claim_tokens - stop_words
    
    if not claim_tokens:
        return False
    
    best_overlap_ratio = 0.0
    
    for source_text in source_texts:
        source_tokens = set(normalize_text(source_text).split())
        source_tokens = source_tokens - stop_words
        
        if not source_tokens:
            continue
            
        # Check if significant portion of claim tokens appear in source
        overlap = len(claim_tokens.intersection(source_tokens))
        overlap_ratio = overlap / len(claim_tokens)
        
        best_overlap_ratio = max(best_overlap_ratio, overlap_ratio)
    
    # Require at least 25% overlap for attribution
    return best_overlap_ratio >= 0.25


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
            text = doc.get('text', '') or doc.get('content', '') or doc.get('excerpt', '') or doc.get('title', '')
        else:
            text = str(doc)
        if text and len(text.strip()) > 10:  # Only substantial text
            source_texts.append(text.strip())
    
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
    
    # Source coverage - how many sources are used (lower threshold)
    used_sources = sum(1 for score in overlap_scores if score > 0.05)
    source_coverage = used_sources / len(source_texts) if source_texts else 0.0
    
    # Grounding consistency - variance in overlap scores
    grounding_consistency = 1.0 - np.var(overlap_scores) if len(overlap_scores) > 1 else 1.0
    
    # Source relevance - average relevance of used sources
    relevant_scores = [score for score in overlap_scores if score > 0.05]
    source_relevance = np.mean(relevant_scores) if relevant_scores else 0.0
    
    return {
        'grounding_score': grounding_score,
        'source_coverage': source_coverage,
        'grounding_consistency': grounding_consistency,
        'source_relevance': source_relevance
    }


def calm_rag_faithfulness_metrics_with_individuals(results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[float]]:
    """
    Calculate comprehensive faithfulness metrics for CALM-RAG evaluation, returning both batch metrics and individual scores.
    
    Args:
        results: List of result dictionaries with CALM-RAG schema
    
    Returns:
        Tuple of (batch_metrics, individual_faithfulness_scores)
    """
    if not results:
        return {}, []
    
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
    individual_faithfulness_scores = []
    
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
        
        # Calculate individual faithfulness score with better weighting
        # Weight the components based on their importance
        overlap_weight = 0.3
        attribution_weight = 0.3
        hallucination_weight = 0.2
        grounding_weight = 0.2
        
        individual_faithfulness = (
            overlap * overlap_weight +
            attribution['attribution_coverage'] * attribution_weight +
            (1.0 - hallucination['hallucination_rate']) * hallucination_weight +
            grounding['grounding_score'] * grounding_weight
        )
        individual_faithfulness_scores.append(individual_faithfulness)
    
    # Calculate average metrics
    metrics = {}
    
    # Answer-source overlap metrics
    metrics['answer_source_overlap'] = np.mean(overlap_scores) if overlap_scores else 0.0
    metrics['answer_source_overlap_std'] = np.std(overlap_scores) if overlap_scores else 0.0
    
    # Attribution metrics
    for key, values in attribution_metrics.items():
        metrics[key] = np.mean(values) if values else 0.0
        metrics[f'{key}_std'] = np.std(values) if values else 0.0
    
    # Hallucination metrics
    for key, values in hallucination_metrics.items():
        metrics[key] = np.mean(values) if values else 0.0
        metrics[f'{key}_std'] = np.std(values) if values else 0.0
    
    # Grounding metrics
    for key, values in grounding_metrics.items():
        metrics[key] = np.mean(values) if values else 0.0
        metrics[f'{key}_std'] = np.std(values) if values else 0.0
    
    # Overall faithfulness score (composite metric with weighted components)
    faithfulness_components = [
        metrics['answer_source_overlap'] * 0.3,
        (1.0 - metrics['hallucination_rate']) * 0.2,  # Invert hallucination rate
        metrics['attribution_coverage'] * 0.3,
        metrics['grounding_score'] * 0.2
    ]
    metrics['overall_faithfulness'] = sum(faithfulness_components)
    
    return metrics, individual_faithfulness_scores


def calm_rag_faithfulness_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate comprehensive faithfulness metrics for CALM-RAG evaluation.
    
    Args:
        results: List of result dictionaries with CALM-RAG schema
    
    Returns:
        Dictionary of faithfulness metrics
    """
    batch_metrics, _ = calm_rag_faithfulness_metrics_with_individuals(results)
    return batch_metrics

