#!/usr/bin/env python3
"""
Comprehensive RAG Model Evaluation Script for CALM-RAG Dataset
Tests various models (GPT, Claude, etc.) and computes all metrics for comparison.
"""

import json
import os
import time
import openai
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
    print("Successfully loaded .env file")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Please ensure your .env file exists and has proper UTF-8 encoding")
    print("Or set environment variables manually")

import numpy as np
from sklearn.isotonic import IsotonicRegression

# Import Mistral AI components
try:
    from mistralai.client import MistralClient
    from mistralai.models import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
from metrics import (
    compute_all_calm_rag_metrics,
    calculate_all_utility_metrics,
    calm_rag_h1_metrics,
    calm_rag_h2_metrics,
    calm_rag_h3_metrics,
    calm_rag_h4_metrics,
    calm_rag_h5_metrics,
    expected_calibration_error,
    calculate_continuous_uncertainty,
    calculate_soft_accuracy,
    LLMGrader,
    create_llm_grader,
    calculate_llm_accuracy
)
from prompts import format_prompt, extract_confidence_from_response

def round_metrics(data: Union[Dict, List, float], precision: int = 3) -> Union[Dict, List, float]:
    """
    Recursively round all numeric values in a nested data structure.
    
    Args:
        data: The data structure to round
        precision: Number of decimal places to round to
    
    Returns:
        Data structure with rounded numeric values
    """
    if isinstance(data, dict):
        return {key: round_metrics(value, precision) for key, value in data.items()}
    elif isinstance(data, list):
        return [round_metrics(item, precision) for item in data]
    elif isinstance(data, (int, float)):
        return round(data, precision)
    else:
        return data

def make_json_serializable(data: Any) -> Any:
    """
    Convert data to JSON-serializable format by handling numpy types and other non-serializable objects.
    
    Args:
        data: The data to make JSON serializable
    
    Returns:
        JSON-serializable version of the data
    """
    if isinstance(data, dict):
        return {key: make_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, (int, float)):
        # Handle numpy types
        if hasattr(data, 'item'):
            return data.item()
        return data
    elif hasattr(data, 'tolist'):  # numpy arrays
        return data.tolist()
    elif data is None or isinstance(data, (str, bool)):
        return data
    else:
        # Convert other types to string as fallback
        return str(data)

def simplify_metric_names(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify metric names by removing redundant prefixes and making them more readable.
    
    Args:
        metrics: Dictionary of metrics with potentially long names
    
    Returns:
        Dictionary with simplified metric names
    """
    simplified = {}
    
    # Define mapping for common metric name simplifications
    name_mappings = {
        # H1 metrics
        'h1_retrieval_recall_confidence_correlation': 'retrieval_recall_confidence_corr',
        'h1_avg_retrieval_recall': 'avg_retrieval_recall',
        'h1_overconfidence_index': 'overconfidence_index',
        'h1_wrong_answer_rate': 'wrong_answer_rate',
        'h1_refusal_rate': 'refusal_rate',
        
        # H2 metrics
        'h2_ece_with_retrieval': 'ece_with_retrieval',
        'h2_ece_without_retrieval': 'ece_without_retrieval',
        'h2_ece_difference': 'ece_improvement',
        'h2_brier_with_retrieval': 'brier_with_retrieval',
        'h2_brier_without_retrieval': 'brier_without_retrieval',
        'h2_brier_difference': 'brier_improvement',
        'h2_confidence_accuracy_corr_with_retrieval': 'confidence_accuracy_corr_with_retrieval',
        'h2_confidence_accuracy_corr_without_retrieval': 'confidence_accuracy_corr_without_retrieval',
        'h2_correlation_difference': 'correlation_improvement',
        
        # H3 metrics
        'h3_hedge_precision': 'hedge_precision',
        'h3_hedge_recall': 'hedge_recall',
        'h3_hedge_f1': 'hedge_f1',
        'h3_lexical_overconfidence_index': 'lexical_overconfidence_index',
        'h3_uncertainty_confidence_correlation': 'uncertainty_confidence_corr',
        'h3_hedge_density': 'hedge_density',
        'h3_confident_wrong_rate': 'confident_wrong_rate',
        'h3_hedge_sophistication': 'hedge_sophistication',
        'h3_advanced_hedge_detection_rate': 'advanced_hedge_detection_rate',
        
        # H4 metrics
        'h4_expected_calibration_error': 'expected_calibration_error',
        'h4_brier_score': 'brier_score',
        'h4_confidence_accuracy_correlation': 'confidence_accuracy_correlation',
        'h4_calibration_ece_after_isotonic': 'calibration_ece_after_isotonic',
        'h4_calibration_improvement': 'calibration_improvement',
        'h4_confidence_distribution_entropy': 'confidence_distribution_entropy',
        'h4_human_model_confidence_correlation': 'human_model_confidence_corr',
        
        # H5 metrics
        'h5_source_quality_confidence_correlation': 'source_quality_confidence_corr',
        'h5_source_diversity_calibration_correlation': 'source_diversity_calibration_corr',
        'h5_quality_weighted_ece': 'quality_weighted_ece',
        'h5_high_quality_source_ece': 'high_quality_source_ece',
        'h5_low_quality_source_ece': 'low_quality_source_ece',
        'h5_quality_calibration_gap': 'quality_calibration_gap',
        
        # Retrieval metrics
        'retrieval_avg_retrieval_recall': 'avg_retrieval_recall',
        'retrieval_avg_retrieval_precision': 'avg_retrieval_precision',
        'retrieval_avg_retrieval_f1': 'avg_retrieval_f1',
        'retrieval_recall_confidence_correlation': 'recall_confidence_correlation',
        'retrieval_accuracy_confidence_correlation': 'accuracy_confidence_correlation',
        'retrieval_avg_source_quality': 'avg_source_quality',
        'retrieval_source_quality_std': 'source_quality_std',
        'retrieval_source_quality_confidence_correlation': 'source_quality_confidence_correlation',
        'retrieval_avg_source_diversity': 'avg_source_diversity',
        'retrieval_source_diversity_std': 'source_diversity_std',
        'retrieval_source_diversity_accuracy_correlation': 'source_diversity_accuracy_correlation',
    }
    
    for key, value in metrics.items():
        # Use mapping if available, otherwise keep original name
        simplified_key = name_mappings.get(key, key)
        simplified[simplified_key] = value
    
    return simplified

def add_metric_descriptions(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add descriptions to metrics to make them more interpretable.
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Dictionary with metric descriptions added
    """
    descriptions = {
        # H1 metrics
        'retrieval_recall_confidence_corr': 'Correlation between retrieval recall and confidence (higher is better)',
        'avg_retrieval_recall': 'Average percentage of relevant documents retrieved (0-1, higher is better)',
        'overconfidence_index': 'Measure of overconfidence in retrieval (0-1, lower is better)',
        'wrong_answer_rate': 'Percentage of wrong answers given (0-1, lower is better)',
        'refusal_rate': 'Percentage of questions refused to answer (0-1, context-dependent)',
        
        # H2 metrics
        'ece_with_retrieval': 'Expected Calibration Error with retrieval (0-1, lower is better)',
        'ece_without_retrieval': 'Expected Calibration Error without retrieval (0-1, lower is better)',
        'ece_improvement': 'Improvement in calibration from retrieval (negative is better)',
        'brier_with_retrieval': 'Brier score with retrieval (0-1, lower is better)',
        'brier_without_retrieval': 'Brier score without retrieval (0-1, lower is better)',
        'brier_improvement': 'Improvement in Brier score from retrieval (negative is better)',
        'confidence_accuracy_corr_with_retrieval': 'Correlation between confidence and accuracy with retrieval (higher is better)',
        'confidence_accuracy_corr_without_retrieval': 'Correlation between confidence and accuracy without retrieval (higher is better)',
        'correlation_improvement': 'Improvement in confidence-accuracy correlation from retrieval (positive is better)',
        
        # H3 metrics
        'hedge_precision': 'Precision of hedge detection (0-1, higher is better)',
        'hedge_recall': 'Recall of hedge detection (0-1, higher is better)',
        'hedge_f1': 'F1 score of hedge detection (0-1, higher is better)',
        'lexical_overconfidence_index': 'Measure of overconfidence in language (0-1, lower is better)',
        'uncertainty_confidence_corr': 'Correlation between uncertainty and confidence (higher is better)',
        'hedge_density': 'Density of hedging language in responses (0-1, context-dependent)',
        'confident_wrong_rate': 'Rate of confident but wrong answers (0-1, lower is better)',
        'hedge_sophistication': 'Sophistication of hedging language (0-1, higher is better)',
        'advanced_hedge_detection_rate': 'Rate of advanced hedge pattern detection (0-1, higher is better)',
        
        # H4 metrics
        'expected_calibration_error': 'Overall expected calibration error (0-1, lower is better)',
        'brier_score': 'Overall Brier score (0-1, lower is better)',
        'confidence_accuracy_correlation': 'Overall confidence-accuracy correlation (higher is better)',
        'calibration_ece_after_isotonic': 'ECE after isotonic calibration (0-1, lower is better)',
        'calibration_improvement': 'Improvement from calibration (positive is better)',
        'confidence_distribution_entropy': 'Entropy of confidence distribution (higher indicates more uncertainty)',
        'human_model_confidence_corr': 'Correlation between human and model confidence (higher is better)',
        
        # H5 metrics
        'source_quality_confidence_corr': 'Correlation between source quality and confidence (higher is better)',
        'source_diversity_calibration_corr': 'Correlation between source diversity and calibration (higher is better)',
        'quality_weighted_ece': 'Quality-weighted expected calibration error (0-1, lower is better)',
        'high_quality_source_ece': 'ECE for high-quality sources (0-1, lower is better)',
        'low_quality_source_ece': 'ECE for low-quality sources (0-1, lower is better)',
        'quality_calibration_gap': 'Gap in calibration between quality levels (lower is better)',
        
        # Retrieval metrics
        'avg_retrieval_precision': 'Average precision of retrieved documents (0-1, higher is better)',
        'avg_retrieval_f1': 'Average F1 score of retrieval (0-1, higher is better)',
        'recall_confidence_correlation': 'Correlation between recall and confidence (higher is better)',
        'accuracy_confidence_correlation': 'Correlation between accuracy and confidence (higher is better)',
        'avg_source_quality': 'Average quality of retrieved sources (0-1, higher is better)',
        'source_quality_std': 'Standard deviation of source quality (lower indicates more consistent quality)',
        'source_quality_confidence_correlation': 'Correlation between source quality and confidence (higher is better)',
        'avg_source_diversity': 'Average diversity of retrieved sources (0-1, higher is better)',
        'source_diversity_std': 'Standard deviation of source diversity (lower indicates more consistent diversity)',
        'source_diversity_accuracy_correlation': 'Correlation between source diversity and accuracy (higher is better)',
    }
    
    # Add descriptions to metrics
    result = {}
    for key, value in metrics.items():
        result[key] = {
            'value': value,
            'description': descriptions.get(key, 'No description available')
        }
    
    return result

def generate_calm_rag_report(evaluation_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a minimal CALM-RAG report focusing only on the core hypotheses.
    
    Args:
        evaluation_summary: Full evaluation summary
    
    Returns:
        Streamlined CALM-RAG report
    """
    calm_rag_metrics = evaluation_summary.get('calm_rag_metrics', {})
    
    # Extract only the core CALM-RAG metrics for the 5 hypotheses
    core_report = {
        'model': evaluation_summary['model'],
        'total_evaluations': evaluation_summary['successful_evaluations'],
        'calm_rag_score': 0.0,  # Will calculate overall score
        
        # H1: Overconfidence under sparse/noisy evidence
        'h1_overconfidence': {
            'avg_retrieval_recall': calm_rag_metrics.get('avg_retrieval_recall', {}).get('value', 0.0),
            'overconfidence_index': calm_rag_metrics.get('overconfidence_index', {}).get('value', 0.0),
            'wrong_answer_rate': calm_rag_metrics.get('wrong_answer_rate', {}).get('value', 0.0)
        },
        
        # H2: Calibration difference with and without retrieval
        'h2_calibration': {
            'ece_with_retrieval': calm_rag_metrics.get('ece_with_retrieval', {}).get('value', 0.0),
            'ece_without_retrieval': calm_rag_metrics.get('ece_without_retrieval', {}).get('value', 0.0),
            'ece_improvement': calm_rag_metrics.get('ece_improvement', {}).get('value', 0.0)
        },
        
        # H3: Hedging language as signal of uncertainty
        'h3_hedging': {
            'hedge_precision': calm_rag_metrics.get('hedge_precision', {}).get('value', 0.0),
            'hedge_recall': calm_rag_metrics.get('hedge_recall', {}).get('value', 0.0),
            'hedge_f1': calm_rag_metrics.get('hedge_f1', {}).get('value', 0.0)
        },
        
        # H4: Self-assessment and numeric calibration
        'h4_self_assessment': {
            'expected_calibration_error': calm_rag_metrics.get('expected_calibration_error', {}).get('value', 0.0),
            'brier_score': calm_rag_metrics.get('brier_score', {}).get('value', 0.0),
            'confidence_accuracy_correlation': calm_rag_metrics.get('confidence_accuracy_correlation', {}).get('value', 0.0)
        },
        
        # H5: Source quality impact on calibration
        'h5_source_quality': {
            'source_quality_confidence_corr': calm_rag_metrics.get('source_quality_confidence_corr', {}).get('value', 0.0),
            'quality_weighted_ece': calm_rag_metrics.get('quality_weighted_ece', {}).get('value', 0.0),
            'quality_calibration_gap': calm_rag_metrics.get('quality_calibration_gap', {}).get('value', 0.0)
        }
    }
    
    # Calculate overall CALM-RAG score (weighted average of key metrics)
    key_metrics = [
        core_report['h1_overconfidence']['avg_retrieval_recall'],
        1 - core_report['h1_overconfidence']['overconfidence_index'],  # Lower is better
        1 - core_report['h2_calibration']['ece_with_retrieval'],  # Lower is better
        core_report['h3_hedging']['hedge_f1'],
        1 - core_report['h4_self_assessment']['expected_calibration_error'],  # Lower is better
        core_report['h5_source_quality']['source_quality_confidence_corr']
    ]
    
    core_report['calm_rag_score'] = round(sum(key_metrics) / len(key_metrics), 3)
    
    return core_report

class RAGModelEvaluator:
    """Evaluates RAG models on the CALM-RAG dataset."""
    
    def __init__(self, dataset_path: str = "calmrag_dataset.json", output_dir: str = "evaluation_results", 
                 use_soft_accuracy: bool = True, use_llm_grading: bool = False, 
                 llm_grading_model: str = "gpt-4o", llm_grading_temperature: float = 0.0):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_soft_accuracy = use_soft_accuracy
        self.use_llm_grading = use_llm_grading
        self.llm_grading_model = llm_grading_model
        self.llm_grading_temperature = llm_grading_temperature
        self.dataset = self._load_dataset()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.mistral_client = None
        self.llama_client = None
        
        # Initialize LLM grader if enabled
        self.llm_grader = None
        if self.use_llm_grading:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.llm_grader = create_llm_grader(
                    openai_api_key, 
                    self.llm_grading_model, 
                    self.llm_grading_temperature
                )
                print(f"LLM grader initialized with model: {self.llm_grading_model}")
            else:
                print("Warning: LLM grading enabled but no OpenAI API key found. Falling back to soft accuracy.")
                self.use_llm_grading = False
        
        # Calibration state
        self.calibration_function = None
        self.is_calibrated = False
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the CALM-RAG dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def setup_openai(self, api_key: str, model: str = "gpt-4o"):
        """Setup OpenAI client."""
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.openai_model = model
    
    def setup_anthropic(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Setup Anthropic client."""
        try:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            self.anthropic_model = model
        except ImportError:
            print("Anthropic client not available. Install with: pip install anthropic")
    
    def setup_google(self, api_key: str, model: str = "gemini-1.5-pro"):
        """Setup Google Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.google_client = genai
            self.google_model = model
        except ImportError:
            print("Google Generative AI client not available. Install with: pip install google-generativeai")
    
    def setup_mistral(self, api_key: str, model: str = "mistral-large-latest"):
        """Setup Mistral AI client."""
        if MISTRAL_AVAILABLE:
            self.mistral_client = MistralClient(api_key=api_key)
            self.mistral_model = model
        else:
            print("Mistral AI client not available. Install with: pip install mistralai")
    
    def setup_llama(self, api_key: str, model: str = "llama-3.1-8b-instruct"):
        """Setup Llama client (using Together AI or similar provider)."""
        try:
            import openai
            # Using OpenAI-compatible API for Llama models
            self.llama_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1"  # Together AI endpoint
            )
            self.llama_model = model
        except ImportError:
            print("OpenAI client not available for Llama. Install with: pip install openai")
    
    def create_rag_prompt(self, question: str, sources: List[Dict[str, Any]], model_name: str = "gpt-4o") -> str:
        """Create a RAG prompt with the question and retrieved sources."""
        # Determine if model supports log probabilities
        logprobs_supported = model_name.startswith(('gpt', 'mistral', 'llama'))
        
        # Only ask for confidence if model doesn't support logprobs
        include_confidence = not logprobs_supported
        
        # Use the appropriate prompt from prompts.py
        return format_prompt(question, sources, model_type="openai", include_confidence=include_confidence)
    
    def calculate_internal_confidence(self, log_probs: List[Dict]) -> float:
        """
        Calculate internal confidence from token log probabilities.
        Higher average log probability = higher confidence.
        """
        if not log_probs:
            return 0.5
        
        # Convert log probs to probabilities
        probs = [np.exp(log_prob['logprob']) for log_prob in log_probs]
        
        # Calculate average probability (higher = more confident)
        avg_prob = np.mean(probs)
        
        # Better normalization that preserves more variation
        # Use the geometric mean of probabilities for better sensitivity
        if len(probs) > 1:
            # Geometric mean gives more sensitivity to low probabilities
            geom_mean = np.exp(np.mean([np.log(p) for p in probs]))
            confidence = geom_mean
        else:
            confidence = avg_prob
        
        # Scale to 0-1 range with more sensitivity
        # Most probabilities are between 0.1-0.9, so we can use a more direct mapping
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def calibrate_log_probs_to_confidence(self, log_probs_list: List[List[Dict]], 
                                         accuracies: List[float]) -> callable:
        """
        Use isotonic regression to calibrate log probs to confidence scores.
        """
        # Calculate raw confidence scores from log probs
        raw_confidences = [self.calculate_internal_confidence(lp) for lp in log_probs_list]
        
        # Filter out None values
        valid_indices = [i for i, conf in enumerate(raw_confidences) if conf is not None]
        valid_raw_confidences = [raw_confidences[i] for i in valid_indices]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        
        if len(valid_raw_confidences) < 2:
            # Not enough data for calibration, return identity function
            return lambda log_probs: self.calculate_internal_confidence(log_probs)
        
        # Fit isotonic regression: raw_confidence -> actual_accuracy
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(valid_raw_confidences, valid_accuracies)
        
        # Return calibration function
        def calibrate_confidence(log_probs):
            raw_conf = self.calculate_internal_confidence(log_probs)
            if raw_conf is None:
                return 0.5
            return iso_reg.transform([raw_conf])[0]
        
        return calibrate_confidence
    
    def update_calibration(self, evaluation_results: List[Dict[str, Any]], raw_results: List[Dict[str, Any]]):
        """
        Update calibration function using evaluation results.
        Uses raw_results (with log_probs) for calibration, not the cleaned evaluation_results.
        """
        log_probs_list = []
        accuracies = []
        
        # Use raw_results for log_probs and evaluation_results for accuracy values
        for i, result in enumerate(raw_results):
            if result and 'log_probs' in result and result['log_probs']:
                log_probs_list.append(result['log_probs'])
                # Get accuracy from evaluation_results, not raw_results
                if i < len(evaluation_results):
                    accuracies.append(evaluation_results[i].get('accuracy', 0.5))
                else:
                    accuracies.append(0.5)
        
        if len(log_probs_list) >= 10:  # Need minimum data for calibration
            self.calibration_function = self.calibrate_log_probs_to_confidence(
                log_probs_list, accuracies
            )
            self.is_calibrated = True
            print(f"Calibration updated with {len(log_probs_list)} samples")
        else:
            print(f"Not enough data for calibration ({len(log_probs_list)} samples)")
    
    def get_calibrated_confidence(self, log_probs: List[Dict]) -> float:
        """
        Get calibrated confidence score using the calibration function.
        """
        if self.calibration_function and self.is_calibrated:
            return self.calibration_function(log_probs)
        else:
            return self.calculate_internal_confidence(log_probs)
    
    
    
    def call_openai_model(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Call OpenAI model with log probabilities for internal confidence calculation."""
        try:
            # Check if model supports log probabilities
            logprobs_supported = self.openai_model in ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k']
            
            if logprobs_supported:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,  # Use 0.0 for deterministic log probs
                    max_tokens=1000,
                    logprobs=True,  # Enable log probabilities
                    top_logprobs=5   # Get top 5 token probabilities
                )
            else:
                # For models that don't support logprobs, use regular completion
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
            
            response_text = response.choices[0].message.content
            
            # Extract log probabilities for each token
            log_probs = []
            if logprobs_supported and response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_info in response.choices[0].logprobs.content:
                    # Convert top_logprobs to serializable format
                    serializable_top_logprobs = []
                    if token_info.top_logprobs:
                        for top_prob in token_info.top_logprobs:
                            serializable_top_logprobs.append({
                                'token': top_prob.token,
                                'logprob': top_prob.logprob
                            })
                    
                    log_probs.append({
                        'token': token_info.token,
                        'logprob': token_info.logprob,
                        'top_logprobs': serializable_top_logprobs
                    })
                # Calculate calibrated internal confidence from log probabilities
                internal_confidence = self.get_calibrated_confidence(log_probs)
            else:
                # For models without logprobs, extract confidence from text
                internal_confidence = extract_confidence_from_response(response_text, "openai") or 0.5
            
            return {
                'response': response_text,
                'confidence': internal_confidence,
                'log_probs': log_probs,
                'model': self.openai_model,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'success': True
            }
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'log_probs': [],
                'model': self.openai_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def call_anthropic_model(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Call Anthropic Claude model."""
        try:
            response = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Extract confidence from response text
            confidence = extract_confidence_from_response(response_text, "anthropic")
            
            return {
                'response': response_text,
                'confidence': confidence or 0.5,  # Fallback to 0.5 if extraction fails
                'log_probs': [],  # No log probs available for Anthropic
                'model': self.anthropic_model,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens if response.usage else 0,
                'success': True
            }
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'model': self.anthropic_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def call_google_model(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Call Google Gemini model."""
        try:
            model = self.google_client.GenerativeModel(self.google_model)
            response = model.generate_content(
                prompt,
                generation_config=self.google_client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000
                )
            )
            
            response_text = response.text
            
            # Extract confidence from response text
            confidence = extract_confidence_from_response(response_text, "gemini")
            
            return {
                'response': response_text,
                'confidence': confidence or 0.5,  # Fallback to 0.5 if extraction fails
                'log_probs': [],  # No log probs available for Gemini
                'model': self.google_model,
                'tokens_used': 0,  # Gemini doesn't provide token usage in basic API
                'success': True
            }
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'model': self.google_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def call_mistral_model(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Call Mistral AI model with log probabilities."""
        try:
            # Create chat message
            messages = [ChatMessage(role="user", content=prompt)]
            
            # Call Mistral API with log probabilities
            response = self.mistral_client.chat(
                model=self.mistral_model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                logprobs=True,  # Enable log probabilities
                top_logprobs=5   # Get top 5 token probabilities
            )
            
            response_text = response.choices[0].message.content
            
            # Extract log probabilities
            log_probs = []
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                for token_info in response.choices[0].logprobs.content:
                    log_probs.append({
                        'token': token_info.token,
                        'logprob': token_info.logprob,
                        'top_logprobs': getattr(token_info, 'top_logprobs', [])
                    })
            
            # Calculate internal confidence from log probabilities
            if log_probs:
                internal_confidence = self.get_calibrated_confidence(log_probs)
            else:
                # Fallback to text extraction if no log probs
                internal_confidence = extract_confidence_from_response(response_text, "mistral") or 0.5
            
            return {
                'response': response_text,
                'confidence': internal_confidence,
                'log_probs': log_probs,
                'model': self.mistral_model,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'success': True
            }
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'log_probs': [],
                'model': self.mistral_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def call_llama_model(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Call Llama model with log probabilities (via Together AI or similar)."""
        try:
            # Using OpenAI-compatible API for Llama models
            response = self.llama_client.chat.completions.create(
                model=self.llama_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
                logprobs=True,  # Enable log probabilities
                top_logprobs=5   # Get top 5 token probabilities
            )
            
            response_text = response.choices[0].message.content
            
            # Extract log probabilities
            log_probs = []
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_info in response.choices[0].logprobs.content:
                    # Convert top_logprobs to serializable format
                    serializable_top_logprobs = []
                    if token_info.top_logprobs:
                        for top_prob in token_info.top_logprobs:
                            serializable_top_logprobs.append({
                                'token': top_prob.token,
                                'logprob': top_prob.logprob
                            })
                    
                    log_probs.append({
                        'token': token_info.token,
                        'logprob': token_info.logprob,
                        'top_logprobs': serializable_top_logprobs
                    })
            
            # Calculate internal confidence from log probabilities
            if log_probs:
                internal_confidence = self.get_calibrated_confidence(log_probs)
            else:
                # Fallback to text extraction if no log probs
                internal_confidence = extract_confidence_from_response(response_text, "llama") or 0.5
            
            return {
                'response': response_text,
                'confidence': internal_confidence,
                'log_probs': log_probs,
                'model': self.llama_model,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'success': True
            }
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'log_probs': [],
                'model': self.llama_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    
    def evaluate_single_entry(self, entry: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Evaluate a single dataset entry with the specified model."""
        # Create RAG prompt
        all_sources = entry['source_sets']['clear'] + entry['source_sets']['ambiguous']
        prompt = self.create_rag_prompt(entry['question'], all_sources, model_name)
        
        # Call model
        try:
            if model_name.startswith('gpt'):
                result = self.call_openai_model(prompt)
            elif model_name.startswith('claude'):
                result = self.call_anthropic_model(prompt)
            elif model_name.startswith('gemini'):
                result = self.call_google_model(prompt)
            elif model_name.startswith('mistral'):
                result = self.call_mistral_model(prompt)
            elif model_name.startswith('llama'):
                result = self.call_llama_model(prompt)
            else:
                # Try generic method for other models
                result = self.call_model_with_logprobs(prompt, model_name)
        except Exception as e:
            print(f"Error calling model {model_name}: {e}")
            return None
        
        if not result['success']:
            print(f"Model call failed for {model_name}: {result.get('error', 'Unknown error')}")
            return None
        
        # Create evaluation result structure
        retrieved_docs = [{'url': s['url'], 'domain': s['domain'], 'category': s['category']} for s in all_sources]
        relevant_docs = [s['url'] for s in entry['source_sets']['clear']]
        
        # Calculate continuous uncertainty score based on multiple factors
        continuous_uncertainty = calculate_continuous_uncertainty(
            entry, retrieved_docs, entry['question']
        )
        
        # Calculate accuracy by comparing prediction with gold answer
        gold_answer = entry.get('gold_answer', '')
        prediction_text = result['response']
        question = entry.get('question', '')
        
        if gold_answer and prediction_text:
            if self.use_llm_grading and self.llm_grader:
                # Use LLM grading for more nuanced evaluation
                try:
                    # Create context from retrieved sources
                    context_parts = []
                    for doc in retrieved_docs:
                        if isinstance(doc, dict) and 'url' in doc:
                            context_parts.append(f"Source: {doc['url']}")
                    
                    context = "\n".join(context_parts) if context_parts else None
                    
                    grading_result = self.llm_grader.grade_answer(
                        question, prediction_text, gold_answer, context
                    )
                    accuracy = grading_result['accuracy_score']
                    grading_info = grading_result
                except Exception as e:
                    print(f"LLM grading failed for entry {entry['id']}: {e}")
                    # Fallback to soft accuracy
                    accuracy = calculate_soft_accuracy(prediction_text, [gold_answer])
                    grading_info = {
                        'accuracy_score': accuracy,
                        'explanation': f'LLM grading failed, used soft accuracy: {str(e)}',
                        'grading_method': 'soft_accuracy_fallback',
                        'error': str(e)
                    }
            elif self.use_soft_accuracy:
                # Use soft accuracy to compare prediction with gold answer
                accuracy = calculate_soft_accuracy(prediction_text, [gold_answer])
                grading_info = {
                    'accuracy_score': accuracy,
                    'explanation': 'Used soft accuracy calculation',
                    'grading_method': 'soft_accuracy'
                }
            else:
                # Use binary accuracy (exact match)
                accuracy = 1.0 if prediction_text.strip().lower() == gold_answer.strip().lower() else 0.0
                grading_info = {
                    'accuracy_score': accuracy,
                    'explanation': 'Used binary accuracy (exact match)',
                    'grading_method': 'binary_accuracy'
                }
        else:
            # Fallback to confidence-based heuristic if no gold answer available
            accuracy = 0.8 if result['confidence'] and result['confidence'] > 0.7 else 0.5
            grading_info = {
                'accuracy_score': accuracy,
                'explanation': 'No gold answer available, used confidence-based heuristic',
                'grading_method': 'confidence_heuristic'
            }
        
        evaluation_result = {
            'entry_id': entry['id'],
            'question': entry['question'],
            'gold_answer': entry['gold_answer'],
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs,
            'confidence': result['confidence'] or 0.5,
            'accuracy': accuracy,
            'prediction_text': result['response'],
            # Note: log_probs removed from raw results to reduce file size
            'continuous_uncertainty': continuous_uncertainty,  # New continuous uncertainty score
            'is_uncertain': bool(result.get('confidence', 0) < 0.6) if result.get('confidence') is not None else True,
            'human_confidence': entry.get('human_confidence'),
            'no_retrieval_confidence': 0.5,  # Mock value
            'no_retrieval_accuracy': 0.3,    # Mock value
            'model': model_name,
            'tokens_used': result['tokens_used'],
            'accuracy_method': grading_info.get('grading_method', 'unknown')
        }
        
        return evaluation_result, result  # Return both cleaned result and original model response
    
    def evaluate_model(self, model_name: str, max_entries: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a model on the dataset."""
        print(f"\nEvaluating {model_name} on CALM-RAG dataset...")
        
        if max_entries:
            dataset_subset = self.dataset[:max_entries]
        else:
            dataset_subset = self.dataset
        
        results = []
        original_model_responses = []
        successful_evaluations = 0
        
        for entry in tqdm(dataset_subset, desc=f"Evaluating {model_name}"):
            try:
                evaluation_result, original_response = self.evaluate_single_entry(entry, model_name)
                if evaluation_result:
                    results.append(evaluation_result)
                    original_model_responses.append(original_response)
                    successful_evaluations += 1
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error evaluating entry {entry['id']}: {e}")
                continue
        
        print(f"Successfully evaluated {successful_evaluations}/{len(dataset_subset)} entries")
        print(f"Raw results count: {len(results)}")
        
        if not results:
            print("No successful evaluations!")
            return {}
        
        print(f"Stored {len(original_model_responses)} original model responses for backup")
        
        # Update calibration after collecting initial results
        if len(results) >= 10:
            print(f"Updating calibration with {len(results)} samples...")
            
            # Clear LLM grader cache to ensure fresh accuracy calculations
            if self.llm_grader:
                print("Clearing LLM grader cache for fresh accuracy calculations...")
                self.llm_grader.clear_cache()
            
            self.update_calibration(results, original_model_responses)  # Pass original model responses for calibration
            
            # Re-evaluate with calibrated confidence if calibration was successful
            if self.is_calibrated:
                print("Re-evaluating with calibrated confidence...")
                calibrated_results = []
                successful_re_evaluations = 0
                
                for entry in tqdm(dataset_subset, desc=f"Re-evaluating {model_name} with calibration"):
                    try:
                        evaluation_result, original_response = self.evaluate_single_entry(entry, model_name)
                        if evaluation_result:
                            calibrated_results.append(evaluation_result)
                            successful_re_evaluations += 1
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        print(f"Error re-evaluating entry {entry['id']}: {e}")
                        continue
                
                print(f"Re-evaluation completed: {successful_re_evaluations}/{len(dataset_subset)} entries successful")
                
                # Only use calibrated results if we have the same number as original
                if len(calibrated_results) == len(results):
                    results = calibrated_results
                    print("Using calibrated results for final metrics")
                else:
                    print(f"Warning: Re-evaluation incomplete ({len(calibrated_results)} vs {len(results)}). Using original results.")
        
        # Compute all metrics
        print("Computing CALM-RAG metrics...")
        print(f"Final results count for metrics: {len(results)}")
        
        # Core CALM-RAG metrics
        calm_rag_metrics = compute_all_calm_rag_metrics(results)
        
        # Utility metrics
        utility_metrics = calculate_all_utility_metrics(results)
        
        # Individual hypothesis metrics
        h1_metrics = calm_rag_h1_metrics(results)
        h2_metrics = calm_rag_h2_metrics(results)
        h3_metrics = calm_rag_h3_metrics(results)
        h4_metrics = calm_rag_h4_metrics(results)
        h5_metrics = calm_rag_h5_metrics(results)
        
        
        # Build streamlined evaluation summary (CALM-RAG focused)
        evaluation_summary = {
            'model': model_name,
            'total_entries': len(dataset_subset),
            'successful_evaluations': successful_evaluations,
            'calibration_info': {
                'was_calibrated': self.is_calibrated,
                'raw_results_count': len(results),
                'calibrated_results_count': len(results) if self.is_calibrated else 0,
                'used_calibrated_results': self.is_calibrated
            },
            'calm_rag_metrics': calm_rag_metrics,
            'hypothesis_metrics': {
                'h1_overconfidence': h1_metrics,
                'h2_calibration_difference': h2_metrics,
                'h3_hedging_language': h3_metrics,
                'h4_self_assessment': h4_metrics,
                'h5_source_quality': h5_metrics
            },
            # Keep only essential result data (no verbose logging)
            'summary_results': [
                {
                    'entry_id': result['entry_id'],
                    'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                    'confidence': result['confidence'],
                    'accuracy': result['accuracy'],
                    'is_uncertain': result['is_uncertain'],
                    'retrieval_recall': result.get('retrieval_recall', 0.0),
                    'has_hedging': 'hedge' in result.get('prediction_text', '').lower()
                }
                for result in results
            ]
        }
        
        # Simplify metric names, add descriptions, and round numeric values for better readability
        if 'calm_rag_metrics' in evaluation_summary:
            simplified_metrics = simplify_metric_names(evaluation_summary['calm_rag_metrics'])
            evaluation_summary['calm_rag_metrics'] = add_metric_descriptions(simplified_metrics)
        evaluation_summary = round_metrics(evaluation_summary, precision=3)
        
        # Ensure all data is JSON serializable
        evaluation_summary = make_json_serializable(evaluation_summary)
        
        # Generate streamlined CALM-RAG report
        calm_rag_report = generate_calm_rag_report(evaluation_summary)
        
        # Save full results
        output_file = os.path.join(self.output_dir, f"{model_name}_evaluation.json")
        
        # Save with error handling for non-serializable objects
        try:
            with open(output_file, 'w') as f:
                json.dump(evaluation_summary, f, indent=2)
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            print("Attempting to fix serialization issues...")
            
            # Recursively convert problematic objects to strings
            def fix_serialization(obj):
                if isinstance(obj, dict):
                    return {k: fix_serialization(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [fix_serialization(item) for item in obj]
                elif isinstance(obj, (bool, int, float, str)):
                    return obj
                else:
                    return str(obj)
            
            evaluation_summary_fixed = fix_serialization(evaluation_summary)
            
            with open(output_file, 'w') as f:
                json.dump(evaluation_summary_fixed, f, indent=2)
        
        # Save streamlined CALM-RAG report
        report_file = os.path.join(self.output_dir, f"{model_name}_calm_rag_report.json")
        with open(report_file, 'w') as f:
            json.dump(calm_rag_report, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Streamlined CALM-RAG report saved to {report_file}")
        
        # Add grading statistics if using LLM grading
        if self.use_llm_grading and self.llm_grader:
            cache_stats = self.llm_grader.get_cache_stats()
            evaluation_summary['grading_statistics'] = {
                'grading_method': 'llm',
                'model_used': self.llm_grading_model,
                'cache_size': cache_stats['cache_size'],
                'total_evaluations': len(results)
            }
        
        return evaluation_summary
    
    def compare_models(self, model_names: List[str], max_entries: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple models on the dataset."""
        print(f"Comparing {len(model_names)} models on CALM-RAG dataset")
        print("=" * 60)
        
        comparison_results = {}
        
        for model_name in model_names:
            try:
                result = self.evaluate_model(model_name, max_entries)
                if result:
                    comparison_results[model_name] = result
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
        
        # Create comparison summary
        comparison_summary = {
            'models_evaluated': list(comparison_results.keys()),
            'comparison_metrics': {},
            'model_performance': {}
        }
        
        # Extract key metrics for comparison
        for model_name, result in comparison_results.items():
            comparison_summary['model_performance'][model_name] = {
                'successful_evaluations': result['successful_evaluations'],
                'total_entries': result['total_entries'],
                'success_rate': result['successful_evaluations'] / result['total_entries']
            }
            
            # Key CALM-RAG metrics for comparison
            calm_rag = result['calm_rag_metrics']
            comparison_summary['comparison_metrics'][model_name] = {
                'overconfidence_index': calm_rag.get('h1_overconfidence_index', 'N/A'),
                'ece_with_retrieval': calm_rag.get('h2_ece_with_retrieval', 'N/A'),
                'ece_without_retrieval': calm_rag.get('h2_ece_without_retrieval', 'N/A'),
                'hedge_f1': calm_rag.get('h3_hedge_f1', 'N/A'),
                'expected_calibration_error': calm_rag.get('h4_expected_calibration_error', 'N/A'),
                'source_quality_confidence_correlation': calm_rag.get('h5_source_quality_confidence_correlation', 'N/A')
            }
        
        # Round all numeric values for better readability and ensure JSON serialization
        comparison_summary = round_metrics(comparison_summary, precision=3)
        comparison_summary = make_json_serializable(comparison_summary)
        
        # Save comparison
        comparison_file = os.path.join(self.output_dir, "model_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\nComparison saved to {comparison_file}")
        
        return comparison_summary

def main():
    """Main evaluation function."""
    print("CALM-RAG Model Evaluation")
    print("=" * 50)
    
    # Initialize evaluator with LLM grading enabled by default
    evaluator = RAGModelEvaluator(
        use_soft_accuracy=True, 
        use_llm_grading=True,  
        llm_grading_model="gpt-4o",
        llm_grading_temperature=0.0
    )
    
    # Print grading method being used
    if evaluator.use_llm_grading and evaluator.llm_grader:
        print(f"Using LLM grading with model: {evaluator.llm_grading_model}")
    elif evaluator.use_soft_accuracy:
        print("Using soft accuracy for evaluation")
    else:
        print("Using binary accuracy for evaluation")
    print()
    
    # Try to get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_api_key:
        print("OpenAI API key not found. Please create a .env file or api_keys.txt file with your API keys.")
        print("Format for .env file:")
        print("OPENAI_API_KEY=your_key_here")
        print("ANTHROPIC_API_KEY=your_key_here")
        print("\nOr create api_keys.txt with:")
        print("OPENAI_API_KEY=your_key_here")
        print("ANTHROPIC_API_KEY=your_key_here")
        return
    
    # Setup models
    evaluator.setup_openai(openai_api_key, "gpt-4o")
    
    if anthropic_api_key:
        evaluator.setup_anthropic(anthropic_api_key, "claude-3-5-sonnet-20241022")
        models_to_evaluate = ["gpt-4o", "claude-3-5-sonnet-20241022"]
    else:
        models_to_evaluate = ["gpt-4o"]
    
    print(f"Models to evaluate: {models_to_evaluate}")
    
    # Evaluate single model with first 10 entries
    print("\nStarting evaluation with first 10 entries...")
    
    try:
        # Evaluate just the first model (gpt-4o) with 10 entries
        result = evaluator.evaluate_model("gpt-4o", max_entries=10)
        
        if result:
            print("\nEvaluation completed successfully!")
            print(f"Results saved in: {evaluator.output_dir}/")
            
            # Print key findings
            print(f"\nKey Findings:")
            print(f"  Model: gpt-4o")
            print(f"  Entries evaluated: {result['successful_evaluations']}/{result['total_entries']}")
            print(f"  Success rate: {result['successful_evaluations']/result['total_entries']:.1%}")
            
            # Print some key metrics
            calm_rag = result['calm_rag_metrics']
            print(f"\nKey CALM-RAG Metrics:")
            print(f"  Overconfidence Index: {calm_rag.get('h1_overconfidence_index', 'N/A'):.3f}")
            print(f"  ECE with Retrieval: {calm_rag.get('h2_ece_with_retrieval', 'N/A'):.3f}")
            print(f"  Hedge F1: {calm_rag.get('h3_hedge_f1', 'N/A'):.3f}")
            print(f"  Expected Calibration Error: {calm_rag.get('h4_expected_calibration_error', 'N/A'):.3f}")
        else:
            print("Evaluation failed - no results returned")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
