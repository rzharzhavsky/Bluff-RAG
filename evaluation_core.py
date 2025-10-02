"""
BLUFF-RAG Evaluation Core Module - Streamlined Version
Main orchestrator for RAG model evaluation with calibration support.
"""

import json
import os
import time
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Successfully loaded .env file")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Please ensure your .env file exists and has proper UTF-8 encoding")

# Import our streamlined modules
from prompts_core import format_prompt, extract_confidence_from_response, parse_response
from metrics_bluff_rag import (
    compute_all_bluff_rag_metrics, calculate_all_utility_metrics,
    bluff_rag_h1_metrics, bluff_rag_h3_metrics,
    bluff_rag_h4_metrics, bluff_rag_h5_metrics,
    calculate_ambiguity_sensitivity_index, calculate_batch_asi,
    calculate_continuous_uncertainty, calculate_llm_accuracy,
    bluff_rag_faithfulness_metrics, bluff_rag_faithfulness_metrics_with_individuals
)
from calibration import ConfidenceCalibrator

# Import Mistral AI components
try:
    from mistralai.client import MistralClient
    from mistralai.models import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


def round_metrics(data: Union[Dict, List, float], precision: int = 3) -> Union[Dict, List, float]:
    """Recursively round all numeric values in a nested data structure."""
    if isinstance(data, dict):
        return {key: round_metrics(value, precision) for key, value in data.items()}
    elif isinstance(data, list):
        return [round_metrics(item, precision) for item in data]
    elif isinstance(data, (int, float)):
        return round(data, precision)
    else:
        return data


def make_json_serializable(data: Any) -> Any:
    """Convert data to JSON-serializable format."""
    if isinstance(data, dict):
        return {key: make_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, (int, float)):
        if hasattr(data, 'item'):
            return data.item()
        return data
    elif hasattr(data, 'tolist'):
        return data.tolist()
    elif data is None or isinstance(data, (str, bool)):
        return data
    else:
        return str(data)


def generate_bluff_rag_report(evaluation_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive BLUFF-RAG report with structure."""
    bluff_rag_metrics = evaluation_summary.get('bluff_rag_metrics', {})
    faithfulness_metrics = evaluation_summary.get('faithfulness_metrics', {})
    asi_metrics = evaluation_summary.get('asi_metrics', {})
    
    # Calculate answer correctness (average of all accuracies)
    # We need to get this from the individual results if available
    print(f"Evaluation summary keys: {list(evaluation_summary.keys())}")
    all_results = []
    
    # Check if results are in summary_results
    if 'summary_results' in evaluation_summary:
        summary_results = evaluation_summary['summary_results']
        if 'clear_results' in summary_results:
            print(f"Clear results count: {len(summary_results['clear_results'])}")
            all_results.extend(summary_results['clear_results'])
        if 'ambiguous_results' in summary_results:
            print(f"Ambiguous results count: {len(summary_results['ambiguous_results'])}")
            all_results.extend(summary_results['ambiguous_results'])
    else:
        # Fallback to direct keys
        if 'clear_results' in evaluation_summary:
            print(f"Clear results count: {len(evaluation_summary['clear_results'])}")
            all_results.extend(evaluation_summary['clear_results'])
        if 'ambiguous_results' in evaluation_summary:
            print(f"Ambiguous results count: {len(evaluation_summary['ambiguous_results'])}")
            all_results.extend(evaluation_summary['ambiguous_results'])
    
    print(f"Total all_results count: {len(all_results)}")
    
    answer_correctness = 0.0
    amount_of_none_accuracies = 0
    for result in all_results:
        if result.get('accuracy') is None:
            amount_of_none_accuracies += 1
    print(f"Amount of none accuracies: {amount_of_none_accuracies}")
    if all_results:
        accuracies = [result.get('accuracy', 0.0) for result in all_results if result.get('accuracy') is not None]
        if accuracies:
            answer_correctness = sum(accuracies) / len(accuracies)
        else:
            print(f"Warning: No valid accuracy values found in {len(all_results)} results")
    else:
        print("Warning: No results found for answer correctness calculation")
    
    # Create streamlined report structure organized by hypothesis
    core_report = {
        'model': evaluation_summary['model'],
        'total_evaluations': evaluation_summary['successful_evaluations'],
        
        # HYPOTHESIS 1: Overconfidence Gap
        'h1_overconfidence_gap': {
            'evidence_confidence_gap': bluff_rag_metrics.get('evidence_confidence_gap', 0.0),
            'overconfidence_index': bluff_rag_metrics.get('overconfidence_index', 0.0),
            'retrieval_recall_confidence_correlation': bluff_rag_metrics.get('retrieval_recall_confidence_correlation', 0.0),
            'expected_calibration_error': bluff_rag_metrics.get('expected_calibration_error', 0.0),
            'confidence_accuracy_correlation': bluff_rag_metrics.get('confidence_accuracy_correlation', 0.0),
            'avg_retrieval_recall': bluff_rag_metrics.get('avg_retrieval_recall', 0.0),
            'ASI': asi_metrics.get('mean_asi', 0.0)
        },
        
        # HYPOTHESIS 2: Hedging Behavior
        'h2_hedging_behavior': {
            'VUI': bluff_rag_metrics.get('hedge_f1', 0.0),
            'source_set_on_hedging': bluff_rag_metrics.get('source_set_on_hedging', 0.0),
            'lexical_overconfidence_index': bluff_rag_metrics.get('lexical_overconfidence_index', 0.0),
            'hedge_precision': bluff_rag_metrics.get('hedge_precision', 0.0),
            'hedge_recall': bluff_rag_metrics.get('hedge_recall', 0.0)
        },
        
        # DIAGNOSTIC METRICS
        'diagnostics': {
            'answer_correctness': answer_correctness,
            'brier_score': bluff_rag_metrics.get('brier_score', 0.0),
            'wrong_answer_rate': bluff_rag_metrics.get('wrong_answer_rate', 0.0),
            'refusal_rate': bluff_rag_metrics.get('refusal_rate', 0.0),
            'source_awareness_score': bluff_rag_metrics.get('h5_source_quality_score', 0.0),
            'overall_faithfulness': faithfulness_metrics.get('overall_faithfulness', 0.0)
        }
    }
    
    return core_report


class RAGModelEvaluator:
    """Streamlined evaluator for RAG models on the BLUFF-RAG dataset."""
    
    def __init__(self, dataset_path: str = "bluffrag_dataset.json", 
                 output_dir: str = "evaluation_results", 
                 use_llm_grading: bool = True):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_llm_grading = use_llm_grading
        self.dataset = self._load_dataset()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.mistral_client = None
        self.llama_client = None
        
        # Initialize calibration system
        self.calibrator = ConfidenceCalibrator()
        
        # Initialize LLM grading client if needed
        if self.use_llm_grading:
            self._initialize_grading_client()
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the BLUFF-RAG dataset."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _initialize_grading_client(self):
        """Initialize OpenAI client for LLM grading."""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("LLM grading client initialized")
            else:
                print("OPENAI_API_KEY not found. LLM grading will fall back to soft accuracy.")
                self.use_llm_grading = False
        except ImportError:
            print("OpenAI library not installed. LLM grading will fall back to soft accuracy.")
            self.use_llm_grading = False
    
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
            self.llama_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1"
            )
            self.llama_model = model
        except ImportError:
            print("OpenAI client not available for Llama. Install with: pip install openai")
    
    def call_openai_model(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Call OpenAI model with log probabilities for internal confidence calculation."""
        try:
            # Check if model supports log probabilities
            logprobs_supported = self.openai_model in ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k']
            
            if logprobs_supported:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000,
                    logprobs=True,
                    top_logprobs=5
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
            
            response_text = response.choices[0].message.content
            
            # Extract log probabilities
            log_probs = []
            if logprobs_supported and response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_info in response.choices[0].logprobs.content:
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
                
                # Calculate calibrated confidence from log probabilities
                internal_confidence = self.calibrator.get_calibrated_confidence(log_probs)
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
            confidence = extract_confidence_from_response(response_text, "anthropic")
            
            return {
                'response': response_text,
                'confidence': confidence or 0.5,
                'log_probs': [],
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
            confidence = extract_confidence_from_response(response_text, "gemini")
            
            return {
                'response': response_text,
                'confidence': confidence or 0.5,
                'log_probs': [],
                'model': self.google_model,
                'tokens_used': 0,
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
            messages = [ChatMessage(role="user", content=prompt)]
            
            response = self.mistral_client.chat(
                model=self.mistral_model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                logprobs=True,
                top_logprobs=5
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
            
            # Calculate calibrated confidence from log probabilities
            if log_probs:
                internal_confidence = self.calibrator.get_calibrated_confidence(log_probs)
            else:
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
            response = self.llama_client.chat.completions.create(
                model=self.llama_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
                logprobs=True,
                top_logprobs=5
            )
            
            response_text = response.choices[0].message.content
            
            # Extract log probabilities
            log_probs = []
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_info in response.choices[0].logprobs.content:
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
            
            # Calculate calibrated confidence from log probabilities
            if log_probs:
                internal_confidence = self.calibrator.get_calibrated_confidence(log_probs)
            else:
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
    
    def evaluate_single_entry(self, entry: Dict[str, Any], model_name: str, 
                            source_set_type: str = "both") -> Dict[str, Any]:
        """Evaluate a single dataset entry with the specified model."""
        # Select sources based on type
        if source_set_type == "clear":
            sources = entry['source_sets']['clear']
        elif source_set_type == "ambiguous":
            sources = entry['source_sets']['ambiguous']
        else:  # "both"
            sources = entry['source_sets']['clear'] + entry['source_sets']['ambiguous']
        
        prompt = format_prompt(entry['question'], sources, model_name)
        
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
                print(f"Unknown model: {model_name}")
                return None
        except Exception as e:
            print(f"Error calling model {model_name}: {e}")
            return None
        
        if not result['success']:
            print(f"Model call failed for {model_name}: {result.get('error', 'Unknown error')}")
            return None
        
        # Parse response
        parsed = parse_response(result['response'])
        
        # Create evaluation result structure with full source data for faithfulness metrics
        retrieved_docs = []
        for s in sources:
            doc = {
                'url': s['url'], 
                'domain': s['domain'], 
                'category': s['category'],
                'title': s.get('title', ''),
                'text': s.get('text', ''),
                'timestamp': s.get('timestamp', ''),
                'score': s.get('score', None)
            }
            retrieved_docs.append(doc)
        
        # Create relevant_docs in same format as retrieved_docs for proper comparison
        relevant_docs = []
        for s in entry['source_sets']['clear']:
            doc = {
                'url': s['url'], 
                'domain': s['domain'], 
                'category': s['category'],
                'title': s.get('title', ''),
                'text': s.get('text', ''),
                'timestamp': s.get('timestamp', ''),
                'score': s.get('score', None)
            }
            relevant_docs.append(doc)
        
        # Calculate accuracy using LLM grading
        gold_answer = entry.get('gold_answer', '')
        if gold_answer and parsed['answer']:
            if self.use_llm_grading:
                # Use LLM grading for semantic accuracy
                accuracy = calculate_llm_accuracy(
                    prediction=parsed['answer'],
                    gold_answer=gold_answer,
                    question=entry.get('question', ''),
                    openai_client=self.openai_client
                )
            else:
                # Fallback to exact string matching
                accuracy = 1.0 if parsed['answer'].strip().lower() == gold_answer.strip().lower() else 0.0
                print("LLM grading didn't work, using 1.0 or 0")
        else:
            accuracy = 0.5  # Fallback
            print("LLM grading didn't work, using 0.5")
        
        # Calculate continuous uncertainty score
        continuous_uncertainty = calculate_continuous_uncertainty(entry, retrieved_docs, entry['question'])
        
        evaluation_result = {
            'entry_id': entry['id'],
            'question': entry['question'],
            'gold_answer': entry['gold_answer'],
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs,
            'confidence': result['confidence'] or 0.5,
            'accuracy': accuracy,
            'prediction_text': parsed['answer'],
            'prediction_explanation': parsed['explanation'],
            'continuous_uncertainty': continuous_uncertainty,
            'is_uncertain': bool(result.get('confidence', 0) < 0.6),
            'model': model_name,
            'tokens_used': result['tokens_used'],
            'set_type': source_set_type,
            'ambiguity_type': entry.get('ambiguity_type', 'conflicting'),
            'log_probs': result.get('log_probs', [])
        }
        
        return evaluation_result, result  # Return both cleaned result and original model response
    
    def evaluate_model(self, model_name: str, max_entries: Optional[int] = None, skip_calibration: bool = False) -> Dict[str, Any]:
        """Evaluate a model on the dataset with two-phase calibration approach."""
        print(f"\nEvaluating {model_name} on BLUFF-RAG dataset...")
        
        if max_entries:
            dataset_subset = self.dataset[:max_entries]
        else:
            dataset_subset = self.dataset
        
        # Store results for both clear and ambiguous sets
        clear_results = []
        ambiguous_results = []
        original_model_responses = []
        successful_evaluations = 0
        
        # Phase 1: Evaluate first 20 entries for calibration (if not skipping)
        if not skip_calibration and len(dataset_subset) >= 20:
            print("Phase 1: Evaluating first 20 entries for calibration...")
            calibration_subset = dataset_subset[:20]
            
            for i, entry in enumerate(tqdm(calibration_subset, desc=f"Calibration phase - {model_name}")):
                try:
                    # Evaluate with clear sources only
                    clear_result, clear_raw = self.evaluate_single_entry(entry, model_name, "clear")
                    if clear_result:
                        clear_results.append(clear_result)
                        original_model_responses.append(clear_raw)
                    
                    # Evaluate with ambiguous sources only
                    ambiguous_result, ambiguous_raw = self.evaluate_single_entry(entry, model_name, "ambiguous")
                    if ambiguous_result:
                        ambiguous_results.append(ambiguous_result)
                        original_model_responses.append(ambiguous_raw)
                    
                    if clear_result and ambiguous_result:
                        successful_evaluations += 1
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error evaluating entry {entry['id']}: {e}")
                    continue
            
            # Create calibration function after collecting 20 entries worth of data
            print(f"\n=== CALIBRATION PHASE ===")
            print(f"Collected {len(original_model_responses)} responses for calibration")
            all_results = clear_results + ambiguous_results
            calibration_success = self.calibrator.update_calibration(all_results, original_model_responses)
            if calibration_success:
                print(f"Calibration function created and frozen!")
                print(f"Calibration function: {self.calibrator.get_calibration_function_description()}")
                print(f"Calibration samples: {self.calibrator.calibration_samples}")
            else:
                print(f"Calibration failed - will use internal confidence")
            
            # Phase 2: Re-run the same 20 entries with frozen calibration
            print(f"\nPhase 2: Re-evaluating first 20 entries with frozen calibration...")
            clear_results = []
            ambiguous_results = []
            original_model_responses = []
            successful_evaluations = 0
            
            for i, entry in enumerate(tqdm(calibration_subset, desc=f"Re-evaluation with calibration - {model_name}")):
                try:
                    # Evaluate with clear sources only
                    clear_result, clear_raw = self.evaluate_single_entry(entry, model_name, "clear")
                    if clear_result:
                        clear_results.append(clear_result)
                        original_model_responses.append(clear_raw)
                    
                    # Evaluate with ambiguous sources only
                    ambiguous_result, ambiguous_raw = self.evaluate_single_entry(entry, model_name, "ambiguous")
                    if ambiguous_result:
                        ambiguous_results.append(ambiguous_result)
                        original_model_responses.append(ambiguous_raw)
                    
                    if clear_result and ambiguous_result:
                        successful_evaluations += 1
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error re-evaluating entry {entry['id']}: {e}")
                    continue
            
            print(f"Phase 2 completed: {successful_evaluations}/{len(calibration_subset)} entries re-evaluated with frozen calibration")
            
            # If we have more entries, continue with the rest using the frozen calibration
            if len(dataset_subset) > 20:
                print(f"\nPhase 3: Evaluating remaining {len(dataset_subset) - 20} entries with frozen calibration...")
                remaining_subset = dataset_subset[20:]
                
                for i, entry in enumerate(tqdm(remaining_subset, desc=f"Remaining entries - {model_name}")):
                    try:
                        # Evaluate with clear sources only
                        clear_result, clear_raw = self.evaluate_single_entry(entry, model_name, "clear")
                        if clear_result:
                            clear_results.append(clear_result)
                            original_model_responses.append(clear_raw)
                        
                        # Evaluate with ambiguous sources only
                        ambiguous_result, ambiguous_raw = self.evaluate_single_entry(entry, model_name, "ambiguous")
                        if ambiguous_result:
                            ambiguous_results.append(ambiguous_result)
                            original_model_responses.append(ambiguous_raw)
                        
                        if clear_result and ambiguous_result:
                            successful_evaluations += 1
                        
                        # Rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error evaluating entry {entry['id']}: {e}")
                        continue
        
        else:
            # Single phase evaluation (either skipping calibration or less than 20 entries)
            print("Single phase evaluation...")
            for i, entry in enumerate(tqdm(dataset_subset, desc=f"Evaluating {model_name}")):
                try:
                    # Evaluate with clear sources only
                    clear_result, clear_raw = self.evaluate_single_entry(entry, model_name, "clear")
                    if clear_result:
                        clear_results.append(clear_result)
                        original_model_responses.append(clear_raw)
                    
                    # Evaluate with ambiguous sources only
                    ambiguous_result, ambiguous_raw = self.evaluate_single_entry(entry, model_name, "ambiguous")
                    if ambiguous_result:
                        ambiguous_results.append(ambiguous_result)
                        original_model_responses.append(ambiguous_raw)
                    
                    if clear_result and ambiguous_result:
                        successful_evaluations += 1
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error evaluating entry {entry['id']}: {e}")
                    continue
        
        print(f"Successfully evaluated {successful_evaluations}/{len(dataset_subset)} entries")
        
        if not clear_results or not ambiguous_results:
            print("No successful evaluations!")
            return {}
        
        # Combine results for standard metrics calculation
        all_results = clear_results + ambiguous_results
        
        # Compute all metrics
        print("Computing BLUFF-RAG metrics...")
        
        # Core BLUFF-RAG metrics
        bluff_rag_metrics = compute_all_bluff_rag_metrics(all_results)
        
        # Calculate faithfulness metrics with individual scores
        faithfulness_batch_metrics, individual_faithfulness_scores = bluff_rag_faithfulness_metrics_with_individuals(all_results)
        
        # Add individual faithfulness scores to results
        for i, result in enumerate(all_results):
            if i < len(individual_faithfulness_scores):
                result['faithfulness'] = individual_faithfulness_scores[i]
        
        # Utility metrics
        utility_metrics = calculate_all_utility_metrics(all_results)
        
        # Calculate ASI metrics
        print("Computing ASI metrics...")
        asi_results = []
        for i in range(min(len(clear_results), len(ambiguous_results))):
            clear_entry = clear_results[i]
            ambiguous_entry = ambiguous_results[i]
            
            if clear_entry['entry_id'] == ambiguous_entry['entry_id']:
                asi_result = calculate_ambiguity_sensitivity_index(clear_entry, ambiguous_entry)
                asi_results.append({
                    'question_id': clear_entry['entry_id'],
                    'asi_score': asi_result['asi'],
                    'asi_components': asi_result
                })
        
        # Calculate batch ASI statistics
        batch_asi = calculate_batch_asi(asi_results)
        
        # Build evaluation summary
        evaluation_summary = {
            'model': model_name,
            'total_entries': len(dataset_subset),
            'successful_evaluations': successful_evaluations,
            'calibration_info': self.calibrator.get_calibration_info(),
            'bluff_rag_metrics': bluff_rag_metrics,
            'faithfulness_metrics': faithfulness_batch_metrics,
            'utility_metrics': utility_metrics,
            'asi_metrics': batch_asi,
            'individual_asi_results': asi_results,
            'summary_results': {
                'clear_results': [
                    {
                        'entry_id': result['entry_id'],
                        'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                        'model_answer': result['prediction_text'],
                        'model_explanation': result['prediction_explanation'][:150] + '...' if len(result['prediction_explanation']) > 150 else result['prediction_explanation'],
                        'gold_answer': result['gold_answer'],
                        'confidence': result['confidence'],
                        'accuracy': result['accuracy'],
                        'is_uncertain': result['is_uncertain'],
                        'set_type': result['set_type'],
                        'faithfulness': result['faithfulness'],
                        'log_probs': result.get('log_probs', []),
                        'retrieved_docs': result.get('retrieved_docs', []),
                        'relevant_docs': result.get('relevant_docs', [])
                    }
                    for result in clear_results
                ],
                'ambiguous_results': [
                    {
                        'entry_id': result['entry_id'],
                        'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                        'model_answer': result['prediction_text'],
                        'model_explanation': result['prediction_explanation'][:150] + '...' if len(result['prediction_explanation']) > 150 else result['prediction_explanation'],
                        'gold_answer': result['gold_answer'],
                        'confidence': result['confidence'],
                        'accuracy': result['accuracy'],
                        'is_uncertain': result['is_uncertain'],
                        'set_type': result['set_type'],
                        'ambiguity_type': result['ambiguity_type'],
                        'faithfulness': result['faithfulness'],
                        'log_probs': result.get('log_probs', []),
                        'retrieved_docs': result.get('retrieved_docs', []),
                        'relevant_docs': result.get('relevant_docs', [])
                    }
                    for result in ambiguous_results
                ]
            }
        }
        
        # Round numeric values and ensure JSON serialization
        evaluation_summary = round_metrics(evaluation_summary, precision=3)
        evaluation_summary = make_json_serializable(evaluation_summary)
        
        # Generate streamlined BLUFF-RAG report
        bluff_rag_report = generate_bluff_rag_report(evaluation_summary)
        
        # Save results
        output_file = os.path.join(self.output_dir, f"{model_name}_evaluation.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        report_file = os.path.join(self.output_dir, f"{model_name}_bluff_rag_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(bluff_rag_report, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Streamlined BLUFF-RAG report saved to {report_file}")
        
        return evaluation_summary
    
    def compare_models(self, model_names: List[str], max_entries: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple models on the dataset."""
        print(f"Comparing {len(model_names)} models on BLUFF-RAG dataset")
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
            
            # Key BLUFF-RAG metrics for comparison
            bluff_rag = result['bluff_rag_metrics']
            faithfulness = result.get('faithfulness_metrics', {})
            comparison_summary['comparison_metrics'][model_name] = {
                'overconfidence_index': bluff_rag.get('overconfidence_index', 'N/A'),
                'hedge_f1': bluff_rag.get('hedge_f1', 'N/A'),
                'expected_calibration_error': bluff_rag.get('expected_calibration_error', 'N/A'),
                'source_quality_confidence_correlation': bluff_rag.get('source_quality_confidence_correlation', 'N/A'),
                # Faithfulness metrics
                'overall_faithfulness': faithfulness.get('overall_faithfulness', 'N/A'),
                'answer_source_overlap': faithfulness.get('answer_source_overlap', 'N/A'),
                'attribution_coverage': faithfulness.get('attribution_coverage', 'N/A'),
                'hallucination_rate': faithfulness.get('hallucination_rate', 'N/A'),
                'grounding_score': faithfulness.get('grounding_score', 'N/A'),
                'factual_consistency': faithfulness.get('factual_consistency', 'N/A')
            }
        
        # Round all numeric values and ensure JSON serialization
        comparison_summary = round_metrics(comparison_summary, precision=3)
        comparison_summary = make_json_serializable(comparison_summary)
        
        # Save comparison
        comparison_file = os.path.join(self.output_dir, "model_comparison.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\nComparison saved to {comparison_file}")
        
        return comparison_summary


def main():
    """Main evaluation function."""
    print("BLUFF-RAG Model Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = RAGModelEvaluator(use_llm_grading=True)
    
    # Try to get all API keys
    print("Loading API keys from environment variables...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    together_api_key = os.getenv("TOGETHER_API_KEY")
    
    # Print which API keys are found
    print("\nAPI Key Status:")
    print(f"  OpenAI API Key: {'Found' if openai_api_key else 'Not found'}")
    print(f"  Anthropic API Key: {'Found' if anthropic_api_key else 'Not found'}")
    print(f"  Google API Key: {'Found' if google_api_key else 'Not found'}")
    print(f"  Mistral API Key: {'Found' if mistral_api_key else 'Not found'}")
    print(f"  Together API Key: {'Found' if together_api_key else 'Not found'}")
    
    if not any([openai_api_key, anthropic_api_key, google_api_key, mistral_api_key, together_api_key]):
        print("\nNo API keys found! Please create a .env file with your API keys.")
        print("Format for .env file:")
        print("OPENAI_API_KEY=your_key_here")
        print("ANTHROPIC_API_KEY=your_key_here")
        print("GOOGLE_API_KEY=your_key_here")
        print("MISTRAL_API_KEY=your_key_here")
        print("TOGETHER_API_KEY=your_key_here")
        return
    
    # Setup models based on available API keys
    models_to_evaluate = []
    
    if openai_api_key:
        print("\nSetting up OpenAI client...")
        evaluator.setup_openai(openai_api_key, "gpt-4o")
        models_to_evaluate.append("gpt-4o")
        print("OpenAI client initialized")
    
    if anthropic_api_key:
        print("\nSetting up Anthropic client...")
        evaluator.setup_anthropic(anthropic_api_key, "claude-3-5-sonnet-20241022")
        models_to_evaluate.append("claude-3-5-sonnet-20241022")
        print("Anthropic client initialized")
    
    if google_api_key:
        print("\nSetting up Google Gemini client...")
        evaluator.setup_google(google_api_key, "gemini-1.5-pro")
        models_to_evaluate.append("gemini-1.5-pro")
        print("Google Gemini client initialized")
    
    if mistral_api_key:
        print("\nSetting up Mistral AI client...")
        evaluator.setup_mistral(mistral_api_key, "mistral-large-latest")
        models_to_evaluate.append("mistral-large-latest")
        print("Mistral AI client initialized")
    
    if together_api_key:
        print("\nSetting up Llama client (via Together AI)...")
        evaluator.setup_llama(together_api_key, "llama-3.1-8b-instruct")
        models_to_evaluate.append("llama-3.1-8b-instruct")
        print("Llama client initialized")
    
    print(f"\nModels to evaluate: {models_to_evaluate}")
    print(f"Total models available: {len(models_to_evaluate)}")
    
    # Filter dataset to only public_health entries
    print("\nFiltering dataset to public_health domain...")
    public_health_entries = [entry for entry in evaluator.dataset if entry['id'].startswith('public_health_')]
    evaluator.dataset = public_health_entries
    print(f"Found {len(public_health_entries)} public_health entries")
    
    # Evaluate with two-phase calibration workflow
    print("\nStarting evaluation with public_health domain...")
    print("Two-phase calibration approach:")
    print("  Phase 1: Evaluate first 20 entries to create calibration function")
    print("  Phase 2: Re-evaluate first 20 entries with frozen calibration")
    print("  Phase 3: Evaluate remaining entries with frozen calibration")
    
    # Helper function for formatting metrics
    def safe_format(value, default='N/A'):
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        else:
            return str(default)
    
    try:
        # Run evaluation for all available models
        if len(models_to_evaluate) == 1:
            # Single model evaluation
            model_name = models_to_evaluate[0]
            print(f"\nEvaluating single model: {model_name}")
            result = evaluator.evaluate_model(model_name, max_entries=50)
            
            if result:
                print("\nEvaluation completed successfully!")
                print(f"Results saved in: {evaluator.output_dir}/")
                
                # Print key findings
                print(f"\nKey Findings:")
                print(f"  Model: {model_name}")
                print(f"  Entries evaluated: {result['successful_evaluations']}/{result['total_entries']}")
                print(f"  Success rate: {result['successful_evaluations']/result['total_entries']:.1%}")
                
                # Print calibration info
                calibration_info = result['calibration_info']
                print(f"\nCalibration Information:")
                print(f"  Was calibrated: {calibration_info['is_calibrated']}")
                print(f"  Calibration samples: {calibration_info['calibration_samples']}")
                print(f"Calibration function: {evaluator.calibrator.get_calibration_function_description()}")

                # Print some key metrics
                bluff_rag = result['bluff_rag_metrics']
                print(f"\nKey BLUFF-RAG Metrics:")
                
                print(f"  Overconfidence Index: {safe_format(bluff_rag.get('overconfidence_index'))}")
                print(f"  Hedge F1: {safe_format(bluff_rag.get('hedge_f1'))}")
                print(f"  Expected Calibration Error: {safe_format(bluff_rag.get('expected_calibration_error'))}")
                
                # Print ASI metrics
                if 'asi_metrics' in result:
                    asi_metrics = result['asi_metrics']
                    print(f"  ASI Score: {asi_metrics.get('mean_asi', 0):.3f}")
                    
                # Print Faithfulness metrics
                faithfulness = result.get('faithfulness_metrics', {})
                print(f"  Overall Faithfulness: {safe_format(faithfulness.get('overall_faithfulness'))}")

            else:
                print("Evaluation failed - no results returned")
        
        else:
            # Multiple model comparison
            print(f"\nComparing {len(models_to_evaluate)} models...")
            comparison_result = evaluator.compare_models(models_to_evaluate, max_entries=50)
            
            if comparison_result:
                print("\nModel comparison completed successfully!")
                print(f"Results saved in: {evaluator.output_dir}/")
                
                # Print comparison summary
                print(f"\nComparison Summary:")
                print(f"  Models evaluated: {comparison_result['models_evaluated']}")
                
                # Print performance comparison
                print(f"\nPerformance Comparison:")
                for model_name, performance in comparison_result['model_performance'].items():
                    print(f"  {model_name}:")
                    print(f"    Success rate: {performance['success_rate']:.1%}")
                    print(f"    Entries evaluated: {performance['successful_evaluations']}/{performance['total_entries']}")
                
                # Print key metrics comparison
                print(f"\nKey Metrics Comparison:")
                for model_name, metrics in comparison_result['comparison_metrics'].items():
                    print(f"  {model_name}:")
                    print(f"    Overconfidence Index: {safe_format(metrics.get('overconfidence_index'))}")
                    print(f"    Hedge F1: {safe_format(metrics.get('hedge_f1'))}")
                    print(f"    Expected Calibration Error: {safe_format(metrics.get('expected_calibration_error'))}")
                    print(f"    Overall Faithfulness: {safe_format(metrics.get('overall_faithfulness'))}")
            else:
                print("Model comparison failed - no results returned")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

