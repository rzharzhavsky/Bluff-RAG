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
    bluff_rag_faithfulness_metrics, bluff_rag_faithfulness_metrics_with_individuals,
    is_refusal_response
)
from internal_confidence_ptrue import calculate_ptrue_confidence

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
            'expected_calibration_error': bluff_rag_metrics.get('expected_calibration_error', 0.0),
            'confidence_accuracy_correlation': bluff_rag_metrics.get('confidence_accuracy_correlation', 0.0),
            'ASI': asi_metrics.get('mean_asi', 0.0)
        },
        
        # HYPOTHESIS 2: Hedging Behavior & Refusal
        'h2_hedging_behavior': {
            'VUI': bluff_rag_metrics.get('hedge_f1', 0.0),
            'source_set_on_hedging': bluff_rag_metrics.get('source_set_on_hedging', 0.0),
            'lexical_overconfidence_index': bluff_rag_metrics.get('lexical_overconfidence_index', 0.0),
            'hedge_precision': bluff_rag_metrics.get('hedge_precision', 0.0),
            'hedge_recall': bluff_rag_metrics.get('hedge_recall', 0.0),
            'refusal_count': bluff_rag_metrics.get('total_refusals', 0.0),
            'refusal_sensitivity': bluff_rag_metrics.get('refusal_sensitivity', 0.0)
        },
        
        # DIAGNOSTIC METRICS
        'diagnostics': {
            'answer_correctness': answer_correctness,
            'missed_refusals': evaluation_summary.get('missed_refusals', 0),
            'brier_score': bluff_rag_metrics.get('brier_score', 0.0),
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
    
    def setup_google(self, project_id: str = None, location: str = "us-east1", model: str = "gemini-2.5-pro"):
        """Setup Google Vertex AI Gemini client with logprob support."""
        try:
            from google.cloud import aiplatform
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            # Initialize Vertex AI
            # Authentication will use GOOGLE_APPLICATION_CREDENTIALS env var or default credentials
            vertexai.init(project=project_id or os.getenv('GCP_PROJECT_ID', 'bluff-474923'), 
                         location=location)
            
            self.google_client = vertexai
            self.google_model_name = model
            self.google_location = location
            print(f"Vertex AI initialized: project={project_id}, location={location}")
        except ImportError:
            print("Vertex AI client not available. Install with: pip install google-cloud-aiplatform")
        except Exception as e:
            print(f"Error setting up Vertex AI: {e}")
    
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
            
            return {
                'response': response_text,
                'confidence': None,  # Will be calculated using p(true) in evaluate_single_entry
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
        """Call Google Vertex AI Gemini model (no logprobs needed for answer generation)."""
        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            
            # Create model instance
            model = GenerativeModel(self.google_model_name)
            
            # Configure generation (NO logprobs for answer generation)
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000
            )
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text
            
            return {
                'response': response_text,
                'confidence': None,  # Will be calculated via p(true)
                'log_probs': [],
                'model': self.google_model_name,
                'tokens_used': 0,
                'success': True
            }
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'model': self.google_model_name if hasattr(self, 'google_model_name') else 'gemini-2.5-pro',
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
            
            return {
                'response': response_text,
                'confidence': None,  # Will be calculated using p(true) in evaluate_single_entry
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
            
            return {
                'response': response_text,
                'confidence': None,  # Will be calculated using p(true) in evaluate_single_entry
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
        
        # Calculate confidence using p(true) approach
        try:
            # Determine model type for p(true) calculation
            model_type = "gemini" if model_name.startswith('gemini') else "openai"
            
            confidence = calculate_ptrue_confidence(
                question=entry['question'],
                answer=parsed['answer'],
                sources=sources,
                openai_client=self.openai_client,  # Will be used for OpenAI models
                model_type=model_type
            )
            # Store p(true) confidence for analysis
            ptrue_confidence = confidence
        except Exception as e:
            print(f"Error calculating p(true) confidence: {e}. Using fallback.")
            confidence = result.get('confidence') or 0.5
            ptrue_confidence = None
        
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
        
        # Check if the model refused to answer
        is_refusal = is_refusal_response(parsed['answer'])
        
        # Calculate accuracy using LLM grading (skip if refused)
        gold_answer = entry.get('gold_answer', '')
        if is_refusal:
            # Reward refusal with 0.4 - better than being wrong (<0.2) but worse than being right (>0.8)
            # This incentivizes the model to refuse when uncertain
            accuracy = 0.4
        elif gold_answer and parsed['answer']:
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
            'confidence': confidence,
            'accuracy': accuracy,
            'prediction_text': parsed['answer'],
            'prediction_explanation': parsed['explanation'],
            'continuous_uncertainty': continuous_uncertainty,
            'is_uncertain': bool(confidence < 0.6),
            'is_refusal': is_refusal,
            'model': model_name,
            'tokens_used': result['tokens_used'],
            'set_type': source_set_type,
            'ambiguity_type': entry.get('ambiguity_type', 'conflicting'),
            'log_probs': result.get('log_probs', [])
        }
        
        return evaluation_result, result  # Return both cleaned result and original model response
    
    def evaluate_model(self, model_name: str, max_entries: Optional[int] = None, skip_calibration: bool = False) -> Dict[str, Any]:
        """Evaluate a model on the dataset using p(true) confidence (no calibration needed)."""
        print(f"\nEvaluating {model_name} on BLUFF-RAG dataset with p(true) confidence...")
        
        if max_entries:
            dataset_subset = self.dataset[:max_entries]
        else:
            dataset_subset = self.dataset
        
        # Store results for both clear and ambiguous sets
        clear_results = []
        ambiguous_results = []
        original_model_responses = []
        successful_evaluations = 0
        
        # Single phase evaluation (p(true) doesn't need calibration)
        print("Evaluating dataset with p(true) confidence calculation...")
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
                time.sleep(0.1)
                
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
        skipped_refusals = 0
        for i in range(min(len(clear_results), len(ambiguous_results))):
            clear_entry = clear_results[i]
            ambiguous_entry = ambiguous_results[i]
            
            if clear_entry['entry_id'] == ambiguous_entry['entry_id']:
                asi_result = calculate_ambiguity_sensitivity_index(clear_entry, ambiguous_entry)
                # Skip if either response was a refusal (asi_result will be None)
                if asi_result is not None:
                    asi_results.append({
                        'question_id': clear_entry['entry_id'],
                        'asi_score': asi_result['asi'],
                        'asi_components': asi_result
                    })
                else:
                    skipped_refusals += 1
        
        print(f"ASI calculated for {len(asi_results)} question pairs ({skipped_refusals} pairs skipped due to refusals)")
        
        # Calculate batch ASI statistics
        batch_asi = calculate_batch_asi(asi_results)
        
        # Calculate missed refusals - questions where model answered incorrectly when it should have refused
        missed_refusals = 0
        for result in all_results:
            # Count cases where accuracy < 0.2 (very wrong) and model didn't refuse
            accuracy = result.get('accuracy')
            if accuracy is not None and accuracy < 0.2 and not result.get('is_refusal', False):
                missed_refusals += 1
        
        print(f"Missed refusals: {missed_refusals} (questions with accuracy < 0.2 that weren't refused)")
        
        # === P(TRUE) DIAGNOSTICS ===
        print(f"\n{'='*60}")
        print("P(TRUE) CONFIDENCE DIAGNOSTICS")
        print(f"{'='*60}")
        
        # 1. Raw correlation: confidence vs accuracy (before any calibration)
        confidences = [r['confidence'] for r in all_results if r.get('confidence') is not None and r.get('accuracy') is not None]
        accuracies_for_corr = [r['accuracy'] for r in all_results if r.get('confidence') is not None and r.get('accuracy') is not None]
        
        raw_corr = None
        p_value = None
        if len(confidences) > 2:
            from scipy.stats import pearsonr
            raw_corr, p_value = pearsonr(confidences, accuracies_for_corr)
            print(f"\n1. RAW CONFIDENCE-ACCURACY CORRELATION:")
            print(f"   Pearson r = {raw_corr:.3f} (p = {p_value:.4f})")
            print(f"   {'✓ Good correlation!' if raw_corr > 0.3 else '✗ Weak correlation'}")
        
        # 2. Confidence distribution
        if confidences:
            import numpy as np
            print(f"\n2. CONFIDENCE DISTRIBUTION:")
            print(f"   Min:    {min(confidences):.3f}")
            print(f"   25th:   {np.percentile(confidences, 25):.3f}")
            print(f"   Median: {np.median(confidences):.3f}")
            print(f"   75th:   {np.percentile(confidences, 75):.3f}")
            print(f"   Max:    {max(confidences):.3f}")
            print(f"   Std:    {np.std(confidences):.3f}")
            print(f"   Range:  {max(confidences) - min(confidences):.3f}")
        
        # 3. Confidence by source quality (clear vs ambiguous)
        clear_confidences = [r['confidence'] for r in clear_results if r.get('confidence') is not None]
        ambiguous_confidences = [r['confidence'] for r in ambiguous_results if r.get('confidence') is not None]
        
        if clear_confidences and ambiguous_confidences:
            print(f"\n3. CONFIDENCE BY SOURCE TYPE:")
            print(f"   Clear sources:     {np.mean(clear_confidences):.3f} ± {np.std(clear_confidences):.3f}")
            print(f"   Ambiguous sources: {np.mean(ambiguous_confidences):.3f} ± {np.std(ambiguous_confidences):.3f}")
            diff = np.mean(clear_confidences) - np.mean(ambiguous_confidences)
            print(f"   Difference:        {diff:.3f} ({'✓ Good!' if diff > 0.05 else '✗ Small difference'})")
            print(f"   → Model {'does' if diff > 0.05 else 'does NOT'} recognize poor source quality")
        
        # 4. Calibration quality (how close confidence is to actual accuracy)
        mean_abs_error = None
        if len(confidences) > 0:
            calibration_errors = [abs(c - a) for c, a in zip(confidences, accuracies_for_corr)]
            mean_abs_error = np.mean(calibration_errors)
            print(f"\n4. CALIBRATION QUALITY:")
            print(f"   Mean Absolute Error: {mean_abs_error:.3f}")
            print(f"   {'✓ Well calibrated!' if mean_abs_error < 0.2 else '✗ Needs calibration' if mean_abs_error < 0.3 else '✗ Poorly calibrated'}")
        
        # 5. Show a few examples
        print(f"\n5. EXAMPLE P(TRUE) JUDGMENTS (first 3):")
        for i, result in enumerate(all_results[:3]):
            print(f"\n   Example {i+1}:")
            print(f"   Question: {result['question'][:80]}...")
            print(f"   Answer: {result['prediction_text'][:80]}...")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Actual Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"   Source Type: {result['set_type']}")
        
        print(f"\n{'='*60}\n")
        
        # Build evaluation summary with p(true) diagnostics
        ptrue_diagnostics = {
            'raw_confidence_accuracy_correlation': float(raw_corr) if raw_corr is not None else None,
            'correlation_p_value': float(p_value) if p_value is not None else None,
            'confidence_distribution': {
                'min': float(min(confidences)) if confidences else None,
                'percentile_25': float(np.percentile(confidences, 25)) if confidences else None,
                'median': float(np.median(confidences)) if confidences else None,
                'percentile_75': float(np.percentile(confidences, 75)) if confidences else None,
                'max': float(max(confidences)) if confidences else None,
                'std': float(np.std(confidences)) if confidences else None,
                'range': float(max(confidences) - min(confidences)) if confidences else None
            },
            'confidence_by_source_type': {
                'clear_mean': float(np.mean(clear_confidences)) if clear_confidences else None,
                'clear_std': float(np.std(clear_confidences)) if clear_confidences else None,
                'ambiguous_mean': float(np.mean(ambiguous_confidences)) if ambiguous_confidences else None,
                'ambiguous_std': float(np.std(ambiguous_confidences)) if ambiguous_confidences else None,
                'difference': float(np.mean(clear_confidences) - np.mean(ambiguous_confidences)) if (clear_confidences and ambiguous_confidences) else None
            },
            'calibration_quality': {
                'mean_absolute_error': float(mean_abs_error) if mean_abs_error is not None else None
            }
        }
        
        evaluation_summary = {
            'model': model_name,
            'total_entries': len(dataset_subset),
            'successful_evaluations': successful_evaluations,
            'confidence_method': 'p(true)',
            'ptrue_diagnostics': ptrue_diagnostics,
            'missed_refusals': missed_refusals,
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
                        'is_refusal': result.get('is_refusal', False),
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
                        'is_refusal': result.get('is_refusal', False),
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
    
    # Setup OpenAI for LLM grading (already initialized in __init__)
    # GPT-4o evaluation is commented out - only evaluating Gemini
    if openai_api_key:
        print("\nOpenAI client available for LLM grading")
        # Uncomment below to evaluate GPT-4o:
        # evaluator.setup_openai(openai_api_key, "gpt-4o")
        # models_to_evaluate.append("gpt-4o")
    
    # Anthropic evaluation commented out
    # if anthropic_api_key:
    #     print("\nSetting up Anthropic client...")
    #     evaluator.setup_anthropic(anthropic_api_key, "claude-3-5-sonnet-20241022")
    #     models_to_evaluate.append("claude-3-5-sonnet-20241022")
    #     print("Anthropic client initialized")
    
    if google_api_key:
        print("\nSetting up Google Vertex AI client...")
        # Set service account credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/ron/MyProjects/Bluff-RAG/gcp_service_account.json'
        evaluator.setup_google(project_id="bluff-474923", location="us-east1", model="gemini-2.5-pro")
        models_to_evaluate.append("gemini-2.5-pro")
        print("Vertex AI Gemini 2.5 Pro client initialized")
    
    # Mistral evaluation commented out
    # if mistral_api_key:
    #     print("\nSetting up Mistral client...")
    #     evaluator.setup_mistral(mistral_api_key, "mistral-large-latest")
    #     models_to_evaluate.append("mistral-large-latest")
    #     print("Mistral client initialized")
    
    # Llama evaluation commented out
    # if together_api_key:
    #     print("\nSetting up Llama client...")
    #     evaluator.setup_llama(together_api_key, "llama-3.1-8b-instruct")
    #     models_to_evaluate.append("llama-3.1-8b-instruct")
    #     print("Llama client initialized")
    
    if not models_to_evaluate:
        print("\nNo API keys found! Please create a .env file with your API keys.")
        print("Format for .env file:")
        print("OPENAI_API_KEY=your_key_here")
        print("ANTHROPIC_API_KEY=your_key_here")
        print("GOOGLE_API_KEY=your_key_here")
        print("MISTRAL_API_KEY=your_key_here")
        print("TOGETHER_API_KEY=your_key_here")
        return
    
    print(f"\nModels to evaluate: {models_to_evaluate}")
    
    # Use the whole dataset for evaluation
    print(f"\nUsing full dataset with {len(evaluator.dataset)} entries")
    
    # Evaluate with p(true) confidence (no calibration needed)
    print("\nStarting evaluation with full dataset...")
    print("Using p(true) confidence approach:")
    print("  - Each answer is evaluated for correctness based on sources")
    print("  - Confidence = p(A) / (p(A) + p(B)) from True/False judgment")
    print("  - No calibration needed - probabilities are already well-calibrated")
    
    # Evaluate all models
    all_results = {}
    
    for model_name in models_to_evaluate:
        try:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}...")
            print(f"{'='*60}")
            
            result = evaluator.evaluate_model(model_name, max_entries=None)
            
            if result:
                all_results[model_name] = result
                print(f"\n{model_name} evaluation completed successfully!")
                
                # Print key findings for this model
                print(f"\nKey Findings for {model_name}:")
                print(f"  Entries evaluated: {result['successful_evaluations']}/{result['total_entries']}")
                print(f"  Success rate: {result['successful_evaluations']/result['total_entries']:.1%}")
                
                # Print confidence method info
                print(f"\nConfidence Method: {result.get('confidence_method', 'p(true)')}")

                # Print some key metrics
                bluff_rag = result['bluff_rag_metrics']
                print(f"\nKey BLUFF-RAG Metrics:")
                
                def safe_format(value, default='N/A'):
                    if isinstance(value, (int, float)):
                        return f"{value:.3f}"
                    else:
                        return str(default)
                
                print(f"  Overconfidence Index: {safe_format(bluff_rag.get('overconfidence_index'))}")
                print(f"  Hedge F1: {safe_format(bluff_rag.get('hedge_f1'))}")
                print(f"  Expected Calibration Error: {safe_format(bluff_rag.get('expected_calibration_error'))}")
                print(f"  Missed Refusals: {result.get('missed_refusals', 0)} (very wrong answers that weren't refused)")
                
                # Print ASI metrics
                if 'asi_metrics' in result:
                    asi_metrics = result['asi_metrics']
                    print(f"  ASI Score: {asi_metrics.get('mean_asi', 0):.3f}")
                    
                # Print Faithfulness metrics
                faithfulness = result.get('faithfulness_metrics', {})
                print(f"  Overall Faithfulness: {safe_format(faithfulness.get('overall_faithfulness'))}")

            else:
                print(f"{model_name} evaluation failed - no results returned")
        
        except Exception as e:
            print(f"{model_name} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison if we have multiple models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Generating model comparison...")
        print(f"{'='*60}")
        
        try:
            comparison_result = evaluator.compare_models(list(all_results.keys()))
            print("Model comparison completed successfully!")
        except Exception as e:
            print(f"Model comparison failed: {e}")
    
    print(f"\nAll evaluations completed!")
    print(f"Results saved in: {evaluator.output_dir}/")
    print(f"Successfully evaluated {len(all_results)} out of {len(models_to_evaluate)} models")


if __name__ == "__main__":
    main()

