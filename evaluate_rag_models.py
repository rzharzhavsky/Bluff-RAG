#!/usr/bin/env python3
"""
Comprehensive RAG Model Evaluation Script for CALM-RAG Dataset
Tests various models (GPT, Claude, etc.) and computes all metrics for comparison.
"""

import json
import os
import time
import openai
from typing import List, Dict, Any, Optional
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
    calculate_all_enhanced_metrics,
    calm_rag_h1_metrics,
    calm_rag_h2_metrics,
    calm_rag_h3_metrics,
    calm_rag_h4_metrics,
    calm_rag_h5_metrics,
    expected_calibration_error,
    calculate_continuous_uncertainty
)
from prompts import format_prompt, extract_confidence_from_response

class RAGModelEvaluator:
    """Evaluates RAG models on the CALM-RAG dataset."""
    
    def __init__(self, dataset_path: str = "calmrag_dataset.json", output_dir: str = "evaluation_results"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.dataset = self._load_dataset()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.mistral_client = None
        self.llama_client = None
        
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
        
        # Normalize to 0-1 range (rough calibration)
        # This is a simple approach - will be refined by isotonic regression
        confidence = min(1.0, avg_prob * 2)
        
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
    
    def update_calibration(self, evaluation_results: List[Dict[str, Any]]):
        """
        Update calibration function using evaluation results.
        """
        log_probs_list = []
        accuracies = []
        
        for result in evaluation_results:
            if result and 'log_probs' in result and result['log_probs']:
                log_probs_list.append(result['log_probs'])
                accuracies.append(result.get('accuracy', 0.5))
        
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
                    log_probs.append({
                        'token': token_info.token,
                        'logprob': token_info.logprob,
                        'top_logprobs': token_info.top_logprobs
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
                    log_probs.append({
                        'token': token_info.token,
                        'logprob': token_info.logprob,
                        'top_logprobs': token_info.top_logprobs
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
        
        if not result['success']:
            return None
        
        # Create evaluation result structure
        retrieved_docs = [{'url': s['url'], 'domain': s['domain'], 'category': s['category']} for s in all_sources]
        relevant_docs = [s['url'] for s in entry['source_sets']['clear']]
        
        # Calculate continuous uncertainty score based on multiple factors
        continuous_uncertainty = calculate_continuous_uncertainty(
            entry, retrieved_docs, entry['question']
        )
        
        # Mock accuracy (in real evaluation, you'd compare with gold answer)
        # For now, we'll use a simple heuristic based on response quality
        accuracy = 0.8 if result['confidence'] and result['confidence'] > 0.7 else 0.5
        
        evaluation_result = {
            'entry_id': entry['id'],
            'question': entry['question'],
            'gold_answer': entry['gold_answer'],
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs,
            'confidence': result['confidence'] or 0.5,
            'accuracy': accuracy,
            'prediction_text': result['response'],
            'log_probs': result.get('log_probs', []),  # Store log probabilities
            'continuous_uncertainty': continuous_uncertainty,  # New continuous uncertainty score
            'is_uncertain': result['confidence'] < 0.6 if result['confidence'] else True,
            'human_confidence': entry.get('human_confidence'),
            'no_retrieval_confidence': 0.5,  # Mock value
            'no_retrieval_accuracy': 0.3,    # Mock value
            'model': model_name,
            'tokens_used': result['tokens_used']
        }
        
        return evaluation_result
    
    def evaluate_model(self, model_name: str, max_entries: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a model on the dataset."""
        print(f"\nEvaluating {model_name} on CALM-RAG dataset...")
        
        if max_entries:
            dataset_subset = self.dataset[:max_entries]
        else:
            dataset_subset = self.dataset
        
        results = []
        successful_evaluations = 0
        
        for entry in tqdm(dataset_subset, desc=f"Evaluating {model_name}"):
            try:
                result = self.evaluate_single_entry(entry, model_name)
                if result:
                    results.append(result)
                    successful_evaluations += 1
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error evaluating entry {entry['id']}: {e}")
                continue
        
        print(f"Successfully evaluated {successful_evaluations}/{len(dataset_subset)} entries")
        
        if not results:
            print("No successful evaluations!")
            return {}
        
        # Update calibration after collecting initial results
        if len(results) >= 10:
            print(f"Updating calibration with {len(results)} samples...")
            self.update_calibration(results)
            
            # Re-evaluate with calibrated confidence if calibration was successful
            if self.is_calibrated:
                print("Re-evaluating with calibrated confidence...")
                calibrated_results = []
                for entry in tqdm(dataset_subset, desc=f"Re-evaluating {model_name} with calibration"):
                    try:
                        result = self.evaluate_single_entry(entry, model_name)
                        if result:
                            calibrated_results.append(result)
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        print(f"Error re-evaluating entry {entry['id']}: {e}")
                        continue
                
                if calibrated_results:
                    results = calibrated_results
                    print(f"Re-evaluation completed with {len(calibrated_results)} samples")
        
        # Compute all metrics
        print("Computing CALM-RAG metrics...")
        
        # Core CALM-RAG metrics
        calm_rag_metrics = compute_all_calm_rag_metrics(results)
        
        # Enhanced metrics
        enhanced_metrics = calculate_all_enhanced_metrics(results)
        
        # Individual hypothesis metrics
        h1_metrics = calm_rag_h1_metrics(results)
        h2_metrics = calm_rag_h2_metrics(results)
        h3_metrics = calm_rag_h3_metrics(results)
        h4_metrics = calm_rag_h4_metrics(results)
        h5_metrics = calm_rag_h5_metrics(results)
        
        
        evaluation_summary = {
            'model': model_name,
            'total_entries': len(dataset_subset),
            'successful_evaluations': successful_evaluations,
            'calm_rag_metrics': calm_rag_metrics,
            'enhanced_metrics': enhanced_metrics,
            'hypothesis_metrics': {
                'h1_overconfidence': h1_metrics,
                'h2_calibration_difference': h2_metrics,
                'h3_hedging_language': h3_metrics,
                'h4_self_assessment': h4_metrics,
                'h5_source_quality': h5_metrics
            },
            'raw_results': results
        }
        
        # Save results
        output_file = os.path.join(self.output_dir, f"{model_name}_evaluation.json")
        with open(output_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
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
    
    # Initialize evaluator
    evaluator = RAGModelEvaluator()
    
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
    
    # Evaluate models (start with small subset for testing)
    print("\nStarting evaluation with first 10 entries for testing...")
    
    try:
        comparison = evaluator.compare_models(models_to_evaluate, max_entries=10)
        
        if comparison:
            print("\nEvaluation completed successfully!")
            print(f"Results saved in: {evaluator.output_dir}/")
            
            # Print key findings
            print("\nKey Findings:")
            for model_name, performance in comparison['model_performance'].items():
                print(f"  {model_name}: {performance['successful_evaluations']}/{performance['total_entries']} entries evaluated")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
