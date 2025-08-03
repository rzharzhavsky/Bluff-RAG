"""
CALM-RAG Runner Module
Main evaluation harness for running RAG models and computing metrics.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import openai
from metrics import compute_all_metrics, retrieval_recall
from prompts import format_prompt, extract_confidence_from_response


class RAGEvaluator:
    """Main class for running CALM-RAG evaluation experiments."""
    
    def __init__(self, dataset_path: str, output_dir: str = "results"):
        """
        Initialize the RAG evaluator.
        
        Args:
            dataset_path: Path to the CALM-RAG dataset JSON file
            output_dir: Directory to save results
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.dataset = self._load_dataset()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model clients
        self.openai_client = None
        self.model_functions = {
            'openai': self.call_model_openai,
            'llama': self.call_model_llama,
            'mistral': self.call_model_mistral,
            'gemini': self.call_model_gemini
        }
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the CALM-RAG dataset from JSON file."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def setup_openai(self, api_key: str, model: str = "gpt-4"):
        """Setup OpenAI client."""
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.openai_model = model
    
    def call_model_openai(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Call OpenAI model with the given prompt.
        
        Args:
            prompt: Formatted prompt string
            temperature: Sampling temperature
        
        Returns:
            Dictionary with response, confidence, and metadata
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Call setup_openai() first.")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            confidence = extract_confidence_from_response(response_text, "openai")
            
            return {
                'response': response_text,
                'confidence': confidence,
                'model': self.openai_model,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'success': True
            }
        
        except Exception as e:
            return {
                'response': '',
                'confidence': None,
                'model': self.openai_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def call_model_llama(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Call LLaMA model (placeholder - requires specific setup).
        
        Args:
            prompt: Formatted prompt string
            temperature: Sampling temperature
        
        Returns:
            Dictionary with response, confidence, and metadata
        """
        # Placeholder implementation
        # In practice, this would use transformers, vLLM, or API calls
        return {
            'response': 'LLaMA model not implemented yet',
            'confidence': None,
            'model': 'llama-2-70b',
            'tokens_used': 0,
            'success': False,
            'error': 'Not implemented'
        }
    
    def call_model_mistral(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Call Mistral model (placeholder - requires specific setup).
        
        Args:
            prompt: Formatted prompt string
            temperature: Sampling temperature
        
        Returns:
            Dictionary with response, confidence, and metadata
        """
        # Placeholder implementation
        return {
            'response': 'Mistral model not implemented yet',
            'confidence': None,
            'model': 'mistral-7b',
            'tokens_used': 0,
            'success': False,
            'error': 'Not implemented'
        }
    
    def call_model_gemini(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Call Gemini model (placeholder - requires specific setup).
        
        Args:
            prompt: Formatted prompt string
            temperature: Sampling temperature
        
        Returns:
            Dictionary with response, confidence, and metadata
        """
        # Placeholder implementation
        return {
            'response': 'Gemini model not implemented yet',
            'confidence': None,
            'model': 'gemini-pro',
            'tokens_used': 0,
            'success': False,
            'error': 'Not implemented'
        }
    
    def evaluate_accuracy(self, prediction: str, gold_answer: str) -> float:
        """
        Evaluate accuracy of prediction against gold answer.
        
        Args:
            prediction: Model's prediction
            gold_answer: Ground truth answer
        
        Returns:
            Binary accuracy score (0 or 1)
        """
        # Simple exact match for now - can be made more sophisticated
        pred_clean = prediction.lower().strip()
        gold_clean = gold_answer.lower().strip()
        
        # Check for exact match or substring match
        if pred_clean == gold_clean:
            return 1.0
        elif gold_clean in pred_clean or pred_clean in gold_clean:
            return 1.0
        else:
            return 0.0
    
    def compute_retrieval_metrics(self, item: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute retrieval-specific metrics for an item.
        
        Args:
            item: Dataset item with source_excerpts
        
        Returns:
            Dictionary of retrieval metrics
        """
        # For now, assume all provided excerpts are "retrieved"
        # In practice, this would compare against a larger corpus
        retrieved_docs = [f"doc_{i}" for i in range(len(item['source_excerpts']))]
        relevant_docs = retrieved_docs  # Simplified assumption
        
        return {
            'retrieval_recall': retrieval_recall(retrieved_docs, relevant_docs),
            'num_retrieved': len(retrieved_docs),
            'num_relevant': len(relevant_docs)
        }
    
    def run_evaluation(self, model_name: str, prompt_type: str = "standard", 
                      max_items: Optional[int] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Run full evaluation on the dataset.
        
        Args:
            model_name: Name of model to evaluate ('openai', 'llama', etc.)
            prompt_type: Type of prompt to use ('standard', 'calibration', 'uncertainty')
            max_items: Maximum number of items to evaluate (None for all)
            temperature: Sampling temperature
        
        Returns:
            Dictionary with results and metrics
        """
        if model_name not in self.model_functions:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_func = self.model_functions[model_name]
        results = []
        
        # Limit dataset if specified
        dataset_subset = self.dataset[:max_items] if max_items else self.dataset
        
        print(f"Running evaluation with {model_name} on {len(dataset_subset)} items...")
        
        for item in tqdm(dataset_subset, desc="Evaluating"):
            # Format prompt
            if prompt_type == "calibration":
                from prompts import format_calibration_prompt
                prompt = format_calibration_prompt(item['question'], item['source_excerpts'], model_name)
            elif prompt_type == "uncertainty":
                from prompts import format_uncertainty_prompt
                prompt = format_uncertainty_prompt(item['question'], item['source_excerpts'])
            else:
                prompt = format_prompt(item['question'], item['source_excerpts'], model_name)
            
            # Call model
            model_result = model_func(prompt, temperature)
            
            if model_result['success']:
                # Evaluate accuracy
                accuracy = self.evaluate_accuracy(model_result['response'], item['gold_answer'])
                
                # Compute retrieval metrics
                retrieval_metrics = self.compute_retrieval_metrics(item)
                
                # Store result
                result = {
                    'id': item['id'],
                    'domain': item['domain'],
                    'question': item['question'],
                    'gold_answer': item['gold_answer'],
                    'prediction': model_result['response'],
                    'confidence': model_result['confidence'] or 0.5,  # Default if not provided
                    'accuracy': accuracy,
                    'human_confidence': item.get('human_confidence', 0.5),
                    'human_hedge_label': item.get('human_hedge_label', ''),
                    'prediction_text': model_result['response'],
                    'is_uncertain': item.get('human_confidence', 0.5) < 0.7,  # Threshold for uncertainty
                    'tokens_used': model_result['tokens_used'],
                    'model': model_result['model'],
                    **retrieval_metrics
                }
                
                results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Compute aggregate metrics
        metrics = compute_all_metrics(results)
        
        # Prepare final results
        evaluation_results = {
            'model_name': model_name,
            'prompt_type': prompt_type,
            'temperature': temperature,
            'num_items': len(results),
            'num_successful': len([r for r in results if r['accuracy'] is not None]),
            'metrics': metrics,
            'individual_results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary from run_evaluation
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"calm_rag_results_{results['model_name']}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        metrics = results['metrics']
        
        print(f"\n=== CALM-RAG Evaluation Summary ===")
        print(f"Model: {results['model_name']}")
        print(f"Items evaluated: {results['num_items']}")
        print(f"Successful evaluations: {results['num_successful']}")
        print(f"\n--- Key Metrics ---")
        print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
        print(f"Overconfidence Index: {metrics['overconfidence_index']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Confidence-Accuracy Correlation: {metrics['confidence_accuracy_correlation']:.4f}")
        print(f"Hedge Precision: {metrics['hedge_precision']:.4f}")
        print(f"Hedge Recall: {metrics['hedge_recall']:.4f}")
        print(f"ECE after Isotonic Calibration: {metrics['ece_after_isotonic']:.4f}")


def main():
    """Example usage of the CALM-RAG evaluator."""
    # Initialize evaluator
    evaluator = RAGEvaluator("example_dataset.json")
    
    # Setup OpenAI (requires API key)
    # evaluator.setup_openai("your-api-key-here")
    
    # Run evaluation
    # results = evaluator.run_evaluation("openai", max_items=10)
    
    # Print summary and save results
    # evaluator.print_summary(results)
    # evaluator.save_results(results)
    
    print("CALM-RAG evaluator initialized. Set up your API keys and run evaluation.")


if __name__ == "__main__":
    main()
