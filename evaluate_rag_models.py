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

from metrics import (
    compute_all_calm_rag_metrics,
    calculate_all_enhanced_metrics,
    calm_rag_h1_metrics,
    calm_rag_h2_metrics,
    calm_rag_h3_metrics,
    calm_rag_h4_metrics,
    calm_rag_h5_metrics
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
    
    def create_rag_prompt(self, question: str, sources: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt with the question and retrieved sources."""
        prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided sources.

Question: {question}

Sources:
"""
        
        for i, source in enumerate(sources, 1):
            prompt += f"{i}. {source.get('title', 'No title')}\n"
            prompt += f"   URL: {source['url']}\n"
            prompt += f"   Content: {source.get('text', '')[:500]}...\n\n"
        
        prompt += """Please provide a comprehensive answer based on the sources above. If the sources don't contain enough information to answer the question confidently, please indicate this.

Answer:"""
        
        return prompt
    
    def call_openai_model(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Call OpenAI model."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000
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
                'confidence': confidence,
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
    
    def evaluate_single_entry(self, entry: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Evaluate a single dataset entry with the specified model."""
        # Create RAG prompt
        all_sources = entry['source_sets']['clear'] + entry['source_sets']['ambiguous']
        prompt = self.create_rag_prompt(entry['question'], all_sources)
        
        # Call model
        if model_name.startswith('gpt'):
            result = self.call_openai_model(prompt)
        elif model_name.startswith('claude'):
            result = self.call_anthropic_model(prompt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if not result['success']:
            return None
        
        # Create evaluation result structure
        retrieved_docs = [{'url': s['url'], 'domain': s['domain'], 'category': s['category']} for s in all_sources]
        relevant_docs = [s['url'] for s in entry['source_sets']['clear']]
        
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
