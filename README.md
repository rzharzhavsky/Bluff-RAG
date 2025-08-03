# CALM-RAG-500 Benchmark

> **Calibration-Aware Language Model Retrieval-Augmented Generation Benchmark**

CALM-RAG-500 is a comprehensive benchmark and evaluation harness for assessing calibration-aware Retrieval-Augmented Generation (RAG) systems across multiple models (GPT-4o, LLaMA-2/70B, Mistral-7B, Gemini).

## 🎯 Core Hypotheses

| ID | Hypothesis | Key Metrics |
|----|------------|-------------|
| **H1** | Sparse/contradictory evidence → verbal over-confidence | Retrieval-Recall vs Confidence ρ, Overconfidence Index (OCI) |
| **H2** | Adding retrieval ↑ accuracy **but** ↑ calibration error | ECE (with-/without-RAG), Brier Score |
| **H3** | **Hedging** frequency aligns with uncertainty | Hedge Precision/Recall, Lexical Overconfidence Index |
| **H4** | Self-reported probs correlate w/ truth but need scaling | Corr(p, y) (ρ), ECE ↓ after isotonic |
| **H5** | Calibration-tuned language ↓ user acceptance of wrong answers | Human Acceptance of Wrong, Hedge Alignment Score |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/CALM-RAG500.git
cd CALM-RAG500
pip install -r requirements.txt
```

### Basic Usage

```python
from runner import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator("example_dataset.json")

# Setup OpenAI (replace with your API key)
evaluator.setup_openai("your-openai-api-key")

# Run evaluation
results = evaluator.run_evaluation("openai", max_items=10)

# Print summary and save results
evaluator.print_summary(results)
evaluator.save_results(results)
```

## 📊 Dataset Schema

Each item in the CALM-RAG-500 dataset follows this structure:

```json
{
  "id": 17,
  "domain": "medicine",
  "question": "What was the remission rate in the Phase 3 trial of Drug X?",
  "source_excerpts": [
    {
      "title": "Phase 3 Clinical Trial Results...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
      "date": "2021-05-10",
      "text": "The Phase 3 randomized controlled trial..."
    }
  ],
  "gold_answer": "45%",
  "human_confidence": 0.6,
  "human_hedge_label": "Likely"
}
```

## 🔧 Core Components

### `metrics.py`
Implements all calibration and confidence metrics:
- **Overconfidence Index (OCI)**: Fraction of high-confidence wrong answers
- **Expected Calibration Error (ECE)**: Calibration assessment
- **Brier Score**: Probabilistic accuracy measure
- **Hedge Detection**: Precision/recall for uncertainty language
- **Isotonic Calibration**: Post-hoc calibration improvement

### `prompts.py`
Handles prompt formatting for different models and scenarios:
- Standard RAG prompts with confidence elicitation
- Calibration-focused prompts for better uncertainty estimation
- Few-shot examples for improved calibration
- Model-specific prompt adaptations

### `runner.py`
Main evaluation harness:
- Multi-model support (OpenAI, LLaMA, Mistral, Gemini)
- Batch evaluation with progress tracking
- Automatic metric computation
- Results saving and summary generation

## 📈 Key Metrics

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Brier Score**: Combines accuracy and calibration
- **Overconfidence Index**: High-confidence errors (τ = 0.8)

### Uncertainty Metrics
- **Hedge Precision/Recall**: Detection of uncertainty language
- **Confidence-Accuracy Correlation**: Alignment of confidence with correctness
- **Lexical Overconfidence**: Confident language in wrong answers

### Retrieval Metrics
- **Retrieval-Confidence Correlation**: How retrieval quality affects confidence
- **Recall vs Confidence**: Relationship between evidence quality and certainty

## 🎛️ Evaluation Modes

### Standard Evaluation
```python
results = evaluator.run_evaluation("openai", prompt_type="standard")
```

### Calibration-Focused
```python
results = evaluator.run_evaluation("openai", prompt_type="calibration")
```

### Uncertainty-Aware
```python
results = evaluator.run_evaluation("openai", prompt_type="uncertainty")
```

## 📋 Reference Datasets

CALM-RAG-500 builds upon several established benchmarks:
- **RAGBench (2023)**: Factuality/robustness
- **PubMedQA (2019)**: Medical QA with confidence
- **MedMCQA (2022)**: Multiple choice with explanations
- **CalibRAG (2024)**: Synthetic calibration triples
- **CLIMATEX (2023)**: Expert confidence levels

## 🔬 Experimental Setup

### Models Supported
- **GPT-4o**
- **LLaMA-2/70B**
- **Mistral-7B** 
- **Gemini**

### Domains Covered
- Medicine
- Climate Science
- Technology
- General Knowledge
- Politics

## 📊 Results Format

Evaluation results include:
```json
{
  "model_name": "openai",
  "num_items": 100,
  "metrics": {
    "expected_calibration_error": 0.15,
    "overconfidence_index": 0.23,
    "brier_score": 0.18,
    "confidence_accuracy_correlation": 0.67
  },
  "individual_results": [...]
}
```


## 📚 Citation

If you use CALM-RAG-500 in your research, please cite:

```bibtex
@misc{calm-rag-500,
  title={CALM-RAG-500: A Calibration-Aware Benchmark for Retrieval-Augmented Generation},
  author={Ron, Emma, Daniel, and Ahan},
  year={2024},
  url={https://github.com/Ahanmr/CALM-RAG500}
}
