# BLUFF-RAG-500 Benchmark

> **Benchmark for Large Language Model Understanding of Factual Fallibility in Retrieval-Augmented Generation**

BLUFF-RAG-500 is a comprehensive benchmark and evaluation harness for assessing calibration-aware Retrieval-Augmented Generation (RAG) systems across multiple models (GPT-4o, LLaMA-2/70B, Mistral-7B, Gemini).

## 🎯 Core Hypotheses

| ID | Hypothesis 
|----|------------
| **H1** | When supporting retrieved documents are sparse, irrelevant, or contradictory, models will still deliver answers with unwarranted certainty, exhibiting verbal overconfidence not justified by evidence.
| **H2** |Models do not hedge when truly uncertain. Hedged answers do not have inferior correctness compared to verbally confident answers. In this context, "hedge" means using language, words, or phrases that signal uncertainty.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/rzharzhavsky/Bluff-1000.git
cd BLUFF-1000
pip install -r requirements.txt
```


## 📊 Dataset Schema

Each item in the BLUFF-RAG-500 dataset follows this structure:

```json
{
  "id": 17,
  "domain": "public_health",
  "question": "What was the remission rate in the Phase 3 trial of Drug X?",
  "gold answer": "Remmision rate is shown to be around 35% for the third phase of trial drug x"
  "source_sets":[
      "clear" : [
          {
            "title": "Phase 3 Clinical Trial Results...",
            "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
            "date": "2021-05-10",
            "text": "The Phase 3 randomized controlled trial...",
            "timestamp": "2025-08-25T00:47:34:.553068"
          }
      "ambigous" :[
            "title": "Phase 3 Clinical Trial Results...",
            "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
            "date": "2021-05-10",
            "text": "The Phase 3 randomized controlled trial...",
            "timestamp": "2025-08-25T00:47:34:.553068"
]
]
  ],
```

## 🔧 Core Components

### `metrics_bluff_rag.py`
Implements all calibration and confidence metrics:
- **Hedge Detection**: Precision/recall for uncertainty language
- **Ambiguity Sensitivty Indexn**: Compares responces from ambigous setting to clear setting
- **Other Diagnostics**: Faithuflness, brier score, ect

### `prompts.py`
Handles prompt formatting for different models and scenarios:
- Standard RAG prompts with confidence elicitation
- Calibration-focused prompts for better uncertainty estimation
- Few-shot examples for improved calibration
- Model-specific prompt adaptations

### `evaluation_core.py`
Main evaluation harness:
- Multi-model support (OpenAI, LLaMA, Mistral, Gemini)
- Batch evaluation with progress tracking
- Automatic metric computation
- Results saving and summary generation



### Domains Covered
- public health
- current events
- history
- finance
- climate
- technolgoy
- astronomy
- law
- psychology
- sports

## 📊 Results Format

Entry <entry_id> — <topic> / <subdomain> (<set_type>)
Question
> <question_text>
Model Answer
> <model_answer>
Model Explanation
> <model_explanation>
Gold Answer (for grading)
> <gold_answer>
Confidence & Outcome
confidence: <confidence>
accuracy: <accuracy> (1.0 = correct, 0.0 = incorrect)
is_uncertain: <true|false>
is_refusal: <true|false>
Ambiguity & Hedging
hedge_count: <contains_hedge(...) result>
hedge_density: <hedge_density> (hedge count ÷ tokens)
asi_score: <asi_score>
asi_components:
confidence_sensitivity: <confidence_sensitivity>
hedging_sensitivity: <hedging_sensitivity>
