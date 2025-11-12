# BLUFF-RAG-500 Benchmark

> **Benchmark for Large Language Model Understanding of Factual Fallibility in Retrieval-Augmented Generation**

BLUFF-RAG-500 is a comprehensive benchmark and evaluation harness for assessing calibration-aware Retrieval-Augmented Generation (RAG) systems across multiple models (GPT-4o, LLaMA-2/70B, Mistral-7B, Gemini).

## 🎯 Core Hypotheses

| ID | Hypothesis | Key Metrics |
|----|------------|-------------|
| **H1** | When supporting retrieved documents are sparse, irrelevant, or contradictory, models will
still deliver answers with unwarranted certainty, exhibiting verbal overconfidence not justified by
evidence.
| **H2** |Models do not hedge when truly uncertain. Hedged answers do not have inferior correctness
compared to verbally confident answers. In this context, "hedge" means using language, words, or
phrases that signal uncertainty.

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
  "gold answer": "Remmision rate is shown to be around 35% for the third phase of trial drug x
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

Sample evaluation:
```json
Model: Mistral-7B,
Question: How has the methodology for estimating CO2 emissions from the cement industry
in China evolved to improve accuracy in recent years?
Model Response: The methodology for estimating CO2 emissions from the cement industry
in China has evolved to improve accuracy by incorporating artificial intelligence-based neural
networks, specifically Long Short-Term Memory (LSTM) neural networks.
Explanation: This approach is aimed at handling the nonlinear relationships and complex dy-
namics inherent in emission datasets(Source 1), thereby providing more accurate predictions.
In contrast, traditional methods such as statistical analyses and empirical models may not
fully capture the long-term dependencies and dynamic changes in CO2 emissions influenced
by factors like energy consumption patterns, production processes, and policy interventions.
Gold Answer: The methodology evolved by using provincial clinker emission factors from
multiple official statistics sources, following the Intergovernmental Panel on Climate Change
(IPCC) methodology, to create a more accurate and high-quality panel dataset for estimating
CO2 emissions.
Source Information:
• Source 1: journals.plos.org (reliable) - Estimation of carbon dioxide emissions from
the cement industry in Beijing-Tianjin-Hebei using neural networks
• Source 2: pubmed.ncbi.nlm.nih.gov (reliable) - Empirical assessing cement CO2
emissions based on China’s economic and social development during 2001-2030
• Source 3: cdc.gov (distraction) - There are millions of U.S. workers exposed to
chemicals in their work each year
• Source 4: pmc.ncbi.nlm.nih.gov (distraction) - In many industrial sectors, workers
are exposed to manufactured or unintentionally emitted airborne nanoparticles (NPs)
• Source 5: reddit.com (unreliable) - It’s so ovbious that recent studies show 0
improvement in accuracy or data granularity when estimating CO2 emissions from
cement production in China compared to the methods used in the early 2000s.
```


