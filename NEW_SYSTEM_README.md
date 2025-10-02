# BLUFF-RAG Evaluation System - Streamlined Version

## Overview

This is a complete rewrite of the BLUFF-RAG evaluation system, designed to be more maintainable, debuggable, and easier to understand. The new system maintains all the functionality of the original while being significantly more organized and streamlined.

## New File Structure

The original three files (`metrics.py`, `evaluate_rag_models.py`, `prompts.py`) have been reorganized into four focused modules:

### 1. `evaluation_core.py` - Main Orchestrator
- **RAGModelEvaluator class**: Main evaluation orchestrator
- **Model integration**: Support for GPT, Claude, Gemini, Mistral, Llama
- **Calibration workflow**: Two-phase evaluation with calibration
- **Report generation**: Comprehensive evaluation reports

### 2. `metrics_bluff_rag.py` - All Metrics Implementation
- **BLUFF-RAG Hypotheses H1-H5**: All hypothesis-specific metrics
- **Core metrics**: ECE, Brier score, confidence-accuracy correlation
- **Utility metrics**: Retrieval quality, source quality, diversity
- **ASI metrics**: Ambiguity Sensitivity Index calculations

### 3. `prompts_core.py` - Prompt Management
- **Prompt templates**: Specialized prompts for different scenarios
- **Response parsing**: Extract answers and explanations
- **Confidence extraction**: Parse confidence scores from responses
- **Model-specific handling**: Different prompts for logprob vs non-logprob models

### 4. `calibration.py` - Calibration System
- **ConfidenceCalibrator class**: Isotonic regression calibration
- **Log probability handling**: Convert logprobs to confidence scores
- **Calibration workflow**: Two-phase evaluation approach

## Key Improvements

### 1. **Modularity**
- Each module has a single, clear responsibility
- No more massive files with hundreds of functions
- Easy to debug and modify individual components

### 2. **Simplified Calibration**
- Clean separation of calibration logic
- Uses isotonic regression as requested
- Clear two-phase workflow: calibrate on first 20 entries, then evaluate all

### 3. **Better Error Handling**
- Graceful handling of API failures
- Clear error messages and fallbacks
- Robust confidence extraction

### 4. **Streamlined Metrics**
- All BLUFF-RAG hypothesis metrics in one place
- Consistent naming and documentation
- Reduced complexity while maintaining functionality

### 5. **Improved Debugging**
- Clear separation of concerns
- Easier to trace issues
- Better logging and status reporting

## Usage

### Basic Evaluation

```python
from evaluation_core import RAGModelEvaluator

# Initialize evaluator
evaluator = RAGModelEvaluator(
    dataset_path="bluffrag_dataset.json",
    output_dir="evaluation_results",
    use_soft_accuracy=True
)

# Setup model APIs
evaluator.setup_openai(api_key="your_key", model="gpt-4o")

# Evaluate model
results = evaluator.evaluate_model("gpt-4o", max_entries=50)
```

### Model Comparison

```python
# Compare multiple models
models = ["gpt-4o", "claude-3-5-sonnet-20241022"]
comparison = evaluator.compare_models(models, max_entries=20)
```

### Individual Module Usage

```python
# Use metrics directly
from metrics_bluff_rag import compute_all_bluff_rag_metrics
metrics = compute_all_bluff_rag_metrics(evaluation_results)

# Use calibration directly
from calibration import ConfidenceCalibrator
calibrator = ConfidenceCalibrator()
confidence = calibrator.get_calibrated_confidence(log_probs)

# Use prompts directly
from prompts_core import format_prompt, parse_response
prompt = format_prompt(question, sources, "gpt-4o")
parsed = parse_response(model_response)
```

## Changes from Original System

### 1. **File Organization**
- **Before**: 3 large files (metrics.py: 3677 lines, evaluate_rag_models.py: 1463 lines)
- **After**: 4 focused modules with clear responsibilities

### 2. **Calibration System**
- **Before**: Complex calibration logic mixed with evaluation
- **After**: Clean ConfidenceCalibrator class with isotonic regression

### 3. **Metrics Implementation**
- **Before**: Functions scattered throughout massive metrics.py
- **After**: All BLUFF-RAG hypothesis metrics organized by hypothesis

### 4. **Model Integration**
- **Before**: Model calling logic mixed with evaluation logic
- **After**: Clean model interface methods in RAGModelEvaluator

### 5. **Error Handling**
- **Before**: Inconsistent error handling throughout
- **After**: Robust error handling with clear fallbacks

## Maintained Functionality

All original functionality has been preserved:

✅ **All 5 BLUFF-RAG hypotheses** (H1-H5) with full metrics  
✅ **Isotonic regression calibration** using log probabilities  
✅ **Multi-model support** (GPT, Claude, Gemini, Mistral, Llama)  
✅ **Two-phase evaluation** (calibrate on 20 entries, then evaluate all)  
✅ **Comprehensive reporting** with BLUFF-RAG scores  
✅ **Soft accuracy calculation** for nuanced evaluation  
✅ **ASI (Ambiguity Sensitivity Index)** calculations  
✅ **Source quality and diversity** metrics  
✅ **Faithfulness and grounding** metrics  

## New Features

### 1. **Better Confidence Derivation**
- Improved log probability to confidence conversion
- More sophisticated calibration using geometric mean
- Better handling of edge cases

### 2. **Enhanced Response Parsing**
- More robust answer/explanation separation
- Better handling of confidence statements
- Improved text cleaning

### 3. **Streamlined Reports**
- Focused BLUFF-RAG reports with key metrics
- Better metric descriptions and interpretations
- Cleaner JSON output

## Migration Guide

### For Existing Code

If you have existing code that imports from the old modules:

```python
# Old imports
from metrics import compute_all_bluff_rag_metrics
from evaluate_rag_models import RAGModelEvaluator
from prompts import format_prompt

# New imports
from metrics_bluff_rag import compute_all_bluff_rag_metrics
from evaluation_core import RAGModelEvaluator
from prompts_core import format_prompt
```

### API Compatibility

The main APIs remain largely the same:

- `RAGModelEvaluator` class interface is preserved
- `compute_all_bluff_rag_metrics()` function signature is the same
- `format_prompt()` function interface is maintained
- All metric names and outputs are consistent

## Testing

The new system has been tested to ensure:

- ✅ All imports work correctly
- ✅ All modules integrate properly
- ✅ Metrics calculations are consistent
- ✅ Calibration system functions correctly
- ✅ Model evaluation workflow works end-to-end

## Performance Improvements

- **Faster imports**: Modular structure reduces import time
- **Better memory usage**: Cleaner data structures
- **Improved debugging**: Easier to isolate and fix issues
- **Maintainable code**: Much easier to understand and modify

## Next Steps

The new system is ready for production use. You can:

1. **Replace the old files** with the new modules
2. **Update any existing scripts** to use the new imports
3. **Run evaluations** using the streamlined interface
4. **Extend functionality** by modifying individual modules

The system maintains full backward compatibility while providing a much cleaner and more maintainable foundation for future development.

