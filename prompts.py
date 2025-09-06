"""
CALM-RAG Prompts Module
Handles prompt formatting for different RAG models and scenarios.
"""

from typing import List, Dict, Any, Optional


def format_prompt(question: str, excerpts: List[Dict[str, Any]], 
                 model_type: str = "openai", include_confidence: bool = True) -> str:
    """
    Format a prompt for RAG evaluation with source excerpts.
    
    Args:
        question: The question to answer
        excerpts: List of source excerpt dictionaries with keys:
                 - title, url, date, text
        model_type: Type of model ("openai", "llama", "mistral", "gemini")
        include_confidence: Whether to ask for confidence scores
    
    Returns:
        Formatted prompt string
    """
    # Format source excerpts
    sources_text = ""
    for i, excerpt in enumerate(excerpts, 1):
        sources_text += f"Source {i}:\n"
        sources_text += f"Title: {excerpt.get('title', 'Unknown')}\n"
        sources_text += f"URL: {excerpt.get('url', 'Unknown')}\n"
        sources_text += f"Date: {excerpt.get('date', 'Unknown')}\n"
        sources_text += f"Text: {excerpt.get('text', '')}\n\n"
    
    # Base prompt template
    base_prompt = f"""You are an expert assistant tasked with answering questions based on provided source materials.

Question: {question}

Source Materials:
{sources_text}

Instructions:
1. Answer the question based ONLY on the information provided in the sources above
2. If the sources contain conflicting information, acknowledge this in your response
3. If the sources don't contain enough information to answer confidently, say so
4. Use appropriate hedging language when uncertain (e.g., "likely", "probably", "possibly", ect)
5. Do not exceed 75 words in your response.
"""

    # Add confidence instruction based on model type
    if include_confidence:
        if model_type == "gemini":
            confidence_prompt = """
6. Include a confidence score (0.0 to 1.0) at the end: "Confidence: X.X"

Please answer:"""
        
        else:
            confidence_prompt = """
6. Provide a confidence score from 0.0 to 1.0: "Confidence: X.X"

Answer:"""
    else:
        confidence_prompt = "\n\nPlease provide your answer:"
    
    return base_prompt + confidence_prompt


def format_calibration_prompt(question: str, excerpts: List[Dict[str, Any]], 
                            model_type: str = "openai") -> str:
    """
    Format a prompt specifically designed to elicit calibrated responses.
    
    Args:
        question: The question to answer
        excerpts: List of source excerpt dictionaries
        model_type: Type of model
    
    Returns:
        Calibration-focused prompt string
    """
    sources_text = ""
    for i, excerpt in enumerate(excerpts, 1):
        sources_text += f"Source {i}:\n"
        sources_text += f"Title: {excerpt.get('title', 'Unknown')}\n"
        sources_text += f"Date: {excerpt.get('date', 'Unknown')}\n"
        sources_text += f"Text: {excerpt.get('text', '')}\n\n"
    
    calibration_prompt = f"""You are an expert assistant. Your goal is to provide accurate answers while being well-calibrated about your uncertainty.

Question: {question}

Available Sources:
{sources_text}

Instructions:
1. Carefully analyze the provided sources
2. If sources are contradictory, incomplete, or unclear, acknowledge this uncertainty
3. Use appropriate hedging language when uncertain (e.g., "likely", "appears to", "suggests")
4. Be precise about what you know vs. what you're inferring
5. Provide a confidence score (0.0-1.0) that reflects your actual certainty
6. Consider the quality, recency, and consistency of sources in your confidence assessment

Format your response as:
Answer: [Your answer here]
Reasoning: [Brief explanation of your reasoning and any uncertainties]
Confidence: [0.0-1.0]

Please respond:"""
    
    return calibration_prompt


def format_uncertainty_prompt(question: str, excerpts: List[Dict[str, Any]]) -> str:
    """
    Format a prompt that explicitly asks about uncertainty and hedge terms.
    
    Args:
        question: The question to answer
        excerpts: List of source excerpt dictionaries
    
    Returns:
        Uncertainty-focused prompt string
    """
    sources_text = ""
    for i, excerpt in enumerate(excerpts, 1):
        sources_text += f"Source {i} ({excerpt.get('date', 'Unknown')}):\n"
        sources_text += f"{excerpt.get('text', '')}\n\n"
    
    uncertainty_prompt = f"""Question: {question}

Sources:
{sources_text}

Please answer this question and explicitly indicate your level of certainty. Use appropriate uncertainty language when the evidence is:
- Contradictory: "The sources present conflicting information..."
- Incomplete: "Based on limited evidence..."
- Unclear: "The available information suggests..."
- Outdated: "According to sources from [date]..."

Provide:
1. Your answer with appropriate hedging language
2. A brief explanation of why you're certain or uncertain
3. A confidence score (0.0 = completely uncertain, 1.0 = completely certain)

Response:"""
    
    return uncertainty_prompt


def extract_confidence_from_response(response: str, model_type: str = "openai") -> Optional[float]:
    """
    Extract confidence score from model response.
    
    Args:
        response: Model's response text
        model_type: Type of model used
    
    Returns:
        Confidence score if found, None otherwise
    """
    import re
    
    # Look for confidence patterns
    patterns = [
        r"Confidence:\s*([0-9]*\.?[0-9]+)",
        r"confidence:\s*([0-9]*\.?[0-9]+)",
        r"Confidence\s*=\s*([0-9]*\.?[0-9]+)",
        r"confidence\s*=\s*([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*confidence",
        r"confidence\s*(?:score|level)?\s*(?:is|of)?\s*([0-9]*\.?[0-9]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                confidence = float(match.group(1))
                # Ensure confidence is in [0, 1] range
                if confidence > 1.0:
                    confidence = confidence / 100.0  # Convert percentage
                return max(0.0, min(1.0, confidence))
            except ValueError:
                continue
    
    return None


def format_few_shot_prompt(question: str, excerpts: List[Dict[str, Any]], 
                          examples: List[Dict[str, Any]], model_type: str = "openai") -> str:
    """
    Format a few-shot prompt with examples for better calibration.
    
    Args:
        question: The question to answer
        excerpts: List of source excerpt dictionaries
        examples: List of example Q&A pairs with confidence scores
        model_type: Type of model
    
    Returns:
        Few-shot prompt string
    """
    # Format examples
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Question: {example['question']}\n"
        examples_text += f"Sources: {example['sources_summary']}\n"
        examples_text += f"Answer: {example['answer']}\n"
        examples_text += f"Confidence: {example['confidence']}\n\n"
    
    # Format current sources
    sources_text = ""
    for i, excerpt in enumerate(excerpts, 1):
        sources_text += f"Source {i}: {excerpt.get('text', '')}\n"
    
    few_shot_prompt = f"""Here are examples of how to answer questions with appropriate confidence levels:

{examples_text}Now please answer this question following the same format:

Question: {question}
Sources: {sources_text}

Answer: """
    
    return few_shot_prompt
