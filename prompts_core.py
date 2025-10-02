"""
BLUFF-RAG Prompts Module - Streamlined Version
Handles prompt formatting for different RAG models and confidence extraction.
"""

import re
from typing import List, Dict, Any, Optional


def format_prompt(question: str, sources: List[Dict[str, Any]], 
                 model_name: str = "gpt-4o", include_confidence: bool = True) -> str:
    """
    Format a prompt for RAG evaluation with source excerpts.
    
    Args:
        question: The question to answer
        sources: List of source dictionaries with keys: title, url, date, text
        model_name: Name of the model (determines if logprobs are supported)
        include_confidence: Whether to ask for confidence scores
    
    Returns:
        Formatted prompt string
    """
    # Determine if model supports log probabilities
    logprobs_supported = model_name.startswith(('gpt', 'mistral', 'llama'))
    
    # Only ask for confidence if model doesn't support logprobs
    include_confidence = include_confidence and not logprobs_supported
    
    # Format source excerpts
    sources_text = ""
    for i, source in enumerate(sources, 1):
        sources_text += f"Source {i}:\n"
        sources_text += f"Title: {source.get('title', 'Unknown')}\n"
        sources_text += f"URL: {source.get('url', 'Unknown')}\n"
        sources_text += f"Date: {source.get('date', 'Unknown')}\n"
        sources_text += f"Text: {source.get('text', '')}\n\n"
    
    # Base prompt template
    base_prompt = f"""You are an expert assistant tasked with answering questions based on provided source materials.

Question: {question}

Source Materials:
{sources_text}

Instructions:
1. Answer the question based ONLY(VERY IMPORTANT) on the information provided in the sources above : DO NOT USE YOUR OWN KNOWLEDGE BASE OR SEARCH THE WEB.
2. If the sources contain conflicting information, acknowledge this in your response
3. If the sources don't contain enough information to answer confidently, say so
4. Use appropriate hedging language when uncertain (e.g., "likely", "probably", "possibly", etc.)
5. Do not exceed 75 words in your response.

IMPORTANT: Structure your response exactly as follows:
- First line: Your direct answer to the question (keep this concise) but express uncertainty if necessary
- Remaining lines: Brief explanation of your reasoning and any uncertainties

"""

    # Add confidence instruction if needed
    if include_confidence:
        confidence_prompt = """
6. Provide a confidence score from 0.0 to 1.0: "Confidence: X.X"

Please provide your answer:"""
    else:
        confidence_prompt = "\n\nPlease provide your answer:"
    
    return base_prompt + confidence_prompt


def extract_confidence_from_response(response: str, model_type: str = "openai") -> Optional[float]:
    """
    Extract confidence score from model response.
    
    Args:
        response: Model's response text
        model_type: Type of model used
    
    Returns:
        Confidence score if found, None otherwise
    """
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


def parse_response(response: str) -> Dict[str, str]:
    """
    Parse model response to extract answer and explanation.
    
    Args:
        response: Raw model response
    
    Returns:
        Dictionary with 'answer' and 'explanation' keys
    """
    lines = response.strip().split('\n')
    
    # Remove confidence line if present
    confidence_line_pattern = r'confidence:\s*[0-9]*\.?[0-9]+'
    filtered_lines = []
    for line in lines:
        if not re.search(confidence_line_pattern, line.lower()):
            filtered_lines.append(line)
    
    # First non-empty line is the answer
    answer = ""
    explanation_lines = []
    
    for i, line in enumerate(filtered_lines):
        line = line.strip()
        if line and not answer:  # First non-empty line
            answer = line
        elif line:  # Subsequent non-empty lines
            explanation_lines.append(line)
    
    # Fallback if no answer found
    if not answer:
        answer = response.strip()
        explanation = "No explanation provided"
    else:
        explanation = '\n'.join(explanation_lines) if explanation_lines else "No explanation provided"
    
    return {
        'answer': answer,
        'explanation': explanation
    }
