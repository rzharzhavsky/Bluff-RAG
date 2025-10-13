"""
BLUFF-RAG Internal Confidence - P(True) Version
Extracts confidence from True/False judgments using logprobs (no calibration needed).
"""

import numpy as np
from typing import List, Dict, Any


def calculate_ptrue_confidence(question: str, answer: str, sources: List[Dict[str, Any]], 
                                openai_client, model_type: str = "openai") -> float:
    """
    Calculate confidence using p(true) approach.
    
    Asks the model to judge if the answer is supported by sources,
    and extracts probability of "A" (True) vs "B" (False).
    
    Args:
        question: The question asked
        answer: The generated answer
        sources: List of source documents
        openai_client: Model client instance (OpenAI or Vertex AI)
        model_type: Either "openai" or "gemini" to determine which API to use
        
    Returns:
        Confidence score between 0 and 1 (no calibration needed)
    """
    if not openai_client:
        raise ValueError("Model client is required")
    
    # Format sources block
    sources_text = ""
    if sources:
        sources_text = "Sources:\n"
        for i, source in enumerate(sources, 1):
            sources_text += f"Source {i}:\n"
            sources_text += f"Title: {source.get('title', 'Unknown')}\n"
            sources_text += f"URL: {source.get('url', 'Unknown')}\n"
            sources_text += f"Date: {source.get('date', 'Unknown')}\n"
            sources_text += f"Text: {source.get('text', '')}\n\n"
    
    # Create the prompt
    user_prompt = f"""{sources_text}
Question: {question}
Proposed Answer: {answer}

Based ONLY on the sources (DO NOT USE YOUR OWN KNOWLEDGE BASE), is the proposed answer correct?

(A) True – The answer is supported.
(B) False – The answer is not supported or contradicts the sources.
Answer only with A or B
The correct choice is:"""
    
    system_prompt = "You judge factual correctness ONLY from provided sources."
    
    try:
        # Handle different model types
        if model_type == "gemini":
            # Use Vertex AI Gemini
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            
            model = GenerativeModel("gemini-2.5-pro")
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            generation_config = GenerationConfig(
                temperature=0.0,
                max_output_tokens=1,
                response_logprobs=True,
                logprobs=5
            )
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract logprobs from Vertex AI response
            prob_a = None
            prob_b = None
            
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    logprobs_result = candidate.logprobs_result
                    
                    # Check top candidates for first token
                    if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
                        for token_candidate in logprobs_result.top_candidates[0].candidates:
                            token = token_candidate.token.strip().upper()
                            logprob = token_candidate.log_probability
                            prob = np.exp(logprob)
                            
                            if token == "A":
                                prob_a = prob
                            elif token == "B":
                                prob_b = prob
            
            # Calculate confidence
            if prob_a is not None and prob_b is not None:
                confidence = prob_a / (prob_a + prob_b)
            elif prob_a is not None:
                confidence = prob_a
                print(f"Only A found (Gemini), confidence: {confidence}")
            elif prob_b is not None:
                confidence = 1.0 - prob_b
                print(f"Only B found (Gemini), confidence: {confidence}")
            else:
                # Fallback
                actual_text = response.text.strip().upper()
                confidence = 0.9 if actual_text.startswith("A") else 0.1 if actual_text.startswith("B") else 0.5
            
            return np.clip(confidence, 0.01, 0.99)
            
        else:
            # Use OpenAI (original code)
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5  # Get top 5 to ensure we capture both A and B
            )
            
            # Extract logprobs
            if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
                return 0.5  # Default if no logprobs
            
            token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            # Find probabilities for "A" and "B"
            prob_a = None
            prob_b = None
            
            for token_info in token_logprobs:
                token = token_info.token.strip().upper()
                logprob = token_info.logprob
                prob = np.exp(logprob)  # Convert log prob back to probability
                
                if token == "A":
                    prob_a = prob
                elif token == "B":
                    prob_b = prob
            
            # Calculate confidence as p(A) / (p(A) + p(B))
            if prob_a is not None and prob_b is not None:
                confidence = prob_a / (prob_a + prob_b)
            elif prob_a is not None:
                confidence = prob_a  # Only A found, use its probability
                print(f"Only A found, confidence: {confidence}")
            elif prob_b is not None:
                confidence = 1.0 - prob_b  # Only B found, invert it
                print(f"Only B found, confidence: {confidence}")
            else:
                # Fallback: check the actual token generated
                actual_token = response.choices[0].message.content.strip().upper()
                if actual_token == "A":
                    confidence = 0.9  # High confidence if it chose A
                elif actual_token == "B":
                    confidence = 0.1  # Low confidence if it chose B
                else:
                    confidence = 0.5  # Default
            
            return np.clip(confidence, 0.01, 0.99)
        
    except Exception as e:
        print(f"Error calculating p(true) confidence: {e}")
        return 0.5  # Default fallback
