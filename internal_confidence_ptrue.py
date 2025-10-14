"""
BLUFF-RAG Internal Confidence - P(True) Version
Extracts confidence from True/False judgments using logprobs (no calibration needed).
"""

import numpy as np
from typing import List, Dict, Any


def calculate_ptrue_confidence(question: str, answer: str, sources: List[Dict[str, Any]], 
                               openai_client, model_type: str = "openai", model_name: str = None) -> float:
    """
    Calculate confidence using p(true) approach.
    
    Asks the model to judge if the answer is supported by sources,
    and extracts probability of "A" (True) vs "B" (False).
    
    Args:
        question: The question asked
        answer: The generated answer
        sources: List of source documents
        openai_client: Model client instance (OpenAI or Vertex AI)
        model_type: Either "openai", "gemini", or "together" to determine which API to use
        
    Returns:
        Confidence score between 0 and 1 (no calibration needed)
    """
    # Only check for openai_client if using OpenAI model
    if model_type == "openai" and not openai_client:
        raise ValueError("OpenAI client is required for OpenAI models")
    
    # Check for together client if using Together models
    if model_type == "together" and not openai_client:
        raise ValueError("Together client is required for Together models")
    
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

Based ONLY on the sources provided, is the proposed answer correct?

Choose one:
A - True
B - False

Your answer:"""
    
    system_prompt = "You judge factual correctness ONLY from provided sources."
    
    try:
        # Handle different model types
        if model_type == "gemini":
            # Use Vertex AI Gemini - use the model name passed in
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            
            # Extract model name from the model_name parameter
            gemini_model = model_name if model_name else "gemini-2.5-pro"
            model = GenerativeModel(gemini_model)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            generation_config = GenerationConfig(
                temperature=0.0,
                max_output_tokens=150,  # Increased to handle thinking mode (49 tokens) + final answer
                response_logprobs=True,
                logprobs=5,
            )
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Filter out thinking parts and get final answer
            cand = response.candidates[0]
            parts = cand.content.parts
            # Keep only the final text parts (ignore reasoning parts)
            final_texts = [p.text for p in parts if hasattr(p, "text") and not getattr(p, "inline_data", None)]
            answer = final_texts[-1] if final_texts else cand.text
            print("Gemini final answer:", answer)
            
            # Extract logprobs from Vertex AI response
            prob_a = None
            prob_b = None
            
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    logprobs_result = candidate.logprobs_result
                    
                    # Search through ALL output tokens (Gemini 2.5 has thinking, so A/B might not be first token)
                    if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
                        for token_position in logprobs_result.top_candidates:
                            for token_candidate in token_position.candidates:
                                token = token_candidate.token.strip().upper()
                                logprob = token_candidate.log_probability
                                prob = np.exp(logprob)
                                
                                if token == "A" and prob_a is None:  # Take first occurrence
                                    prob_a = prob
                                elif token == "B" and prob_b is None:
                                    prob_b = prob
                                
                                # Stop if we found both
                                if prob_a is not None and prob_b is not None:
                                    break
                            if prob_a is not None and prob_b is not None:
                                break
            
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
                # Fallback: search the filtered answer text for A or B
                actual_text = answer.strip().upper()
                
                # Look for standalone A or B (handling thinking mode output)
                if "\n(A)" in actual_text or actual_text.endswith("A") or "\nA\n" in actual_text or actual_text.endswith("(A)"):
                    confidence = 0.9
                elif "\n(B)" in actual_text or actual_text.endswith("B") or "\nB\n" in actual_text or actual_text.endswith("(B)"):
                    confidence = 0.1
                elif actual_text.startswith("A"):
                    confidence = 0.9
                elif actual_text.startswith("B"):
                    confidence = 0.1
                else:
                    print(f"Warning: Could not find A or B in Gemini filtered answer: {actual_text[:100]}")
                    confidence = 0.5
            
            return np.clip(confidence, 0.01, 0.99)
            
        elif model_type == "together":
        # Use Together SDK (for Mistral and Llama models)
            ptrue_model = "mistralai/Mistral-7B-Instruct-v0.3"

            response = openai_client.chat.completions.create(
                model=ptrue_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens=1,          # single-token decision
                temperature=0.0,
                logprobs=True,
                top_logprobs=50,       # ensure A/B appear
                stop=["\n"]            # avoid leading newline as first token
            )

            # --- Extract logprobs using Together chat schema ---
            def _norm(tok: str) -> str:
                return tok.replace("▁", " ").strip().upper()

            prob_a = None
            prob_b = None

            # Extract logprobs from Together SDK response
            logprobs = response.choices[0].logprobs
            print(f"DEBUG: Logprobs available: {logprobs is not None}")
            if logprobs:
                print(f"DEBUG: Logprobs type: {type(logprobs)}")
                print(f"DEBUG: Has tokens: {hasattr(logprobs, 'tokens')}")
                print(f"DEBUG: Has top_logprobs: {hasattr(logprobs, 'top_logprobs')}")
            
            if logprobs and hasattr(logprobs, 'tokens') and logprobs.tokens is not None and len(logprobs.tokens) > 0:
                print(f"DEBUG: Tokens: {logprobs.tokens}")
                print(f"DEBUG: Token logprobs: {logprobs.token_logprobs}")
                
                # Method 1: Check tokens directly
                if logprobs.token_logprobs is not None:
                    for i, token in enumerate(logprobs.tokens):
                        token_clean = _norm(token)
                        print(f"DEBUG: Processing token '{token}' -> '{token_clean}'")
                        if i < len(logprobs.token_logprobs):
                            logprob = logprobs.token_logprobs[i]
                            prob = float(np.exp(logprob))
                            print(f"DEBUG: Token '{token_clean}' has prob {prob}")
                            if token_clean == "A" and prob_a is None:
                                prob_a = prob
                                print(f"DEBUG: Set prob_a = {prob_a}")
                            elif token_clean == "B" and prob_b is None:
                                prob_b = prob
                                print(f"DEBUG: Set prob_b = {prob_b}")
                else:
                    print("DEBUG: Token logprobs is None")
                
                # Method 2: Check top_logprobs if available (it's a list, not a dict)
                if hasattr(logprobs, 'top_logprobs') and logprobs.top_logprobs is not None and len(logprobs.top_logprobs) > 0:
                    print(f"DEBUG: Top logprobs: {logprobs.top_logprobs}")
                    for i, top_dict in enumerate(logprobs.top_logprobs):
                        print(f"DEBUG: Position {i}: {top_dict}")
                        if isinstance(top_dict, dict):
                            for token, logprob in top_dict.items():
                                token_clean = _norm(token)
                                prob = float(np.exp(logprob))
                                print(f"DEBUG: Top token '{token_clean}' -> prob {prob}")
                                if token_clean == "A" and prob_a is None:
                                    prob_a = prob
                                    print(f"DEBUG: Set prob_a = {prob_a}")
                                elif token_clean == "B" and prob_b is None:
                                    prob_b = prob
                                    print(f"DEBUG: Set prob_b = {prob_b}")
                else:
                    print("DEBUG: No top_logprobs available")
            else:
                print("DEBUG: No tokens available or tokens is None")
            
            # Calculate confidence
            print(f"DEBUG: Found prob_a = {prob_a}, prob_b = {prob_b}")
            if prob_a is not None and prob_b is not None:
                confidence = prob_a / (prob_a + prob_b)
                print(f"DEBUG: Using both probabilities: {confidence}")
            elif prob_a is not None:
                confidence = prob_a
                print(f"DEBUG: Using only A: {confidence}")
            elif prob_b is not None:
                confidence = 1.0 - prob_b
                print(f"DEBUG: Using only B: {confidence}")
            else:
                # Fallback to response content
                actual_response = response.choices[0].message.content.strip().upper()
                print(f"DEBUG: Fallback to response: '{actual_response}'")
                if "B -" in actual_response or actual_response.startswith("B"):
                    confidence = 0.1
                elif "A -" in actual_response or actual_response.startswith("A"):
                    confidence = 0.9
                else:
                    confidence = 0.5
            
            return np.clip(confidence, 0.01, 0.99)

        else:
            # Use OpenAI (original code) - determine model
            ptrue_model = model_name if model_name else "gpt-4o"
            
            response = openai_client.chat.completions.create(
                model=ptrue_model,
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
