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

CLASSIFICATION TASK: Respond with exactly one letter only.
A = True (correct)
B = False (incorrect)

Answer:"""
    
    system_prompt = "You are a classification system. You judge factual correctness ONLY from provided sources. You must respond with exactly one letter: A (True) or B (False). Do not write sentences or explanations. Consider that sources may be incomplete or ambiguous."
    
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
                temperature=0.3,
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
            # Since Together SDK doesn't provide full probability distributions,
            # we'll use multiple sampling to estimate confidence
            ptrue_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

            # Sample multiple times to estimate probability distribution
            num_samples = 5
            a_count = 0
            b_count = 0
            
            for _ in range(num_samples):
                response = openai_client.chat.completions.create(
                    model=ptrue_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    max_tokens=1,
                    temperature=1,  # Higher temperature for more variation
                    stop=["\n", " ", ".", ":", "!", "?", "To", "The", "This", "That"]
                )
                
                answer = response.choices[0].message.content.strip().upper()
                if answer == "A":
                    a_count += 1
                elif answer == "B":
                    b_count += 1
            
            # Calculate confidence based on sampling
            total_samples = a_count + b_count
            if total_samples > 0:
                confidence = a_count / total_samples
                print(f"DEBUG: Sampled {num_samples} times: A={a_count}, B={b_count}, confidence={confidence}")
            else:
                confidence = 0.5  # Default if no valid responses
                print("DEBUG: No valid A/B responses in sampling")

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
                temperature=0.3,
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