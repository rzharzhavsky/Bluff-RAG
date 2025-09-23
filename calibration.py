"""
CALM-RAG Calibration Module
Handles isotonic regression calibration for confidence scores derived from log probabilities.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import List, Dict, Any, Optional, Callable


class ConfidenceCalibrator:
    """Handles calibration of confidence scores using isotonic regression."""
    
    def __init__(self):
        self.calibration_function: Optional[Callable] = None
        self.is_calibrated = False
        self.calibration_samples = 0
    
    def calculate_internal_confidence(self, log_probs: List[Dict]) -> float:
        """
        Calculate internal confidence from token log probabilities.
        Uses multiple factors to create more varied confidence scores.
        
        Args:
            log_probs: List of token log probability dictionaries
            
        Returns:
            Confidence score between 0 and 1
        """
        if not log_probs:
            return 0.5
        
        # Convert log probs to probabilities
        probs = [np.exp(log_prob['logprob']) for log_prob in log_probs]
        
        if len(probs) == 0:
            return 0.5
        
        # Multiple confidence signals for better variation
        
        # 1. Geometric mean (sensitive to low probabilities)
        geom_mean = np.exp(np.mean([np.log(max(p, 1e-10)) for p in probs]))
        
        # 2. Minimum probability (worst case)
        min_prob = min(probs)
        
        # 3. Standard deviation (uncertainty measure - lower std = higher confidence)
        prob_std = np.std(probs)
        consistency_factor = max(0, 1 - prob_std * 2)  # Scale std to 0-1 range
        
        # 4. Entropy-based measure
        # Higher entropy = more uncertainty = lower confidence
        entropy = -np.sum([p * np.log(max(p, 1e-10)) for p in probs])
        max_entropy = np.log(len(probs))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        entropy_confidence = 1 - normalized_entropy
        
        # Combine multiple signals with weights
        confidence = (
            0.4 * geom_mean +           # Main signal: geometric mean
            0.2 * min_prob +            # Worst case scenario
            0.2 * consistency_factor +  # Consistency across tokens
            0.2 * entropy_confidence    # Information-theoretic measure
        )
        
        # Apply a more aggressive scaling to increase variation
        # Map from typical range [0.3, 0.9] to [0.1, 0.9]
        confidence = 0.1 + 0.8 * confidence
        
        # Ensure bounds
        confidence = max(0.05, min(0.95, confidence))
        
        return confidence
    
    def calibrate_log_probs_to_confidence(self, log_probs_list: List[List[Dict]], 
                                         accuracies: List[float]) -> Callable:
        """
        Use isotonic regression to calibrate log probs to confidence scores.
        
        Args:
            log_probs_list: List of log probability sequences
            accuracies: Corresponding accuracy scores
            
        Returns:
            Calibration function that takes log_probs and returns calibrated confidence
        """
        # Calculate raw confidence scores from log probs
        raw_confidences = [self.calculate_internal_confidence(lp) for lp in log_probs_list]
        
        # Filter out None values
        valid_indices = [i for i, conf in enumerate(raw_confidences) if conf is not None]
        valid_raw_confidences = [raw_confidences[i] for i in valid_indices]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        
        if len(valid_raw_confidences) < 5:  # Need more data for reliable calibration
            # Not enough data for calibration, return identity function
            print(f"Warning: Only {len(valid_raw_confidences)} samples for calibration, using raw confidence")
            return lambda log_probs: self.calculate_internal_confidence(log_probs)
        
        # Check if we have enough variation in raw confidences
        confidence_range = max(valid_raw_confidences) - min(valid_raw_confidences)
        if confidence_range < 0.1:  # Very little variation
            print(f"Warning: Low confidence variation ({confidence_range:.3f}), calibration may be unreliable")
        
        # Use a more sophisticated calibration approach that preserves variation
        
        # 1. First, try isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(valid_raw_confidences, valid_accuracies)
        
        # 2. Check if isotonic regression creates too much flattening
        iso_predictions = iso_reg.predict(valid_raw_confidences)
        iso_variation = np.std(iso_predictions)
        accuracy_variation = np.std(valid_accuracies)
        
        print(f"Calibration mapping: confidence range [{min(valid_raw_confidences):.3f}, {max(valid_raw_confidences):.3f}] -> accuracy range [{min(valid_accuracies):.3f}, {max(valid_accuracies):.3f}]")
        print(f"Isotonic variation: {iso_variation:.3f}, Target variation: {accuracy_variation:.3f}")
        
        # If isotonic regression preserves too little variation, use linear scaling
        if iso_variation < 0.05:  # Very low variation
            print("Using linear scaling instead of isotonic regression (preserves more variation)")
            
            # Linear scaling: map raw confidence range to accuracy range
            raw_min, raw_max = min(valid_raw_confidences), max(valid_raw_confidences)
            acc_min, acc_max = min(valid_accuracies), max(valid_accuracies)
            
            def calibrate_confidence(log_probs):
                raw_conf = self.calculate_internal_confidence(log_probs)
                if raw_conf is None:
                    return 0.5
                
                # Linear scaling
                if raw_max == raw_min:
                    return acc_min
                
                # Scale and shift
                normalized = (raw_conf - raw_min) / (raw_max - raw_min)
                calibrated_conf = acc_min + normalized * (acc_max - acc_min)
                return max(0.0, min(1.0, calibrated_conf))
            
            return calibrate_confidence
        
        else:
            # Use isotonic regression as it preserves good variation
            def calibrate_confidence(log_probs):
                raw_conf = self.calculate_internal_confidence(log_probs)
                if raw_conf is None:
                    return 0.5
                
                # Apply isotonic regression (use predict, not transform)
                calibrated_conf = iso_reg.predict([raw_conf])[0]
                return max(0.0, min(1.0, calibrated_conf))
            
            return calibrate_confidence
    
    def update_calibration(self, evaluation_results: List[Dict[str, Any]], 
                          raw_results: List[Dict[str, Any]]) -> bool:
        """
        Update calibration function using evaluation results.
        
        Args:
            evaluation_results: List of evaluation results with accuracy scores
            raw_results: List of raw model responses with log_probs
            
        Returns:
            True if calibration was successful, False otherwise
        """
        log_probs_list = []
        accuracies = []
        
        # Extract log_probs and accuracies for calibration
        for i, result in enumerate(raw_results):
            if result and 'log_probs' in result and result['log_probs']:
                log_probs_list.append(result['log_probs'])
                # Get accuracy from evaluation_results
                if i < len(evaluation_results):
                    accuracies.append(evaluation_results[i].get('accuracy', 0.5))
                else:
                    accuracies.append(0.5)
        
        if len(log_probs_list) >= 10:  # Need minimum data for calibration
            self.calibration_function = self.calibrate_log_probs_to_confidence(
                log_probs_list, accuracies
            )
            self.is_calibrated = True
            self.calibration_samples = len(log_probs_list)
            return True
        else:
            self.is_calibrated = False
            self.calibration_samples = len(log_probs_list)
            return False
    
    def get_calibrated_confidence(self, log_probs: List[Dict]) -> float:
        """
        Get calibrated confidence score using the calibration function.
        
        Args:
            log_probs: List of token log probability dictionaries
            
        Returns:
            Calibrated confidence score
        """
        if self.calibration_function and self.is_calibrated:
            try:
                return self.calibration_function(log_probs)
            except Exception:
                return self.calculate_internal_confidence(log_probs)
        else:
            return self.calculate_internal_confidence(log_probs)
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get information about the current calibration state.
        
        Returns:
            Dictionary with calibration information
        """
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_samples': self.calibration_samples,
            'has_calibration_function': self.calibration_function is not None
        }
    
    def get_calibration_function_description(self) -> str:
        """
        Get a human-readable description of the calibration function.
        
        Returns:
            String description of the calibration function
        """
        if not self.calibration_function or not self.is_calibrated:
            return "No calibration function available"
        
        # Try to extract function parameters by inspecting the function
        try:
            # Get the function's source code or closure variables
            import inspect
            
            # Check if it's a lambda or nested function
            if hasattr(self.calibration_function, '__code__'):
                # Try to get closure variables
                if self.calibration_function.__closure__:
                    closure_vars = {}
                    for i, cell in enumerate(self.calibration_function.__closure__):
                        var_name = self.calibration_function.__code__.co_freevars[i]
                        closure_vars[var_name] = cell.cell_contents
                    
                    # Check if it's linear scaling
                    if 'raw_min' in closure_vars and 'raw_max' in closure_vars:
                        raw_min = closure_vars['raw_min']
                        raw_max = closure_vars['raw_max']
                        acc_min = closure_vars.get('acc_min', 0.0)
                        acc_max = closure_vars.get('acc_max', 1.0)
                        return f"Linear scaling: raw_conf [{raw_min:.3f}, {raw_max:.3f}] -> calibrated_conf [{acc_min:.3f}, {acc_max:.3f}]"
                    
                    # Check if it's isotonic regression
                    elif 'iso_reg' in closure_vars:
                        iso_reg = closure_vars['iso_reg']
                        return f"Isotonic regression with {len(iso_reg.X_) if hasattr(iso_reg, 'X_') else 'unknown'} calibration points"
            
            # Fallback to function name
            return f"Calibration function: {self.calibration_function.__name__ if hasattr(self.calibration_function, '__name__') else 'lambda'}"
            
        except Exception as e:
            return f"Calibration function: {type(self.calibration_function).__name__} (details unavailable)"

