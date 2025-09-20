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
        Higher average log probability = higher confidence.
        
        Args:
            log_probs: List of token log probability dictionaries
            
        Returns:
            Confidence score between 0 and 1
        """
        if not log_probs:
            return 0.5
        
        # Convert log probs to probabilities
        probs = [np.exp(log_prob['logprob']) for log_prob in log_probs]
        
        # Calculate average probability (higher = more confident)
        avg_prob = np.mean(probs)
        
        # Better normalization that preserves more variation
        # Use the geometric mean of probabilities for better sensitivity
        if len(probs) > 1:
            # Geometric mean gives more sensitivity to low probabilities
            geom_mean = np.exp(np.mean([np.log(p) for p in probs]))
            confidence = geom_mean
        else:
            confidence = avg_prob
        
        # Scale to 0-1 range with more sensitivity
        confidence = max(0.0, min(1.0, confidence))
        
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
        
        if len(valid_raw_confidences) < 2:
            # Not enough data for calibration, return identity function
            return lambda log_probs: self.calculate_internal_confidence(log_probs)
        
        # Fit isotonic regression: raw_confidence -> actual_accuracy
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(valid_raw_confidences, valid_accuracies)
        
        # Return calibration function
        def calibrate_confidence(log_probs):
            raw_conf = self.calculate_internal_confidence(log_probs)
            if raw_conf is None:
                return 0.5
            return iso_reg.transform([raw_conf])[0]
        
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

