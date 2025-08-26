"""
Concept Alignment Module for ACB-Agent

This module implements the concept alignment procedure from Section 3.3 of the paper,
which aligns learned concepts with human-interpretable labels through correlation analysis
and statistical alignment scoring.

Key Components:
- ConceptAlignmentTracker: Manages concept-human label alignment over time
- compute_concept_alignment: Core alignment computation algorithm
- alignment_based_regularization: Regularization loss based on alignment scores
- human_feedback_integration: Integration of human feedback for concept refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict, deque
from scipy.stats import pearsonr, spearmanr
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptAlignmentTracker:
    """
    Tracks and manages concept-human label alignment over training.
    
    This class implements the core alignment procedure from Section 3.3,
    maintaining alignment scores, correlation statistics, and alignment history.
    """
    
    def __init__(
        self,
        concept_dim: int,
        alignment_threshold: float = 0.6,
        min_samples: int = 100,
        history_length: int = 1000,
        correlation_method: str = 'pearson',
        concept_names: Optional[List[str]] = None
    ):
        """
        Initialize concept alignment tracker.
        
        Args:
            concept_dim: Number of concept dimensions
            alignment_threshold: Minimum correlation for concept alignment (τ_align = 0.6)
            min_samples: Minimum samples needed for reliable alignment computation
            history_length: Maximum history length for alignment tracking
            correlation_method: Correlation method ('pearson' or 'spearman')
            concept_names: Optional concept names for interpretability
        """
        self.concept_dim = concept_dim
        self.alignment_threshold = alignment_threshold
        self.min_samples = min_samples
        self.history_length = history_length
        self.correlation_method = correlation_method
        self.concept_names = concept_names or [f"concept_{i}" for i in range(concept_dim)]
        
        # Alignment tracking
        self.concept_activations_history = deque(maxlen=history_length)
        self.human_labels_history = deque(maxlen=history_length)
        self.alignment_scores = torch.zeros(concept_dim)
        self.alignment_history = []
        self.aligned_concepts = set()
        
        # Statistics tracking
        self.correlation_matrix = torch.zeros(concept_dim, concept_dim)
        self.alignment_confidence = torch.zeros(concept_dim)
        self.update_count = 0
        
        logger.info(f"Initialized ConceptAlignmentTracker with {concept_dim} concepts, "
                   f"threshold={alignment_threshold}")
    
    def add_alignment_data(
        self,
        concept_activations: torch.Tensor,
        human_labels: torch.Tensor
    ) -> None:
        """
        Add new concept activation and human label data for alignment tracking.
        
        Args:
            concept_activations: Concept activations [batch_size, concept_dim]
            human_labels: Human labels [batch_size, concept_dim] (binary or continuous)
        """
        # Ensure tensors are on CPU for correlation computation
        concept_activations = concept_activations.detach().cpu()
        human_labels = human_labels.detach().cpu()
        
        # Add to history
        self.concept_activations_history.extend(concept_activations.numpy())
        self.human_labels_history.extend(human_labels.numpy())
        
        # Update alignment if we have enough samples
        if len(self.concept_activations_history) >= self.min_samples:
            self._update_alignment_scores()
    
    def _update_alignment_scores(self) -> None:
        """Update concept alignment scores based on current history."""
        if len(self.concept_activations_history) < self.min_samples:
            return
        
        # Convert history to arrays
        concept_data = np.array(list(self.concept_activations_history))
        human_data = np.array(list(self.human_labels_history))
        
        # Compute alignment for each concept dimension
        new_alignment_scores = []
        new_confidence_scores = []
        
        for i in range(self.concept_dim):
            concept_values = concept_data[:, i]
            human_values = human_data[:, i]
            
            # Compute correlation
            if self.correlation_method == 'pearson':
                correlation, p_value = pearsonr(concept_values, human_values)
            else:  # spearman
                correlation, p_value = spearmanr(concept_values, human_values)
            
            # Handle NaN correlations
            if np.isnan(correlation):
                correlation = 0.0
                p_value = 1.0
            
            alignment_score = abs(correlation)  # Use absolute correlation
            confidence = 1.0 - p_value  # Higher confidence for lower p-values
            
            new_alignment_scores.append(alignment_score)
            new_confidence_scores.append(confidence)
        
        # Update alignment scores
        self.alignment_scores = torch.tensor(new_alignment_scores, dtype=torch.float32)
        self.alignment_confidence = torch.tensor(new_confidence_scores, dtype=torch.float32)
        
        # Update aligned concepts set
        self.aligned_concepts = set(
            i for i, score in enumerate(self.alignment_scores)
            if score >= self.alignment_threshold
        )
        
        # Record alignment history
        alignment_record = {
            'step': self.update_count,
            'alignment_scores': self.alignment_scores.clone(),
            'aligned_concepts': list(self.aligned_concepts),
            'alignment_ratio': len(self.aligned_concepts) / self.concept_dim
        }
        self.alignment_history.append(alignment_record)
        
        self.update_count += 1
        
        logger.debug(f"Updated alignment: {len(self.aligned_concepts)}/{self.concept_dim} "
                    f"concepts aligned (ratio: {alignment_record['alignment_ratio']:.3f})")
    
    def get_alignment_metrics(self) -> Dict[str, Union[float, torch.Tensor, List]]:
        """
        Get comprehensive alignment metrics.
        
        Returns:
            Dictionary containing alignment metrics and statistics
        """
        if len(self.alignment_scores) == 0:
            return {
                'alignment_ratio': 0.0,
                'mean_alignment_score': 0.0,
                'aligned_concepts': [],
                'alignment_scores': torch.zeros(self.concept_dim),
                'alignment_confidence': torch.zeros(self.concept_dim)
            }
        
        return {
            'alignment_ratio': len(self.aligned_concepts) / self.concept_dim,
            'mean_alignment_score': self.alignment_scores.mean().item(),
            'max_alignment_score': self.alignment_scores.max().item(),
            'min_alignment_score': self.alignment_scores.min().item(),
            'aligned_concepts': list(self.aligned_concepts),
            'alignment_scores': self.alignment_scores.clone(),
            'alignment_confidence': self.alignment_confidence.clone(),
            'concept_names': self.concept_names,
            'total_samples': len(self.concept_activations_history)
        }
    
    def get_aligned_concept_names(self) -> List[str]:
        """Get names of aligned concepts."""
        return [self.concept_names[i] for i in self.aligned_concepts]
    
    def compute_alignment_loss(
        self,
        concept_activations: torch.Tensor,
        target_alignment: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute alignment-based regularization loss.
        
        Args:
            concept_activations: Current concept activations [batch_size, concept_dim]
            target_alignment: Optional target alignment pattern
            
        Returns:
            Alignment regularization loss
        """
        if len(self.aligned_concepts) == 0:
            return torch.tensor(0.0, device=concept_activations.device)
        
        # Encourage activation of aligned concepts
        aligned_mask = torch.zeros(self.concept_dim, device=concept_activations.device)
        for i in self.aligned_concepts:
            aligned_mask[i] = 1.0
        
        # Compute alignment loss - encourage aligned concepts to be active
        alignment_loss = -torch.mean(
            concept_activations * aligned_mask.unsqueeze(0)
        )
        
        return alignment_loss


def compute_concept_alignment(
    model_concepts: torch.Tensor,
    human_labels: torch.Tensor,
    threshold: float = 0.6,
    method: str = 'pearson'
) -> Dict[str, Union[float, torch.Tensor, List[int]]]:
    """
    Core concept alignment computation algorithm from Section 3.3.
    
    Algorithm:
    1. For each concept dimension i:
       - Compute correlation ρ_i = corr(C_i, H_i)
       - If ρ_i > τ_align: mark as aligned
    2. Return alignment_score = |aligned| / |total|
    
    Args:
        model_concepts: Model concept activations [N, concept_dim]
        human_labels: Human concept labels [N, concept_dim]
        threshold: Alignment threshold τ_align (default: 0.6)
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Dictionary with alignment metrics
    """
    # Convert to numpy for correlation computation
    model_concepts_np = model_concepts.detach().cpu().numpy()
    human_labels_np = human_labels.detach().cpu().numpy()
    
    concept_dim = model_concepts.shape[1]
    correlations = []
    p_values = []
    aligned_concepts = []
    
    # Compute correlation for each concept dimension
    for i in range(concept_dim):
        model_vals = model_concepts_np[:, i]
        human_vals = human_labels_np[:, i]
        
        # Compute correlation
        if method == 'pearson':
            corr, p_val = pearsonr(model_vals, human_vals)
        else:  # spearman
            corr, p_val = spearmanr(model_vals, human_vals)
        
        # Handle NaN correlations
        if np.isnan(corr):
            corr = 0.0
            p_val = 1.0
        
        correlations.append(abs(corr))  # Use absolute correlation
        p_values.append(p_val)
        
        # Check if concept is aligned
        if abs(corr) > threshold:
            aligned_concepts.append(i)
    
    # Compute alignment metrics
    alignment_score = len(aligned_concepts) / concept_dim
    mean_correlation = np.mean(correlations)
    
    return {
        'alignment_score': alignment_score,
        'mean_correlation': mean_correlation,
        'correlations': torch.tensor(correlations, dtype=torch.float32),
        'p_values': torch.tensor(p_values, dtype=torch.float32),
        'aligned_concepts': aligned_concepts,
        'num_aligned': len(aligned_concepts),
        'threshold': threshold
    }


def alignment_based_regularization(
    concept_activations: torch.Tensor,
    alignment_scores: torch.Tensor,
    regularization_strength: float = 0.1
) -> torch.Tensor:
    """
    Compute alignment-based regularization loss.
    
    Encourages the model to use concepts that are well-aligned with human understanding.
    
    Args:
        concept_activations: Current concept activations [batch_size, concept_dim]
        alignment_scores: Concept alignment scores [concept_dim]
        regularization_strength: Strength of regularization
        
    Returns:
        Regularization loss tensor
    """
    # Encourage activation of well-aligned concepts
    alignment_weights = alignment_scores.to(concept_activations.device)
    
    # Compute weighted activation loss
    weighted_activations = concept_activations * alignment_weights.unsqueeze(0)
    regularization_loss = -regularization_strength * torch.mean(weighted_activations)
    
    return regularization_loss


class HumanFeedbackIntegrator:
    """
    Integrates human feedback for concept refinement and alignment improvement.
    """
    
    def __init__(
        self,
        concept_dim: int,
        feedback_weight: float = 0.1,
        feedback_decay: float = 0.95
    ):
        """
        Initialize human feedback integrator.
        
        Args:
            concept_dim: Number of concept dimensions
            feedback_weight: Weight for human feedback in alignment computation
            feedback_decay: Decay factor for feedback over time
        """
        self.concept_dim = concept_dim
        self.feedback_weight = feedback_weight
        self.feedback_decay = feedback_decay
        
        # Feedback tracking
        self.positive_feedback = torch.zeros(concept_dim)
        self.negative_feedback = torch.zeros(concept_dim)
        self.feedback_counts = torch.zeros(concept_dim)
        
    def add_feedback(
        self,
        concept_indices: List[int],
        feedback_scores: List[float]
    ) -> None:
        """
        Add human feedback for specific concepts.
        
        Args:
            concept_indices: Indices of concepts receiving feedback
            feedback_scores: Feedback scores (positive > 0, negative < 0)
        """
        for idx, score in zip(concept_indices, feedback_scores):
            if 0 <= idx < self.concept_dim:
                if score > 0:
                    self.positive_feedback[idx] += score
                else:
                    self.negative_feedback[idx] += abs(score)
                self.feedback_counts[idx] += 1
    
    def get_feedback_weights(self) -> torch.Tensor:
        """
        Get concept weights based on accumulated feedback.
        
        Returns:
            Feedback-based concept weights [concept_dim]
        """
        # Compute feedback ratios
        total_feedback = self.positive_feedback + self.negative_feedback
        feedback_ratios = torch.where(
            total_feedback > 0,
            self.positive_feedback / total_feedback,
            torch.ones_like(total_feedback) * 0.5
        )
        
        # Apply feedback decay
        self.positive_feedback *= self.feedback_decay
        self.negative_feedback *= self.feedback_decay
        
        return feedback_ratios
    
    def compute_feedback_loss(
        self,
        concept_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feedback-based regularization loss.
        
        Args:
            concept_activations: Current concept activations [batch_size, concept_dim]
            
        Returns:
            Feedback regularization loss
        """
        feedback_weights = self.get_feedback_weights()
        feedback_weights = feedback_weights.to(concept_activations.device)
        
        # Encourage concepts with positive feedback, discourage negative
        feedback_loss = -self.feedback_weight * torch.mean(
            concept_activations * feedback_weights.unsqueeze(0)
        )
        
        return feedback_loss


def create_alignment_trainer(
    concept_dim: int,
    alignment_threshold: float = 0.6,
    regularization_strength: float = 0.1,
    use_human_feedback: bool = True,
    **kwargs
) -> Dict[str, object]:
    """
    Factory function to create complete concept alignment training setup.
    
    Args:
        concept_dim: Number of concept dimensions
        alignment_threshold: Alignment threshold for concept-human correlation
        regularization_strength: Strength of alignment regularization
        use_human_feedback: Whether to use human feedback integration
        **kwargs: Additional arguments for components
        
    Returns:
        Dictionary containing alignment training components
    """
    # Create alignment tracker
    alignment_tracker = ConceptAlignmentTracker(
        concept_dim=concept_dim,
        alignment_threshold=alignment_threshold,
        **kwargs
    )
    
    # Create human feedback integrator if requested
    feedback_integrator = None
    if use_human_feedback:
        feedback_integrator = HumanFeedbackIntegrator(
            concept_dim=concept_dim,
            feedback_weight=regularization_strength
        )
    
    return {
        'alignment_tracker': alignment_tracker,
        'feedback_integrator': feedback_integrator,
        'compute_alignment': compute_concept_alignment,
        'alignment_regularization': alignment_based_regularization,
        'regularization_strength': regularization_strength
    }


def evaluate_concept_interpretability(
    concept_activations: torch.Tensor,
    concept_names: List[str],
    human_labels: Optional[torch.Tensor] = None,
    top_k: int = 5
) -> Dict[str, Union[List, float]]:
    """
    Evaluate concept interpretability through various metrics.
    
    Args:
        concept_activations: Concept activations [N, concept_dim]
        concept_names: Names of concepts for interpretability
        human_labels: Optional human labels for alignment evaluation
        top_k: Number of top concepts to analyze
        
    Returns:
        Dictionary with interpretability metrics
    """
    concept_dim = concept_activations.shape[1]
    
    # Compute concept statistics
    mean_activations = concept_activations.mean(dim=0)
    std_activations = concept_activations.std(dim=0)
    
    # Find most and least active concepts
    top_concepts_idx = torch.topk(mean_activations, top_k).indices
    bottom_concepts_idx = torch.topk(mean_activations, top_k, largest=False).indices
    
    top_concepts = [(concept_names[i], mean_activations[i].item()) for i in top_concepts_idx]
    bottom_concepts = [(concept_names[i], mean_activations[i].item()) for i in bottom_concepts_idx]
    
    # Compute concept diversity (entropy)
    concept_probs = F.softmax(mean_activations, dim=0)
    concept_entropy = -torch.sum(concept_probs * torch.log(concept_probs + 1e-8))
    
    results = {
        'top_concepts': top_concepts,
        'bottom_concepts': bottom_concepts,
        'concept_entropy': concept_entropy.item(),
        'mean_activation': mean_activations.mean().item(),
        'activation_std': std_activations.mean().item()
    }
    
    # Add alignment metrics if human labels provided
    if human_labels is not None:
        alignment_metrics = compute_concept_alignment(concept_activations, human_labels)
        results.update(alignment_metrics)
    
    return results


# Example usage and testing functions
def test_concept_alignment():
    """Test concept alignment functionality."""
    print("Testing Concept Alignment Module...")
    
    # Create test data
    concept_dim = 10
    batch_size = 100
    
    # Simulate concept activations and human labels
    concept_activations = torch.sigmoid(torch.randn(batch_size, concept_dim))
    
    # Create correlated human labels for some concepts
    human_labels = torch.zeros(batch_size, concept_dim)
    for i in range(concept_dim):
        if i < 5:  # First 5 concepts are aligned
            human_labels[:, i] = concept_activations[:, i] + 0.1 * torch.randn(batch_size)
        else:  # Last 5 concepts are random
            human_labels[:, i] = torch.rand(batch_size)
    
    # Test alignment computation
    alignment_results = compute_concept_alignment(concept_activations, human_labels)
    print(f"Alignment Results: {alignment_results}")
    
    # Test alignment tracker
    tracker = ConceptAlignmentTracker(concept_dim)
    tracker.add_alignment_data(concept_activations, human_labels)
    
    metrics = tracker.get_alignment_metrics()
    print(f"Alignment Metrics: {metrics}")
    
    # Test human feedback integration
    feedback_integrator = HumanFeedbackIntegrator(concept_dim)
    feedback_integrator.add_feedback([0, 1, 2], [1.0, 0.8, -0.5])
    
    feedback_weights = feedback_integrator.get_feedback_weights()
    print(f"Feedback Weights: {feedback_weights}")
    
    print("Concept Alignment Module test completed successfully!")


if __name__ == "__main__":
    test_concept_alignment()