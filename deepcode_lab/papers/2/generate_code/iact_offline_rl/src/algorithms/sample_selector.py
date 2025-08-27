"""
Sample Selection Module for IACT (Importance-Aware Co-Teaching) Algorithm

This module implements the co-teaching sample selection mechanism where two policies
teach each other by selecting high-quality samples based on importance weights and
policy confidence scores.

Key Components:
- SampleSelector: Main class for coordinating sample selection between dual policies
- CoTeachingScheduler: Manages selection rates and curriculum learning
- SampleQualityMetrics: Computes quality scores for sample selection
- BatchSelector: Utilities for efficient batch-wise sample selection

Mathematical Foundation:
- Sample quality score: Q(s,a) = w(s) * π(a|s) * confidence(s,a)
- Selection criterion: Top-k samples with highest quality scores
- Curriculum learning: Gradually decrease selection rate from high to low
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Sample selection strategies for co-teaching"""
    IMPORTANCE_WEIGHTED = "importance_weighted"
    CONFIDENCE_BASED = "confidence_based"
    COMBINED = "combined"
    RANDOM = "random"


@dataclass
class SelectionMetrics:
    """Metrics for tracking sample selection quality"""
    selected_indices: torch.Tensor
    quality_scores: torch.Tensor
    selection_rate: float
    mean_importance: float
    mean_confidence: float
    selection_threshold: float


class CoTeachingScheduler:
    """
    Manages the curriculum learning schedule for co-teaching sample selection.
    
    The selection rate starts high (to select easy/high-quality samples) and
    gradually decreases to include more challenging samples as training progresses.
    """
    
    def __init__(
        self,
        initial_rate: float = 0.8,
        final_rate: float = 0.3,
        decay_type: str = "linear",
        decay_steps: int = 10000,
        warmup_steps: int = 1000
    ):
        """
        Initialize co-teaching scheduler.
        
        Args:
            initial_rate: Starting selection rate (high for easy samples)
            final_rate: Final selection rate (lower for harder samples)
            decay_type: Type of decay ("linear", "exponential", "cosine")
            decay_steps: Number of steps for rate decay
            warmup_steps: Number of warmup steps with constant initial rate
        """
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        logger.info(f"CoTeachingScheduler initialized: {initial_rate} -> {final_rate} over {decay_steps} steps")
    
    def get_selection_rate(self, step: Optional[int] = None) -> float:
        """
        Get current selection rate based on training step.
        
        Args:
            step: Current training step (uses internal counter if None)
            
        Returns:
            Current selection rate
        """
        if step is not None:
            self.current_step = step
        
        # Warmup phase
        if self.current_step < self.warmup_steps:
            return self.initial_rate
        
        # Decay phase
        progress = min(1.0, (self.current_step - self.warmup_steps) / self.decay_steps)
        
        if self.decay_type == "linear":
            rate = self.initial_rate - (self.initial_rate - self.final_rate) * progress
        elif self.decay_type == "exponential":
            rate = self.final_rate + (self.initial_rate - self.final_rate) * np.exp(-5 * progress)
        elif self.decay_type == "cosine":
            rate = self.final_rate + (self.initial_rate - self.final_rate) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        return max(self.final_rate, min(self.initial_rate, rate))
    
    def step(self) -> float:
        """Advance scheduler by one step and return current rate"""
        self.current_step += 1
        return self.get_selection_rate()


class SampleQualityMetrics:
    """
    Computes quality metrics for sample selection in co-teaching.
    
    Quality is determined by combining importance weights (from density ratio estimation)
    with policy confidence scores to identify the most informative samples.
    """
    
    def __init__(
        self,
        importance_weight: float = 1.0,
        confidence_weight: float = 1.0,
        uncertainty_weight: float = 0.0,
        normalize_scores: bool = True
    ):
        """
        Initialize quality metrics computation.
        
        Args:
            importance_weight: Weight for importance scores
            confidence_weight: Weight for confidence scores
            uncertainty_weight: Weight for uncertainty scores (higher uncertainty = lower quality)
            normalize_scores: Whether to normalize quality scores
        """
        self.importance_weight = importance_weight
        self.confidence_weight = confidence_weight
        self.uncertainty_weight = uncertainty_weight
        self.normalize_scores = normalize_scores
    
    def compute_quality_scores(
        self,
        importance_weights: torch.Tensor,
        confidence_scores: torch.Tensor,
        uncertainty_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sample quality scores for selection.
        
        Args:
            importance_weights: State importance weights w(s)
            confidence_scores: Policy confidence scores π(a|s)
            uncertainty_scores: Optional uncertainty estimates
            
        Returns:
            Quality scores for each sample
        """
        # Ensure tensors are on the same device
        device = importance_weights.device
        confidence_scores = confidence_scores.to(device)
        
        # Base quality: importance * confidence
        quality_scores = (
            self.importance_weight * importance_weights +
            self.confidence_weight * confidence_scores
        )
        
        # Subtract uncertainty if provided
        if uncertainty_scores is not None:
            uncertainty_scores = uncertainty_scores.to(device)
            quality_scores -= self.uncertainty_weight * uncertainty_scores
        
        # Normalize scores if requested
        if self.normalize_scores:
            quality_scores = self._normalize_scores(quality_scores)
        
        return quality_scores
    
    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range"""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return torch.ones_like(scores)
    
    def compute_confidence_scores(
        self,
        policy,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy confidence scores for given state-action pairs.
        
        Args:
            policy: Policy network
            states: State tensor
            actions: Action tensor
            
        Returns:
            Confidence scores
        """
        with torch.no_grad():
            log_probs = policy.log_prob(states, actions)
            confidence_scores = torch.exp(log_probs)
            
            # Clip extreme values
            confidence_scores = torch.clamp(confidence_scores, min=1e-8, max=1.0)
            
        return confidence_scores


class BatchSelector:
    """
    Efficient utilities for batch-wise sample selection.
    
    Handles the actual selection of samples from batches based on quality scores
    and selection rates, with support for different selection strategies.
    """
    
    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.COMBINED):
        """
        Initialize batch selector.
        
        Args:
            strategy: Sample selection strategy
        """
        self.strategy = strategy
    
    def select_samples(
        self,
        batch: Dict[str, torch.Tensor],
        quality_scores: torch.Tensor,
        selection_rate: float,
        min_samples: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Select samples from batch based on quality scores.
        
        Args:
            batch: Dictionary containing batch data
            quality_scores: Quality scores for each sample
            selection_rate: Fraction of samples to select
            min_samples: Minimum number of samples to select
            
        Returns:
            Tuple of (selected_batch, selected_indices)
        """
        batch_size = len(quality_scores)
        n_select = max(min_samples, int(batch_size * selection_rate))
        
        if self.strategy == SelectionStrategy.RANDOM:
            # Random selection for baseline
            selected_indices = torch.randperm(batch_size)[:n_select]
        else:
            # Select top-k samples based on quality scores
            selected_indices = torch.topk(quality_scores, n_select, largest=True).indices
        
        # Create selected batch
        selected_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                selected_batch[key] = value[selected_indices]
            else:
                selected_batch[key] = [value[i] for i in selected_indices.cpu().numpy()]
        
        return selected_batch, selected_indices
    
    def compute_selection_threshold(
        self,
        quality_scores: torch.Tensor,
        selection_rate: float
    ) -> float:
        """
        Compute quality threshold for sample selection.
        
        Args:
            quality_scores: Quality scores for all samples
            selection_rate: Fraction of samples to select
            
        Returns:
            Quality threshold for selection
        """
        n_select = int(len(quality_scores) * selection_rate)
        if n_select == 0:
            return float('inf')
        
        sorted_scores, _ = torch.sort(quality_scores, descending=True)
        threshold = sorted_scores[min(n_select - 1, len(sorted_scores) - 1)].item()
        
        return threshold


class SampleSelector:
    """
    Main sample selector for IACT co-teaching algorithm.
    
    Coordinates the co-teaching process between dual policies by selecting
    high-quality samples based on importance weights and policy confidence.
    Each policy selects samples for training the other policy.
    """
    
    def __init__(
        self,
        scheduler_config: Optional[Dict[str, Any]] = None,
        quality_config: Optional[Dict[str, Any]] = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.COMBINED,
        device: str = "cpu"
    ):
        """
        Initialize sample selector.
        
        Args:
            scheduler_config: Configuration for co-teaching scheduler
            quality_config: Configuration for quality metrics
            selection_strategy: Sample selection strategy
            device: Device for computations
        """
        self.device = device
        
        # Initialize scheduler
        scheduler_config = scheduler_config or {}
        self.scheduler = CoTeachingScheduler(**scheduler_config)
        
        # Initialize quality metrics
        quality_config = quality_config or {}
        self.quality_metrics = SampleQualityMetrics(**quality_config)
        
        # Initialize batch selector
        self.batch_selector = BatchSelector(strategy=selection_strategy)
        
        # Tracking variables
        self.selection_history = []
        self.current_step = 0
        
        logger.info(f"SampleSelector initialized with strategy: {selection_strategy}")
    
    def select_for_policy(
        self,
        selecting_policy,
        target_policy_name: str,
        batch: Dict[str, torch.Tensor],
        importance_weights: torch.Tensor,
        step: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], SelectionMetrics]:
        """
        Select samples for training a target policy using a selecting policy.
        
        Args:
            selecting_policy: Policy that performs the selection
            target_policy_name: Name of the policy being trained
            batch: Batch of training data
            importance_weights: Importance weights for each sample
            step: Current training step
            
        Returns:
            Tuple of (selected_batch, selection_metrics)
        """
        if step is not None:
            self.current_step = step
        
        # Get current selection rate
        selection_rate = self.scheduler.get_selection_rate(self.current_step)
        
        # Extract states and actions from batch
        states = batch['states']
        actions = batch['actions']
        
        # Compute confidence scores using selecting policy
        confidence_scores = self.quality_metrics.compute_confidence_scores(
            selecting_policy, states, actions
        )
        
        # Compute quality scores
        quality_scores = self.quality_metrics.compute_quality_scores(
            importance_weights, confidence_scores
        )
        
        # Select samples
        selected_batch, selected_indices = self.batch_selector.select_samples(
            batch, quality_scores, selection_rate
        )
        
        # Compute selection threshold
        selection_threshold = self.batch_selector.compute_selection_threshold(
            quality_scores, selection_rate
        )
        
        # Create selection metrics
        metrics = SelectionMetrics(
            selected_indices=selected_indices,
            quality_scores=quality_scores,
            selection_rate=selection_rate,
            mean_importance=importance_weights.mean().item(),
            mean_confidence=confidence_scores.mean().item(),
            selection_threshold=selection_threshold
        )
        
        # Update history
        self.selection_history.append({
            'step': self.current_step,
            'target_policy': target_policy_name,
            'selection_rate': selection_rate,
            'n_selected': len(selected_indices),
            'mean_quality': quality_scores.mean().item(),
            'threshold': selection_threshold
        })
        
        # Advance step
        self.current_step += 1
        
        return selected_batch, metrics
    
    def co_teaching_selection(
        self,
        policy_a,
        policy_b,
        batch: Dict[str, torch.Tensor],
        importance_weights: torch.Tensor,
        step: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, SelectionMetrics]]:
        """
        Perform co-teaching selection where each policy selects samples for the other.
        
        Args:
            policy_a: First policy
            policy_b: Second policy
            batch: Batch of training data
            importance_weights: Importance weights for each sample
            step: Current training step
            
        Returns:
            Tuple of (batch_for_a, batch_for_b, selection_metrics)
        """
        # Policy A selects samples for Policy B
        batch_for_b, metrics_a = self.select_for_policy(
            policy_a, "policy_b", batch, importance_weights, step
        )
        
        # Policy B selects samples for Policy A
        batch_for_a, metrics_b = self.select_for_policy(
            policy_b, "policy_a", batch, importance_weights, step
        )
        
        metrics = {
            'policy_a_selects_for_b': metrics_a,
            'policy_b_selects_for_a': metrics_b
        }
        
        return batch_for_a, batch_for_b, metrics
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the selection process.
        
        Returns:
            Dictionary containing selection statistics
        """
        if not self.selection_history:
            return {}
        
        recent_history = self.selection_history[-100:]  # Last 100 selections
        
        stats = {
            'total_selections': len(self.selection_history),
            'current_step': self.current_step,
            'current_selection_rate': self.scheduler.get_selection_rate(),
            'mean_selection_rate': np.mean([h['selection_rate'] for h in recent_history]),
            'mean_quality_score': np.mean([h['mean_quality'] for h in recent_history]),
            'mean_samples_selected': np.mean([h['n_selected'] for h in recent_history]),
            'selection_rate_trend': self._compute_trend([h['selection_rate'] for h in recent_history])
        }
        
        return stats
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction for a list of values"""
        if len(values) < 2:
            return "stable"
        
        slope = (values[-1] - values[0]) / len(values)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def reset(self):
        """Reset the sample selector state"""
        self.scheduler.current_step = 0
        self.current_step = 0
        self.selection_history.clear()
        logger.info("SampleSelector reset")


# Factory functions for easy instantiation

def create_sample_selector(
    config: Optional[Dict[str, Any]] = None,
    device: str = "cpu"
) -> SampleSelector:
    """
    Create a configured sample selector.
    
    Args:
        config: Configuration dictionary
        device: Device for computations
        
    Returns:
        Configured SampleSelector instance
    """
    config = config or {}
    
    # Extract sub-configurations
    scheduler_config = config.get('scheduler', {})
    quality_config = config.get('quality', {})
    strategy = SelectionStrategy(config.get('strategy', 'combined'))
    
    return SampleSelector(
        scheduler_config=scheduler_config,
        quality_config=quality_config,
        selection_strategy=strategy,
        device=device
    )


def create_co_teaching_scheduler(
    initial_rate: float = 0.8,
    final_rate: float = 0.3,
    decay_steps: int = 10000,
    **kwargs
) -> CoTeachingScheduler:
    """
    Create a co-teaching scheduler with common settings.
    
    Args:
        initial_rate: Starting selection rate
        final_rate: Final selection rate
        decay_steps: Number of decay steps
        **kwargs: Additional scheduler parameters
        
    Returns:
        Configured CoTeachingScheduler
    """
    return CoTeachingScheduler(
        initial_rate=initial_rate,
        final_rate=final_rate,
        decay_steps=decay_steps,
        **kwargs
    )


# Utility functions for sample selection analysis

def analyze_selection_quality(
    selected_batch: Dict[str, torch.Tensor],
    original_batch: Dict[str, torch.Tensor],
    importance_weights: torch.Tensor,
    selected_indices: torch.Tensor
) -> Dict[str, float]:
    """
    Analyze the quality of sample selection.
    
    Args:
        selected_batch: Selected samples
        original_batch: Original batch
        importance_weights: Importance weights for original batch
        selected_indices: Indices of selected samples
        
    Returns:
        Dictionary with quality analysis metrics
    """
    selected_importance = importance_weights[selected_indices]
    
    analysis = {
        'selection_ratio': len(selected_indices) / len(importance_weights),
        'mean_selected_importance': selected_importance.mean().item(),
        'mean_original_importance': importance_weights.mean().item(),
        'importance_improvement': (selected_importance.mean() / importance_weights.mean()).item(),
        'max_selected_importance': selected_importance.max().item(),
        'min_selected_importance': selected_importance.min().item(),
        'importance_std': selected_importance.std().item()
    }
    
    return analysis


def visualize_selection_history(
    selection_history: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the selection history (requires matplotlib).
    
    Args:
        selection_history: History of selections
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        steps = [h['step'] for h in selection_history]
        rates = [h['selection_rate'] for h in selection_history]
        qualities = [h['mean_quality'] for h in selection_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Selection rate over time
        ax1.plot(steps, rates, 'b-', label='Selection Rate')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Selection Rate')
        ax1.set_title('Co-Teaching Selection Rate Over Time')
        ax1.grid(True)
        ax1.legend()
        
        # Quality scores over time
        ax2.plot(steps, qualities, 'r-', label='Mean Quality Score')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Sample Quality Over Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Selection history plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        logger.warning("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing SampleSelector implementation...")
    
    # Create sample selector
    config = {
        'scheduler': {
            'initial_rate': 0.8,
            'final_rate': 0.3,
            'decay_steps': 1000
        },
        'quality': {
            'importance_weight': 1.0,
            'confidence_weight': 0.5
        },
        'strategy': 'combined'
    }
    
    selector = create_sample_selector(config)
    
    # Test scheduler
    print("Testing scheduler...")
    for step in range(0, 1500, 100):
        rate = selector.scheduler.get_selection_rate(step)
        print(f"Step {step}: Selection rate = {rate:.3f}")
    
    # Test quality metrics
    print("\nTesting quality metrics...")
    importance_weights = torch.rand(100)
    confidence_scores = torch.rand(100)
    
    quality_scores = selector.quality_metrics.compute_quality_scores(
        importance_weights, confidence_scores
    )
    
    print(f"Quality scores shape: {quality_scores.shape}")
    print(f"Quality scores range: [{quality_scores.min():.3f}, {quality_scores.max():.3f}]")
    
    # Test batch selection
    print("\nTesting batch selection...")
    batch = {
        'states': torch.randn(100, 10),
        'actions': torch.randn(100, 4),
        'rewards': torch.randn(100, 1),
        'next_states': torch.randn(100, 10),
        'dones': torch.randint(0, 2, (100, 1)).float()
    }
    
    selected_batch, selected_indices = selector.batch_selector.select_samples(
        batch, quality_scores, selection_rate=0.5
    )
    
    print(f"Original batch size: {len(batch['states'])}")
    print(f"Selected batch size: {len(selected_batch['states'])}")
    print(f"Selected indices: {selected_indices[:10]}...")
    
    # Test selection analysis
    analysis = analyze_selection_quality(
        selected_batch, batch, importance_weights, selected_indices
    )
    
    print(f"\nSelection analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nSampleSelector implementation test completed successfully!")