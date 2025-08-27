"""
Behavior Cloning Regularizer for IACT Algorithm

This module implements behavior cloning regularization to prevent out-of-distribution (OOD) actions
in offline reinforcement learning. The regularizer constrains the learned policy to stay close to
the behavior policy that generated the offline dataset.

Key Features:
- Importance-weighted behavior cloning loss
- Dynamic regularization weight scheduling
- Multiple regularization strategies (KL divergence, MSE, etc.)
- Integration with co-teaching framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BCRegularizerConfig:
    """Configuration for behavior cloning regularizer."""
    
    # Regularization type
    regularization_type: str = "kl_divergence"  # "kl_divergence", "mse", "log_prob"
    
    # Weight scheduling
    initial_weight: float = 1.0
    final_weight: float = 0.1
    decay_schedule: str = "linear"  # "linear", "exponential", "cosine"
    decay_epochs: int = 1000
    
    # Importance weighting
    use_importance_weights: bool = True
    importance_weight_clip: float = 10.0
    
    # Adaptive weighting
    adaptive_weighting: bool = True
    target_kl: float = 0.1
    adaptation_rate: float = 0.01
    
    # Numerical stability
    epsilon: float = 1e-8
    gradient_clip: float = 1.0


class BCRegularizer(nn.Module):
    """
    Behavior Cloning Regularizer for offline RL.
    
    Implements various forms of behavior cloning regularization to prevent
    the learned policy from deviating too far from the behavior policy.
    """
    
    def __init__(self, config: Optional[BCRegularizerConfig] = None):
        super().__init__()
        self.config = config or BCRegularizerConfig()
        
        # Current regularization weight
        self.current_weight = self.config.initial_weight
        
        # Adaptive weight tracking
        self.adaptive_weight = self.config.initial_weight
        self.recent_kl_values = []
        
        # Statistics tracking
        self.total_steps = 0
        self.regularization_stats = {
            'bc_loss': [],
            'kl_divergence': [],
            'weight': [],
            'importance_weights_mean': [],
            'importance_weights_std': []
        }
        
        logger.info(f"Initialized BC Regularizer with config: {self.config}")
    
    def compute_bc_loss(
        self,
        policy_actions: torch.Tensor,
        behavior_actions: torch.Tensor,
        policy_log_probs: Optional[torch.Tensor] = None,
        behavior_log_probs: Optional[torch.Tensor] = None,
        importance_weights: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute behavior cloning regularization loss.
        
        Args:
            policy_actions: Actions from current policy [batch_size, action_dim]
            behavior_actions: Actions from behavior policy [batch_size, action_dim]
            policy_log_probs: Log probabilities from current policy [batch_size]
            behavior_log_probs: Log probabilities from behavior policy [batch_size]
            importance_weights: Importance weights for samples [batch_size]
            states: State observations [batch_size, state_dim]
            
        Returns:
            Dictionary containing:
                - bc_loss: Behavior cloning loss
                - kl_divergence: KL divergence between policies
                - weighted_loss: Importance-weighted loss
                - regularization_weight: Current regularization weight
        """
        batch_size = policy_actions.shape[0]
        device = policy_actions.device
        
        # Handle importance weights
        if importance_weights is None:
            importance_weights = torch.ones(batch_size, device=device)
        else:
            # Clip importance weights for stability
            importance_weights = torch.clamp(
                importance_weights, 
                max=self.config.importance_weight_clip
            )
        
        # Compute base BC loss based on regularization type
        if self.config.regularization_type == "mse":
            bc_loss = self._compute_mse_loss(policy_actions, behavior_actions)
        elif self.config.regularization_type == "log_prob":
            bc_loss = self._compute_log_prob_loss(policy_log_probs, behavior_log_probs)
        elif self.config.regularization_type == "kl_divergence":
            bc_loss = self._compute_kl_loss(policy_log_probs, behavior_log_probs)
        else:
            raise ValueError(f"Unknown regularization type: {self.config.regularization_type}")
        
        # Apply importance weighting if enabled
        if self.config.use_importance_weights:
            weighted_loss = (bc_loss * importance_weights).mean()
        else:
            weighted_loss = bc_loss.mean()
        
        # Compute KL divergence for monitoring
        if policy_log_probs is not None and behavior_log_probs is not None:
            kl_div = self._compute_kl_divergence(policy_log_probs, behavior_log_probs)
        else:
            kl_div = torch.tensor(0.0, device=device)
        
        # Update adaptive weight if enabled
        if self.config.adaptive_weighting:
            self._update_adaptive_weight(kl_div.item())
        
        # Apply regularization weight
        final_loss = self.current_weight * weighted_loss
        
        # Update statistics
        self._update_statistics(bc_loss, kl_div, importance_weights)
        
        return {
            'bc_loss': bc_loss.mean(),
            'kl_divergence': kl_div,
            'weighted_loss': weighted_loss,
            'final_loss': final_loss,
            'regularization_weight': self.current_weight,
            'importance_weights_mean': importance_weights.mean(),
            'importance_weights_std': importance_weights.std()
        }
    
    def _compute_mse_loss(
        self, 
        policy_actions: torch.Tensor, 
        behavior_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss between policy and behavior actions."""
        return F.mse_loss(policy_actions, behavior_actions, reduction='none').sum(dim=-1)
    
    def _compute_log_prob_loss(
        self, 
        policy_log_probs: torch.Tensor, 
        behavior_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log probability loss."""
        if policy_log_probs is None or behavior_log_probs is None:
            raise ValueError("Log probabilities required for log_prob regularization")
        return -policy_log_probs
    
    def _compute_kl_loss(
        self, 
        policy_log_probs: torch.Tensor, 
        behavior_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence loss."""
        if policy_log_probs is None or behavior_log_probs is None:
            raise ValueError("Log probabilities required for KL divergence regularization")
        return self._compute_kl_divergence(policy_log_probs, behavior_log_probs)
    
    def _compute_kl_divergence(
        self, 
        policy_log_probs: torch.Tensor, 
        behavior_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and behavior distributions."""
        # KL(π || β) = E[log π(a|s) - log β(a|s)]
        kl_div = policy_log_probs - behavior_log_probs
        return kl_div
    
    def update_weight(self, epoch: int) -> float:
        """
        Update regularization weight based on schedule.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Updated regularization weight
        """
        if self.config.decay_schedule == "linear":
            progress = min(epoch / self.config.decay_epochs, 1.0)
            self.current_weight = (
                self.config.initial_weight * (1 - progress) + 
                self.config.final_weight * progress
            )
        elif self.config.decay_schedule == "exponential":
            decay_rate = (self.config.final_weight / self.config.initial_weight) ** (1 / self.config.decay_epochs)
            self.current_weight = self.config.initial_weight * (decay_rate ** epoch)
        elif self.config.decay_schedule == "cosine":
            progress = min(epoch / self.config.decay_epochs, 1.0)
            self.current_weight = (
                self.config.final_weight + 
                0.5 * (self.config.initial_weight - self.config.final_weight) * 
                (1 + np.cos(np.pi * progress))
            )
        
        # Use adaptive weight if enabled
        if self.config.adaptive_weighting:
            self.current_weight = self.adaptive_weight
        
        return self.current_weight
    
    def _update_adaptive_weight(self, kl_value: float):
        """Update adaptive regularization weight based on KL divergence."""
        self.recent_kl_values.append(kl_value)
        
        # Keep only recent values
        if len(self.recent_kl_values) > 100:
            self.recent_kl_values.pop(0)
        
        # Compute average KL
        avg_kl = np.mean(self.recent_kl_values)
        
        # Adjust weight based on target KL
        if avg_kl > self.config.target_kl:
            # KL too high, increase regularization
            self.adaptive_weight *= (1 + self.config.adaptation_rate)
        else:
            # KL acceptable, decrease regularization
            self.adaptive_weight *= (1 - self.config.adaptation_rate)
        
        # Clamp adaptive weight
        self.adaptive_weight = np.clip(
            self.adaptive_weight,
            self.config.final_weight,
            self.config.initial_weight * 2
        )
    
    def _update_statistics(
        self, 
        bc_loss: torch.Tensor, 
        kl_div: torch.Tensor, 
        importance_weights: torch.Tensor
    ):
        """Update regularization statistics."""
        self.regularization_stats['bc_loss'].append(bc_loss.mean().item())
        self.regularization_stats['kl_divergence'].append(kl_div.mean().item())
        self.regularization_stats['weight'].append(self.current_weight)
        self.regularization_stats['importance_weights_mean'].append(importance_weights.mean().item())
        self.regularization_stats['importance_weights_std'].append(importance_weights.std().item())
        
        # Keep only recent statistics
        max_history = 1000
        for key in self.regularization_stats:
            if len(self.regularization_stats[key]) > max_history:
                self.regularization_stats[key] = self.regularization_stats[key][-max_history:]
        
        self.total_steps += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current regularization statistics."""
        if not self.regularization_stats['bc_loss']:
            return {}
        
        stats = {}
        for key, values in self.regularization_stats.items():
            if values:
                stats[f'bc_reg_{key}_mean'] = np.mean(values[-100:])  # Recent average
                stats[f'bc_reg_{key}_std'] = np.std(values[-100:])
        
        stats['bc_reg_total_steps'] = self.total_steps
        stats['bc_reg_current_weight'] = self.current_weight
        
        return stats
    
    def reset_statistics(self):
        """Reset regularization statistics."""
        for key in self.regularization_stats:
            self.regularization_stats[key] = []
        self.total_steps = 0
    
    def save_state(self) -> Dict:
        """Save regularizer state for checkpointing."""
        return {
            'current_weight': self.current_weight,
            'adaptive_weight': self.adaptive_weight,
            'recent_kl_values': self.recent_kl_values,
            'total_steps': self.total_steps,
            'regularization_stats': self.regularization_stats,
            'config': self.config
        }
    
    def load_state(self, state: Dict):
        """Load regularizer state from checkpoint."""
        self.current_weight = state['current_weight']
        self.adaptive_weight = state['adaptive_weight']
        self.recent_kl_values = state['recent_kl_values']
        self.total_steps = state['total_steps']
        self.regularization_stats = state['regularization_stats']
        # Note: config is not loaded to preserve initialization settings


class ImportanceWeightedBCRegularizer(BCRegularizer):
    """
    Specialized BC regularizer with enhanced importance weighting.
    
    This version provides additional importance weighting strategies
    specifically designed for the IACT algorithm.
    """
    
    def __init__(self, config: Optional[BCRegularizerConfig] = None):
        super().__init__(config)
        
        # Enhanced importance weighting parameters
        self.importance_weight_history = []
        self.weight_adaptation_rate = 0.1
    
    def compute_enhanced_bc_loss(
        self,
        policy_actions: torch.Tensor,
        behavior_actions: torch.Tensor,
        importance_weights: torch.Tensor,
        policy_confidence: Optional[torch.Tensor] = None,
        co_teaching_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced BC loss with multiple weighting strategies.
        
        Args:
            policy_actions: Actions from current policy
            behavior_actions: Actions from behavior policy
            importance_weights: KLIEP importance weights
            policy_confidence: Policy confidence scores
            co_teaching_weights: Co-teaching sample weights
            
        Returns:
            Enhanced BC loss dictionary
        """
        # Base BC loss
        base_loss = self._compute_mse_loss(policy_actions, behavior_actions)
        
        # Combine different weighting strategies
        combined_weights = importance_weights
        
        if policy_confidence is not None:
            # Weight by policy confidence (higher confidence = lower BC weight)
            confidence_weights = 1.0 / (policy_confidence + self.config.epsilon)
            combined_weights = combined_weights * confidence_weights
        
        if co_teaching_weights is not None:
            # Incorporate co-teaching sample selection
            combined_weights = combined_weights * co_teaching_weights
        
        # Normalize weights
        combined_weights = combined_weights / (combined_weights.mean() + self.config.epsilon)
        
        # Clip for stability
        combined_weights = torch.clamp(combined_weights, max=self.config.importance_weight_clip)
        
        # Compute weighted loss
        weighted_loss = (base_loss * combined_weights).mean()
        
        # Apply regularization weight
        final_loss = self.current_weight * weighted_loss
        
        return {
            'bc_loss': base_loss.mean(),
            'weighted_loss': weighted_loss,
            'final_loss': final_loss,
            'combined_weights_mean': combined_weights.mean(),
            'combined_weights_std': combined_weights.std(),
            'regularization_weight': self.current_weight
        }


def create_bc_regularizer(
    regularization_type: str = "kl_divergence",
    initial_weight: float = 1.0,
    final_weight: float = 0.1,
    use_importance_weights: bool = True,
    **kwargs
) -> BCRegularizer:
    """
    Factory function to create BC regularizer.
    
    Args:
        regularization_type: Type of regularization ("kl_divergence", "mse", "log_prob")
        initial_weight: Initial regularization weight
        final_weight: Final regularization weight
        use_importance_weights: Whether to use importance weighting
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BC regularizer
    """
    config = BCRegularizerConfig(
        regularization_type=regularization_type,
        initial_weight=initial_weight,
        final_weight=final_weight,
        use_importance_weights=use_importance_weights,
        **kwargs
    )
    
    return BCRegularizer(config)


def create_enhanced_bc_regularizer(
    regularization_type: str = "kl_divergence",
    initial_weight: float = 1.0,
    final_weight: float = 0.1,
    use_importance_weights: bool = True,
    **kwargs
) -> ImportanceWeightedBCRegularizer:
    """
    Factory function to create enhanced BC regularizer.
    
    Args:
        regularization_type: Type of regularization
        initial_weight: Initial regularization weight
        final_weight: Final regularization weight
        use_importance_weights: Whether to use importance weighting
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured enhanced BC regularizer
    """
    config = BCRegularizerConfig(
        regularization_type=regularization_type,
        initial_weight=initial_weight,
        final_weight=final_weight,
        use_importance_weights=use_importance_weights,
        **kwargs
    )
    
    return ImportanceWeightedBCRegularizer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test BC regularizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    batch_size = 64
    action_dim = 4
    
    policy_actions = torch.randn(batch_size, action_dim, device=device)
    behavior_actions = torch.randn(batch_size, action_dim, device=device)
    policy_log_probs = torch.randn(batch_size, device=device)
    behavior_log_probs = torch.randn(batch_size, device=device)
    importance_weights = torch.rand(batch_size, device=device) * 2
    
    # Test basic BC regularizer
    bc_reg = create_bc_regularizer(regularization_type="kl_divergence")
    
    loss_dict = bc_reg.compute_bc_loss(
        policy_actions=policy_actions,
        behavior_actions=behavior_actions,
        policy_log_probs=policy_log_probs,
        behavior_log_probs=behavior_log_probs,
        importance_weights=importance_weights
    )
    
    print("BC Regularizer Test Results:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test enhanced BC regularizer
    enhanced_bc_reg = create_enhanced_bc_regularizer()
    
    enhanced_loss_dict = enhanced_bc_reg.compute_enhanced_bc_loss(
        policy_actions=policy_actions,
        behavior_actions=behavior_actions,
        importance_weights=importance_weights,
        policy_confidence=torch.rand(batch_size, device=device)
    )
    
    print("\nEnhanced BC Regularizer Test Results:")
    for key, value in enhanced_loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test weight scheduling
    print("\nWeight Scheduling Test:")
    for epoch in [0, 100, 500, 1000]:
        weight = bc_reg.update_weight(epoch)
        print(f"  Epoch {epoch}: weight = {weight:.4f}")
    
    print("\nBC Regularizer implementation completed successfully!")