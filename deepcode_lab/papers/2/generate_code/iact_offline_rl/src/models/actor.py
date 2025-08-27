"""
Actor Network Implementation for IACT Offline RL

This module implements the actor (policy) networks for the IACT algorithm,
building upon the base GaussianPolicy from networks.py. The actor is responsible
for learning the policy that maximizes expected returns while being regularized
by behavior cloning to prevent out-of-distribution actions.

Key Features:
- Gaussian policy for continuous action spaces
- Behavior cloning regularization
- Support for co-teaching framework
- State-dependent action sampling with importance weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .networks import GaussianPolicy, MLP


class Actor(nn.Module):
    """
    Actor network for IACT algorithm implementing Gaussian policy with BC regularization.
    
    The actor learns a policy π(a|s) that is regularized by the behavior policy π_β(a|s)
    to prevent out-of-distribution actions. It supports importance-weighted training
    and co-teaching sample selection.
    
    Architecture:
    - 3-layer MLP [state_dim, 256, 256, 2*action_dim] 
    - Output: mean and log_std for Gaussian distribution
    - Activation: ReLU for hidden layers, no activation for output
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        max_action: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout: float = 0.0,
        layer_norm: bool = False
    ):
        """
        Initialize Actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            max_action: Maximum action value for scaling
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
        """
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build the policy network using GaussianPolicy from networks.py
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            max_action=max_action,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout=dropout,
            layer_norm=layer_norm
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            mean: Action mean of shape (batch_size, action_dim)
            log_std: Action log standard deviation of shape (batch_size, action_dim)
        """
        return self.policy.forward(state)
    
    def sample(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False,
        with_log_prob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            deterministic: If True, return mean action without noise
            with_log_prob: If True, return log probabilities
            
        Returns:
            action: Sampled actions of shape (batch_size, action_dim)
            log_prob: Log probabilities of shape (batch_size,) if with_log_prob=True
        """
        if with_log_prob:
            return self.policy.sample(state, deterministic)
        else:
            action, _ = self.policy.sample(state, deterministic)
            return action, None
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given actions under current policy.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            log_prob: Log probabilities of shape (batch_size,)
        """
        return self.policy.log_prob(state, action)
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """
        Get the action distribution for given states.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            distribution: Normal distribution over actions
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)
    
    def compute_bc_loss(
        self, 
        state: torch.Tensor, 
        behavior_action: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute behavior cloning loss for regularization.
        
        The BC loss encourages the policy to stay close to the behavior policy
        to prevent out-of-distribution actions:
        L_BC = -E[w(s) * log π(a_β|s)]
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            behavior_action: Behavior policy actions of shape (batch_size, action_dim)
            importance_weights: Optional importance weights of shape (batch_size,)
            
        Returns:
            bc_loss: Behavior cloning loss scalar
        """
        log_prob = self.log_prob(state, behavior_action)
        
        if importance_weights is not None:
            # Weight the BC loss by importance weights
            bc_loss = -(importance_weights * log_prob).mean()
        else:
            bc_loss = -log_prob.mean()
        
        return bc_loss
    
    def compute_policy_loss(
        self,
        state: torch.Tensor,
        q_values: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        bc_regularizer: float = 0.0,
        behavior_action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full policy loss including actor loss and BC regularization.
        
        The total loss is:
        L_total = L_actor + λ_BC * L_BC
        
        where:
        - L_actor = -E[w(s) * Q(s,a)] (importance-weighted policy gradient)
        - L_BC = -E[w(s) * log π(a_β|s)] (behavior cloning regularization)
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            q_values: Q-values for sampled actions of shape (batch_size,)
            importance_weights: Optional importance weights of shape (batch_size,)
            bc_regularizer: Behavior cloning regularization coefficient
            behavior_action: Behavior actions for BC loss of shape (batch_size, action_dim)
            
        Returns:
            loss_dict: Dictionary containing 'total_loss', 'actor_loss', 'bc_loss'
        """
        # Sample actions and compute actor loss
        action, log_prob = self.sample(state, deterministic=False, with_log_prob=True)
        
        if importance_weights is not None:
            # Importance-weighted policy gradient loss
            actor_loss = -(importance_weights * q_values).mean()
        else:
            actor_loss = -q_values.mean()
        
        total_loss = actor_loss
        loss_dict = {
            'actor_loss': actor_loss,
            'total_loss': total_loss
        }
        
        # Add behavior cloning regularization if specified
        if bc_regularizer > 0.0 and behavior_action is not None:
            bc_loss = self.compute_bc_loss(state, behavior_action, importance_weights)
            total_loss = actor_loss + bc_regularizer * bc_loss
            loss_dict.update({
                'bc_loss': bc_loss,
                'total_loss': total_loss
            })
        
        return loss_dict
    
    def get_confidence_scores(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence scores for co-teaching sample selection.
        
        Confidence is measured as the probability density of actions under the current policy.
        Higher confidence indicates the policy is more certain about the action.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            confidence: Confidence scores of shape (batch_size,)
        """
        log_prob = self.log_prob(state, action)
        confidence = torch.exp(log_prob)  # Convert log prob to probability
        return confidence
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions under the current policy for analysis.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            eval_dict: Dictionary with 'log_prob', 'entropy', 'mean', 'std'
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        log_prob = self.log_prob(state, action)
        
        # Compute entropy of the policy
        entropy = 0.5 * (1 + torch.log(2 * np.pi * std**2)).sum(dim=-1)
        
        return {
            'log_prob': log_prob,
            'entropy': entropy,
            'mean': mean,
            'std': std
        }


class DualActor(nn.Module):
    """
    Dual Actor implementation for co-teaching framework.
    
    This class manages two actor networks that teach each other by selecting
    high-quality samples based on importance weights and confidence scores.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        max_action: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout: float = 0.0,
        layer_norm: bool = False
    ):
        """
        Initialize dual actor networks.
        
        Args:
            Same as Actor class
        """
        super(DualActor, self).__init__()
        
        # Create two identical actor networks
        self.actor_a = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            max_action=max_action,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout=dropout,
            layer_norm=layer_norm
        )
        
        self.actor_b = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            max_action=max_action,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout=dropout,
            layer_norm=layer_norm
        )
    
    def select_samples_for_actor(
        self,
        selecting_actor: str,
        state: torch.Tensor,
        action: torch.Tensor,
        importance_weights: torch.Tensor,
        selection_rate: float
    ) -> torch.Tensor:
        """
        Select high-quality samples for training the other actor.
        
        Actor A selects samples for Actor B based on A's confidence and importance weights.
        Actor B selects samples for Actor A based on B's confidence and importance weights.
        
        Args:
            selecting_actor: 'a' or 'b' indicating which actor is selecting
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            importance_weights: Importance weights of shape (batch_size,)
            selection_rate: Fraction of samples to select (0.0 to 1.0)
            
        Returns:
            selected_indices: Indices of selected samples
        """
        # Get the selecting actor
        if selecting_actor.lower() == 'a':
            actor = self.actor_a
        elif selecting_actor.lower() == 'b':
            actor = self.actor_b
        else:
            raise ValueError("selecting_actor must be 'a' or 'b'")
        
        # Compute confidence scores
        confidence_scores = actor.get_confidence_scores(state, action)
        
        # Combine importance weights with confidence scores
        selection_scores = importance_weights * confidence_scores
        
        # Select top-k samples
        batch_size = state.shape[0]
        k = max(1, int(batch_size * selection_rate))
        
        _, selected_indices = torch.topk(selection_scores, k, largest=True)
        
        return selected_indices
    
    def get_actor(self, actor_name: str) -> Actor:
        """Get specific actor by name."""
        if actor_name.lower() == 'a':
            return self.actor_a
        elif actor_name.lower() == 'b':
            return self.actor_b
        else:
            raise ValueError("actor_name must be 'a' or 'b'")
    
    def parameters_a(self):
        """Get parameters of actor A."""
        return self.actor_a.parameters()
    
    def parameters_b(self):
        """Get parameters of actor B."""
        return self.actor_b.parameters()
    
    def state_dict_a(self):
        """Get state dict of actor A."""
        return self.actor_a.state_dict()
    
    def state_dict_b(self):
        """Get state dict of actor B."""
        return self.actor_b.state_dict()
    
    def load_state_dict_a(self, state_dict):
        """Load state dict for actor A."""
        self.actor_a.load_state_dict(state_dict)
    
    def load_state_dict_b(self, state_dict):
        """Load state dict for actor B."""
        self.actor_b.load_state_dict(state_dict)


def create_actor(
    state_dim: int,
    action_dim: int,
    config: Optional[Dict[str, Any]] = None
) -> Actor:
    """
    Factory function to create an Actor with default or custom configuration.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        config: Optional configuration dictionary
        
    Returns:
        actor: Configured Actor instance
    """
    default_config = {
        'hidden_dims': (256, 256),
        'activation': 'relu',
        'max_action': 1.0,
        'log_std_min': -20.0,
        'log_std_max': 2.0,
        'dropout': 0.0,
        'layer_norm': False
    }
    
    if config is not None:
        default_config.update(config)
    
    return Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        **default_config
    )


def create_dual_actor(
    state_dim: int,
    action_dim: int,
    config: Optional[Dict[str, Any]] = None
) -> DualActor:
    """
    Factory function to create a DualActor with default or custom configuration.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        config: Optional configuration dictionary
        
    Returns:
        dual_actor: Configured DualActor instance
    """
    default_config = {
        'hidden_dims': (256, 256),
        'activation': 'relu',
        'max_action': 1.0,
        'log_std_min': -20.0,
        'log_std_max': 2.0,
        'dropout': 0.0,
        'layer_norm': False
    }
    
    if config is not None:
        default_config.update(config)
    
    return DualActor(
        state_dim=state_dim,
        action_dim=action_dim,
        **default_config
    )