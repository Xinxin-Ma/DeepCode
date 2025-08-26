"""
ACB-Agent Policy Network Implementation

This module implements the policy network component of the ACB-Agent that maps
concept activations to action probabilities. The key constraint is that policy
decisions are based ONLY on concept activations, not raw state features.

Mathematical formulation: π_θ(a|s) = π_θ(a|c) where c = CB_φ(s) (Equation 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class ACBPolicyNetwork(nn.Module):
    """
    ACB Policy Network that maps concept activations to action probabilities.
    
    Architecture:
    - Input: concept_activations [batch, concept_dim]
    - Hidden layers: Linear(concept_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU
    - Output: Linear(128, action_dim) -> Softmax
    
    Key constraint: Policy decisions based ONLY on concept activations, not raw state
    """
    
    def __init__(self, 
                 concept_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 2,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 use_layer_norm: bool = False):
        """
        Initialize ACB Policy Network.
        
        Args:
            concept_dim: Dimension of concept activation space
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'tanh', 'gelu')
            use_layer_norm: Whether to use layer normalization
        """
        super(ACBPolicyNetwork, self).__init__()
        
        self.concept_dim = concept_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Activation function selection
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(concept_dim, hidden_dim))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # Initialize output layer with smaller weights for stable training
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: concept activations -> action probabilities.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
                                Concept activations from concept bottleneck layer
        
        Returns:
            action_probs: Tensor of shape [batch_size, action_dim]
                         Action probability distribution
        """
        if concept_activations.dim() != 2:
            raise ValueError(f"Expected 2D input, got {concept_activations.dim()}D")
        
        if concept_activations.size(1) != self.concept_dim:
            raise ValueError(f"Expected concept_dim={self.concept_dim}, got {concept_activations.size(1)}")
        
        x = concept_activations
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            x = self.dropouts[i](x)
        
        # Output layer
        logits = self.output_layer(x)
        
        # Apply softmax to get action probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs
    
    def get_action_logits(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Get raw action logits (before softmax).
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
        
        Returns:
            logits: Tensor of shape [batch_size, action_dim]
        """
        x = concept_activations
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        logits = self.output_layer(x)
        return logits
    
    def sample_action(self, concept_activations: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
            deterministic: If True, select action with highest probability
        
        Returns:
            actions: Tensor of shape [batch_size] with sampled actions
            log_probs: Tensor of shape [batch_size] with log probabilities
        """
        action_probs = self.forward(concept_activations)
        
        if deterministic:
            # Select action with highest probability
            actions = torch.argmax(action_probs, dim=-1)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    def get_action_log_probs(self, concept_activations: torch.Tensor, 
                           actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given actions.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
            actions: Tensor of shape [batch_size] with action indices
        
        Returns:
            log_probs: Tensor of shape [batch_size] with log probabilities
        """
        action_probs = self.forward(concept_activations)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
        return log_probs
    
    def get_entropy(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distribution for exploration bonus.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
        
        Returns:
            entropy: Tensor of shape [batch_size] with entropy values
        """
        action_probs = self.forward(concept_activations)
        log_probs = torch.log(action_probs + 1e-8)  # Add small epsilon for numerical stability
        entropy = -torch.sum(action_probs * log_probs, dim=-1)
        return entropy


class ValueNetwork(nn.Module):
    """
    Value network for advantage estimation in policy gradient methods.
    Takes concept activations as input and outputs state value estimates.
    """
    
    def __init__(self, 
                 concept_dim: int,
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 2,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize Value Network.
        
        Args:
            concept_dim: Dimension of concept activation space
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            dropout_rate: Dropout rate for regularization
            activation: Activation function
        """
        super(ValueNetwork, self).__init__()
        
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(concept_dim, hidden_dim))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer (single value)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: concept activations -> state value.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
        
        Returns:
            values: Tensor of shape [batch_size, 1] with state values
        """
        x = concept_activations
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        # Output layer
        values = self.output_layer(x)
        
        return values


class ActorCriticACB(nn.Module):
    """
    Combined Actor-Critic network for ACB-Agent.
    Shares concept processing but has separate policy and value heads.
    """
    
    def __init__(self,
                 concept_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 2,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 use_layer_norm: bool = False):
        """
        Initialize Actor-Critic ACB network.
        
        Args:
            concept_dim: Dimension of concept activation space
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            dropout_rate: Dropout rate
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
        """
        super(ActorCriticACB, self).__init__()
        
        self.concept_dim = concept_dim
        self.action_dim = action_dim
        
        # Policy network (actor)
        self.policy_net = ACBPolicyNetwork(
            concept_dim=concept_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_layer_norm=use_layer_norm
        )
        
        # Value network (critic)
        self.value_net = ValueNetwork(
            concept_dim=concept_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation
        )
    
    def forward(self, concept_activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both policy and value networks.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
        
        Returns:
            action_probs: Tensor of shape [batch_size, action_dim]
            values: Tensor of shape [batch_size, 1]
        """
        action_probs = self.policy_net(concept_activations)
        values = self.value_net(concept_activations)
        
        return action_probs, values
    
    def act(self, concept_activations: torch.Tensor, 
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action and compute value estimate.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
            deterministic: Whether to select deterministic action
        
        Returns:
            actions: Selected actions
            log_probs: Log probabilities of selected actions
            values: State value estimates
        """
        actions, log_probs = self.policy_net.sample_action(concept_activations, deterministic)
        values = self.value_net(concept_activations)
        
        return actions, log_probs, values
    
    def evaluate_actions(self, concept_activations: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for policy gradient updates.
        
        Args:
            concept_activations: Tensor of shape [batch_size, concept_dim]
            actions: Tensor of shape [batch_size] with action indices
        
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        log_probs = self.policy_net.get_action_log_probs(concept_activations, actions)
        values = self.value_net(concept_activations)
        entropy = self.policy_net.get_entropy(concept_activations)
        
        return log_probs, values, entropy


def create_acb_policy_network(concept_dim: int, 
                             action_dim: int,
                             network_type: str = 'policy',
                             **kwargs) -> nn.Module:
    """
    Factory function to create ACB policy networks.
    
    Args:
        concept_dim: Dimension of concept space
        action_dim: Number of actions
        network_type: Type of network ('policy', 'value', 'actor_critic')
        **kwargs: Additional network parameters
    
    Returns:
        network: Initialized network module
    """
    if network_type == 'policy':
        return ACBPolicyNetwork(concept_dim, action_dim, **kwargs)
    elif network_type == 'value':
        return ValueNetwork(concept_dim, **kwargs)
    elif network_type == 'actor_critic':
        return ActorCriticACB(concept_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# Example usage and testing functions
if __name__ == "__main__":
    # Test ACB Policy Network
    concept_dim = 64
    action_dim = 4
    batch_size = 32
    
    # Create policy network
    policy_net = ACBPolicyNetwork(concept_dim, action_dim)
    
    # Test forward pass
    concept_activations = torch.randn(batch_size, concept_dim)
    action_probs = policy_net(concept_activations)
    
    print(f"Policy Network Test:")
    print(f"Input shape: {concept_activations.shape}")
    print(f"Output shape: {action_probs.shape}")
    print(f"Action probabilities sum: {action_probs.sum(dim=-1).mean():.4f}")
    
    # Test action sampling
    actions, log_probs = policy_net.sample_action(concept_activations)
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    
    # Test Actor-Critic network
    actor_critic = ActorCriticACB(concept_dim, action_dim)
    action_probs, values = actor_critic(concept_activations)
    
    print(f"\nActor-Critic Test:")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Values shape: {values.shape}")
    
    print("\nAll tests passed!")