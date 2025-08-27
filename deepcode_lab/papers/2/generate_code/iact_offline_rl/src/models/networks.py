"""
Base neural network components for IACT offline RL implementation.
Provides common network architectures and utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
        layer_norm: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        
        # Activation functions
        self.activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU
        }
        
        if activation not in self.activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if layer_norm and i < len(dims) - 2:  # No layer norm on output
                self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
        
        # Activation functions
        self.activation = self.activations[activation]()
        self.output_activation = None
        if output_activation is not None:
            if output_activation not in self.activations:
                raise ValueError(f"Unsupported output activation: {output_activation}")
            self.output_activation = self.activations[output_activation]()
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            # Apply layer normalization
            if self.layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        # Apply output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


class GaussianPolicy(nn.Module):
    """Gaussian policy network for continuous action spaces."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared backbone
        self.backbone = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize output layers
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and log_std of the policy distribution.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            mean: Action mean of shape (batch_size, action_dim)
            log_std: Action log standard deviation of shape (batch_size, action_dim)
        """
        features = self.backbone(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)
            
            # Compute log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given actions.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            log_prob: Log probability of actions
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Inverse tanh to get pre-tanh action
        action_clamped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        x_t = torch.atanh(action_clamped)
        
        # Compute log probability
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob


class QNetwork(nn.Module):
    """Q-function network for continuous action spaces."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False
    ):
        super().__init__()
        
        self.q_network = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            q_value: Q-value tensor of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.q_network(x)


class DoubleQNetwork(nn.Module):
    """Double Q-network for reduced overestimation bias."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False
    ):
        super().__init__()
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims, activation, dropout, layer_norm)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims, activation, dropout, layer_norm)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q1_value: Q-value from first network
            q2_value: Q-value from second network
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through first Q-network only."""
        return self.q1(state, action)


def create_target_network(network: nn.Module) -> nn.Module:
    """Create a target network as a copy of the given network."""
    target_network = type(network)(**network.__dict__)
    target_network.load_state_dict(network.state_dict())
    
    # Freeze target network parameters
    for param in target_network.parameters():
        param.requires_grad = False
    
    return target_network


def soft_update(target_network: nn.Module, source_network: nn.Module, tau: float):
    """
    Soft update target network parameters using Polyak averaging.
    
    Args:
        target_network: Target network to update
        source_network: Source network to copy from
        tau: Interpolation parameter (0 = no update, 1 = full copy)
    """
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_network: nn.Module, source_network: nn.Module):
    """Hard update target network by copying all parameters."""
    target_network.load_state_dict(source_network.state_dict())