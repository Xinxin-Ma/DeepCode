"""
Critic Networks for IACT Offline Reinforcement Learning

This module implements the critic networks for the IACT algorithm, including:
- Double Q-learning critics to reduce overestimation bias
- Target networks with soft updates
- Importance-weighted Q-learning updates
- Co-teaching sample selection for critic training
- Integration with the IACT framework

Key Components:
- Critic: Single Q-network with importance weighting
- DualCritic: Double Q-learning implementation
- Target network management
- Loss computation with importance weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .networks import QNetwork, DoubleQNetwork, soft_update, hard_update


class Critic(nn.Module):
    """
    Single Q-network critic with importance weighting support.
    
    This critic implements:
    - Q-value estimation for state-action pairs
    - Importance-weighted TD learning
    - Target network with soft updates
    - Loss computation for offline RL
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None
    ):
        """
        Initialize the Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary with hyperparameters
        """
        super(Critic, self).__init__()
        
        # Default configuration
        default_config = {
            'hidden_dims': [256, 256],
            'activation': 'relu',
            'dropout': 0.0,
            'layer_norm': False,
            'tau': 0.005,  # Target network update rate
            'gamma': 0.99,  # Discount factor
            'target_update_freq': 1,  # Target update frequency
        }
        
        self.config = {**default_config, **(config or {})}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.target_update_freq = self.config['target_update_freq']
        
        # Main Q-network
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation'],
            dropout=self.config['dropout'],
            layer_norm=self.config['layer_norm']
        )
        
        # Target Q-network
        self.target_q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation'],
            dropout=self.config['dropout'],
            layer_norm=self.config['layer_norm']
        )
        
        # Initialize target network with same weights
        hard_update(self.target_q_network, self.q_network)
        
        # Training step counter
        self.update_step = 0
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Q-values [batch_size, 1]
        """
        return self.q_network(state, action)
    
    def target_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the target Q-network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Target Q-values [batch_size, 1]
        """
        with torch.no_grad():
            return self.target_q_network(state, action)
    
    def compute_td_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        done: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal difference loss with importance weighting.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Current action [batch_size, action_dim]
            reward: Reward [batch_size, 1]
            next_state: Next state [batch_size, state_dim]
            next_action: Next action [batch_size, action_dim]
            done: Done flag [batch_size, 1]
            importance_weights: Importance weights [batch_size, 1]
            
        Returns:
            Dictionary containing loss components
        """
        # Current Q-values
        current_q = self.forward(state, action)
        
        # Target Q-values
        with torch.no_grad():
            target_q = self.target_forward(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # TD error
        td_error = current_q - target_q
        
        # Importance-weighted loss
        if importance_weights is not None:
            # Normalize importance weights
            importance_weights = importance_weights / importance_weights.mean()
            td_loss = (importance_weights * td_error.pow(2)).mean()
        else:
            td_loss = F.mse_loss(current_q, target_q)
        
        return {
            'td_loss': td_loss,
            'current_q': current_q.mean(),
            'target_q': target_q.mean(),
            'td_error': td_error.abs().mean()
        }
    
    def update_target_network(self):
        """Update target network using soft update."""
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            soft_update(self.target_q_network, self.q_network, self.tau)
    
    def get_q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for given state-action pairs.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Q-values [batch_size, 1]
        """
        return self.forward(state, action)


class DualCritic(nn.Module):
    """
    Double Q-learning critic implementation for IACT.
    
    This critic implements:
    - Double Q-learning to reduce overestimation bias
    - Importance-weighted updates for both critics
    - Co-teaching sample selection between critics
    - Target networks with soft updates
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None
    ):
        """
        Initialize the DualCritic with two Q-networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary with hyperparameters
        """
        super(DualCritic, self).__init__()
        
        # Default configuration
        default_config = {
            'hidden_dims': [256, 256],
            'activation': 'relu',
            'dropout': 0.0,
            'layer_norm': False,
            'tau': 0.005,
            'gamma': 0.99,
            'target_update_freq': 1,
            'co_teaching_selection_rate': 0.7,  # Initial selection rate for co-teaching
            'selection_rate_decay': 0.99,  # Decay rate for selection rate
            'min_selection_rate': 0.3,  # Minimum selection rate
        }
        
        self.config = {**default_config, **(config or {})}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.target_update_freq = self.config['target_update_freq']
        
        # Co-teaching parameters
        self.selection_rate = self.config['co_teaching_selection_rate']
        self.selection_rate_decay = self.config['selection_rate_decay']
        self.min_selection_rate = self.config['min_selection_rate']
        
        # Double Q-networks
        self.q1_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation'],
            dropout=self.config['dropout'],
            layer_norm=self.config['layer_norm']
        )
        
        self.q2_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation'],
            dropout=self.config['dropout'],
            layer_norm=self.config['layer_norm']
        )
        
        # Target networks
        self.target_q1_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation'],
            dropout=self.config['dropout'],
            layer_norm=self.config['layer_norm']
        )
        
        self.target_q2_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation'],
            dropout=self.config['dropout'],
            layer_norm=self.config['layer_norm']
        )
        
        # Initialize target networks
        hard_update(self.target_q1_network, self.q1_network)
        hard_update(self.target_q2_network, self.q2_network)
        
        # Training step counter
        self.update_step = 0
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Tuple of Q-values from both networks [batch_size, 1]
        """
        q1 = self.q1_network(state, action)
        q2 = self.q2_network(state, action)
        return q1, q2
    
    def target_forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both target Q-networks.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Tuple of target Q-values from both networks [batch_size, 1]
        """
        with torch.no_grad():
            target_q1 = self.target_q1_network(state, action)
            target_q2 = self.target_q2_network(state, action)
            return target_q1, target_q2
    
    def compute_double_q_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        done: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute double Q-learning loss with importance weighting.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Current action [batch_size, action_dim]
            reward: Reward [batch_size, 1]
            next_state: Next state [batch_size, state_dim]
            next_action: Next action [batch_size, action_dim]
            done: Done flag [batch_size, 1]
            importance_weights: Importance weights [batch_size, 1]
            
        Returns:
            Dictionary containing loss components
        """
        # Current Q-values
        current_q1, current_q2 = self.forward(state, action)
        
        # Target Q-values using double Q-learning
        with torch.no_grad():
            target_q1, target_q2 = self.target_forward(next_state, next_action)
            # Use minimum of both target Q-values to reduce overestimation
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # TD errors
        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q
        
        # Importance-weighted losses
        if importance_weights is not None:
            # Normalize importance weights
            importance_weights = importance_weights / importance_weights.mean()
            q1_loss = (importance_weights * td_error1.pow(2)).mean()
            q2_loss = (importance_weights * td_error2.pow(2)).mean()
        else:
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)
        
        total_loss = q1_loss + q2_loss
        
        return {
            'total_loss': total_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'current_q1': current_q1.mean(),
            'current_q2': current_q2.mean(),
            'target_q': target_q.mean(),
            'td_error1': td_error1.abs().mean(),
            'td_error2': td_error2.abs().mean()
        }
    
    def select_samples_for_critic(
        self,
        selecting_critic: str,
        state: torch.Tensor,
        action: torch.Tensor,
        importance_weights: torch.Tensor,
        selection_rate: Optional[float] = None
    ) -> torch.Tensor:
        """
        Select samples for co-teaching between critics.
        
        Args:
            selecting_critic: Which critic is selecting ('q1' or 'q2')
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            importance_weights: Importance weights [batch_size, 1]
            selection_rate: Override selection rate
            
        Returns:
            Selected sample indices
        """
        if selection_rate is None:
            selection_rate = self.selection_rate
        
        with torch.no_grad():
            if selecting_critic == 'q1':
                q_values = self.q1_network(state, action)
            elif selecting_critic == 'q2':
                q_values = self.q2_network(state, action)
            else:
                raise ValueError(f"Invalid selecting_critic: {selecting_critic}")
            
            # Compute selection scores: importance_weight * |Q-value|
            # Higher Q-values indicate more confident predictions
            q_confidence = torch.abs(q_values)
            selection_scores = importance_weights.squeeze(-1) * q_confidence.squeeze(-1)
            
            # Select top-k samples
            batch_size = state.size(0)
            k = max(1, int(batch_size * selection_rate))
            
            _, selected_indices = torch.topk(selection_scores, k, dim=0)
            
        return selected_indices
    
    def update_target_networks(self):
        """Update both target networks using soft update."""
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            soft_update(self.target_q1_network, self.q1_network, self.tau)
            soft_update(self.target_q2_network, self.q2_network, self.tau)
    
    def update_selection_rate(self):
        """Update co-teaching selection rate with decay."""
        self.selection_rate = max(
            self.min_selection_rate,
            self.selection_rate * self.selection_rate_decay
        )
    
    def get_q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values using the minimum of both networks (conservative estimate).
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Conservative Q-values [batch_size, 1]
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
    
    def get_individual_q_values(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Q-values from both networks individually.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Tuple of Q-values from both networks
        """
        return self.forward(state, action)


def create_critic(
    state_dim: int,
    action_dim: int,
    config: Optional[Dict] = None
) -> Critic:
    """
    Factory function to create a Critic instance.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        config: Configuration dictionary
        
    Returns:
        Configured Critic instance
    """
    return Critic(state_dim, action_dim, config)


def create_dual_critic(
    state_dim: int,
    action_dim: int,
    config: Optional[Dict] = None
) -> DualCritic:
    """
    Factory function to create a DualCritic instance.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        config: Configuration dictionary
        
    Returns:
        Configured DualCritic instance
    """
    return DualCritic(state_dim, action_dim, config)


# Example usage and testing
if __name__ == "__main__":
    # Test the critic implementations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters
    state_dim = 17  # Example: HalfCheetah state dimension
    action_dim = 6  # Example: HalfCheetah action dimension
    batch_size = 256
    
    print("Testing Critic implementations...")
    
    # Create test data
    state = torch.randn(batch_size, state_dim).to(device)
    action = torch.randn(batch_size, action_dim).to(device)
    reward = torch.randn(batch_size, 1).to(device)
    next_state = torch.randn(batch_size, state_dim).to(device)
    next_action = torch.randn(batch_size, action_dim).to(device)
    done = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    importance_weights = torch.rand(batch_size, 1).to(device) + 0.5  # Avoid zero weights
    
    # Test Single Critic
    print("\n=== Testing Single Critic ===")
    critic = create_critic(state_dim, action_dim).to(device)
    
    # Forward pass
    q_values = critic(state, action)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values mean: {q_values.mean().item():.4f}")
    
    # Compute loss
    loss_dict = critic.compute_td_loss(
        state, action, reward, next_state, next_action, done, importance_weights
    )
    print(f"TD Loss: {loss_dict['td_loss'].item():.4f}")
    print(f"Current Q mean: {loss_dict['current_q'].item():.4f}")
    print(f"Target Q mean: {loss_dict['target_q'].item():.4f}")
    
    # Test Dual Critic
    print("\n=== Testing Dual Critic ===")
    dual_critic = create_dual_critic(state_dim, action_dim).to(device)
    
    # Forward pass
    q1_values, q2_values = dual_critic(state, action)
    print(f"Q1 values shape: {q1_values.shape}")
    print(f"Q2 values shape: {q2_values.shape}")
    print(f"Q1 values mean: {q1_values.mean().item():.4f}")
    print(f"Q2 values mean: {q2_values.mean().item():.4f}")
    
    # Compute double Q loss
    loss_dict = dual_critic.compute_double_q_loss(
        state, action, reward, next_state, next_action, done, importance_weights
    )
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Q1 Loss: {loss_dict['q1_loss'].item():.4f}")
    print(f"Q2 Loss: {loss_dict['q2_loss'].item():.4f}")
    
    # Test co-teaching sample selection
    selected_indices = dual_critic.select_samples_for_critic(
        'q1', state, action, importance_weights, selection_rate=0.5
    )
    print(f"Selected samples: {len(selected_indices)} out of {batch_size}")
    print(f"Selection rate: {len(selected_indices) / batch_size:.2f}")
    
    # Test target network updates
    dual_critic.update_target_networks()
    dual_critic.update_selection_rate()
    print(f"Updated selection rate: {dual_critic.selection_rate:.4f}")
    
    print("\n✅ All critic tests passed!")