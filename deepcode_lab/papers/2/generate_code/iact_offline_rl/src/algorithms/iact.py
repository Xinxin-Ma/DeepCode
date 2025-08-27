"""
IACT (Importance-Aware Co-Teaching) Algorithm Implementation

This module implements the main IACT algorithm that combines:
1. State importance estimation via KLIEP density ratio
2. Co-teaching mechanism between dual policies
3. Behavior cloning regularization
4. Importance-weighted training

Paper: "Importance-Aware Co-Teaching for Offline Reinforcement Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import copy

# Import our custom modules
from ..models.actor import DualActor, create_dual_actor
from ..models.critic import DualCritic, create_dual_critic
from .importance_estimator import ImportanceEstimator, create_importance_estimator
from .sample_selector import SampleSelector, SelectionMetrics

logger = logging.getLogger(__name__)


@dataclass
class IACTConfig:
    """Configuration for IACT algorithm"""
    # Network architecture
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = None
    activation: str = 'relu'
    
    # Training parameters
    batch_size: int = 256
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate
    
    # IACT specific parameters
    bc_regularizer_weight: float = 0.1  # λ_BC in paper
    importance_weight_clip: float = 10.0  # Clip importance weights
    importance_normalization: str = 'mean'  # 'mean' or 'max'
    
    # Co-teaching parameters
    initial_selection_rate: float = 0.8
    final_selection_rate: float = 0.4
    selection_decay_steps: int = 50000
    selection_strategy: str = 'curriculum'  # 'curriculum' or 'fixed'
    
    # Importance estimation parameters
    importance_update_freq: int = 1000  # Update importance weights every N steps
    kliep_n_kernels: int = 100
    kliep_sigma: float = 1.0
    
    # Training parameters
    max_steps: int = 1000000
    eval_freq: int = 5000
    log_freq: int = 1000
    save_freq: int = 10000
    
    # Device
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


@dataclass
class TrainingMetrics:
    """Training metrics for IACT"""
    step: int
    actor_a_loss: float
    actor_b_loss: float
    critic_a_loss: float
    critic_b_loss: float
    bc_loss_a: float
    bc_loss_b: float
    importance_weight_mean: float
    importance_weight_std: float
    selection_rate: float
    selected_samples_a: int
    selected_samples_b: int
    q_value_mean: float
    policy_entropy: float


class IACTAlgorithm:
    """
    Main IACT Algorithm Implementation
    
    Combines importance-aware sample weighting with co-teaching between dual policies
    to address distribution shift and data quality issues in offline RL.
    """
    
    def __init__(self, config: IACTConfig):
        """
        Initialize IACT algorithm
        
        Args:
            config: IACT configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.step = 0
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize importance estimator
        self.importance_estimator = create_importance_estimator({
            'n_kernels': config.kliep_n_kernels,
            'sigma': config.kliep_sigma,
            'device': config.device
        })
        
        # Initialize sample selector
        self.sample_selector = SampleSelector(
            scheduler_config={
                'initial_rate': config.initial_selection_rate,
                'final_rate': config.final_selection_rate,
                'decay_steps': config.selection_decay_steps,
                'strategy': config.selection_strategy
            },
            device=config.device
        )
        
        # Training state
        self.importance_weights = None
        self.last_importance_update = 0
        self.training_metrics = []
        
        logger.info(f"Initialized IACT algorithm with config: {config}")
    
    def _init_networks(self):
        """Initialize actor and critic networks"""
        # Create dual actors
        self.dual_actor = create_dual_actor(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            config={
                'hidden_dims': self.config.hidden_dims,
                'activation': self.config.activation,
                'device': self.config.device
            }
        )
        
        # Create dual critics
        self.dual_critic = create_dual_critic(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            config={
                'hidden_dims': self.config.hidden_dims,
                'activation': self.config.activation,
                'gamma': self.config.gamma,
                'tau': self.config.tau,
                'device': self.config.device
            }
        )
        
        # Move to device
        self.dual_actor.to(self.device)
        self.dual_critic.to(self.device)
    
    def _init_optimizers(self):
        """Initialize optimizers for all networks"""
        self.actor_a_optimizer = torch.optim.Adam(
            self.dual_actor.actor_a.parameters(),
            lr=self.config.learning_rate_actor
        )
        self.actor_b_optimizer = torch.optim.Adam(
            self.dual_actor.actor_b.parameters(),
            lr=self.config.learning_rate_actor
        )
        self.critic_a_optimizer = torch.optim.Adam(
            self.dual_critic.critic_a.parameters(),
            lr=self.config.learning_rate_critic
        )
        self.critic_b_optimizer = torch.optim.Adam(
            self.dual_critic.critic_b.parameters(),
            lr=self.config.learning_rate_critic
        )
    
    def update_importance_weights(self, states: torch.Tensor, behavior_states: torch.Tensor):
        """
        Update importance weights using KLIEP
        
        Args:
            states: Current policy states
            behavior_states: Behavior policy states from dataset
        """
        logger.info("Updating importance weights using KLIEP...")
        
        # Fit importance estimator
        self.importance_estimator.fit(states, behavior_states)
        
        # Estimate importance weights
        raw_weights = self.importance_estimator.estimate_weights(states)
        
        # Normalize and clip weights
        if self.config.importance_normalization == 'mean':
            normalized_weights = raw_weights / (raw_weights.mean() + 1e-8)
        elif self.config.importance_normalization == 'max':
            normalized_weights = raw_weights / (raw_weights.max() + 1e-8)
        else:
            normalized_weights = raw_weights
        
        # Clip weights to prevent extreme values
        self.importance_weights = torch.clamp(
            normalized_weights, 
            min=1.0/self.config.importance_weight_clip,
            max=self.config.importance_weight_clip
        )
        
        self.last_importance_update = self.step
        
        logger.info(f"Updated importance weights - Mean: {self.importance_weights.mean():.4f}, "
                   f"Std: {self.importance_weights.std():.4f}, "
                   f"Min: {self.importance_weights.min():.4f}, "
                   f"Max: {self.importance_weights.max():.4f}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """
        Perform one training step of IACT algorithm
        
        Args:
            batch: Training batch containing states, actions, rewards, next_states, dones
            
        Returns:
            Training metrics
        """
        self.step += 1
        
        # Extract batch data
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        batch_size = states.shape[0]
        
        # Update importance weights if needed
        if (self.importance_weights is None or 
            self.step - self.last_importance_update >= self.config.importance_update_freq):
            # Use current states as policy states and all states as behavior states
            # In practice, you might want to maintain separate distributions
            self.update_importance_weights(states, states)
        
        # Get importance weights for current batch
        if self.importance_weights is not None:
            # Sample importance weights for current batch
            batch_importance_weights = self.importance_weights[:batch_size]
        else:
            # Use uniform weights if importance weights not available
            batch_importance_weights = torch.ones(batch_size, device=self.device)
        
        # Perform co-teaching sample selection
        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        batch_for_a, batch_for_b, selection_metrics = self.sample_selector.co_teaching_selection(
            policy_a=self.dual_actor.actor_a,
            policy_b=self.dual_actor.actor_b,
            batch=batch_dict,
            importance_weights=batch_importance_weights,
            step=self.step
        )
        
        # Train Actor A and Critic A on samples selected by Actor B
        metrics_a = self._train_actor_critic_pair(
            actor=self.dual_actor.actor_a,
            critic=self.dual_critic.critic_a,
            actor_optimizer=self.actor_a_optimizer,
            critic_optimizer=self.critic_a_optimizer,
            batch=batch_for_a,
            importance_weights=batch_importance_weights[:len(batch_for_a['states'])],
            behavior_actions=batch_for_a['actions'],
            actor_name='A'
        )
        
        # Train Actor B and Critic B on samples selected by Actor A
        metrics_b = self._train_actor_critic_pair(
            actor=self.dual_actor.actor_b,
            critic=self.dual_critic.critic_b,
            actor_optimizer=self.actor_b_optimizer,
            critic_optimizer=self.critic_b_optimizer,
            batch=batch_for_b,
            importance_weights=batch_importance_weights[:len(batch_for_b['states'])],
            behavior_actions=batch_for_b['actions'],
            actor_name='B'
        )
        
        # Update target networks
        self.dual_critic.update_target_networks()
        
        # Compute additional metrics
        with torch.no_grad():
            # Policy entropy
            actions_a, log_probs_a = self.dual_actor.actor_a.sample(states, with_log_prob=True)
            policy_entropy = -log_probs_a.mean().item()
            
            # Q-value statistics
            q_values_a = self.dual_critic.critic_a(states, actions)
            q_value_mean = q_values_a.mean().item()
        
        # Create training metrics
        training_metrics = TrainingMetrics(
            step=self.step,
            actor_a_loss=metrics_a['actor_loss'],
            actor_b_loss=metrics_b['actor_loss'],
            critic_a_loss=metrics_a['critic_loss'],
            critic_b_loss=metrics_b['critic_loss'],
            bc_loss_a=metrics_a['bc_loss'],
            bc_loss_b=metrics_b['bc_loss'],
            importance_weight_mean=batch_importance_weights.mean().item(),
            importance_weight_std=batch_importance_weights.std().item(),
            selection_rate=selection_metrics.selection_rate,
            selected_samples_a=len(batch_for_a['states']),
            selected_samples_b=len(batch_for_b['states']),
            q_value_mean=q_value_mean,
            policy_entropy=policy_entropy
        )
        
        self.training_metrics.append(training_metrics)
        
        return training_metrics
    
    def _train_actor_critic_pair(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        importance_weights: torch.Tensor,
        behavior_actions: torch.Tensor,
        actor_name: str
    ) -> Dict[str, float]:
        """
        Train a single actor-critic pair
        
        Args:
            actor: Actor network
            critic: Critic network
            actor_optimizer: Actor optimizer
            critic_optimizer: Critic optimizer
            batch: Training batch
            importance_weights: Importance weights for samples
            behavior_actions: Behavior policy actions for BC regularization
            actor_name: Name of actor ('A' or 'B')
            
        Returns:
            Training metrics dictionary
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Train Critic
        critic_optimizer.zero_grad()
        
        # Get next actions from current actor
        with torch.no_grad():
            next_actions, _ = actor.sample(next_states)
        
        # Compute critic loss
        critic_loss_dict = critic.compute_td_loss(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            next_action=next_actions,
            done=dones,
            importance_weights=importance_weights
        )
        
        critic_loss = critic_loss_dict['total_loss']
        critic_loss.backward()
        critic_optimizer.step()
        
        # Train Actor
        actor_optimizer.zero_grad()
        
        # Sample actions from current actor
        actor_actions, log_probs = actor.sample(states, with_log_prob=True)
        
        # Get Q-values for actor actions
        q_values = critic(states, actor_actions)
        
        # Compute actor loss (negative Q-value with importance weighting)
        actor_loss = -(importance_weights * q_values).mean()
        
        # Add behavior cloning regularization
        bc_loss = actor.compute_bc_loss(
            state=states,
            behavior_action=behavior_actions,
            importance_weights=importance_weights
        )
        
        total_actor_loss = actor_loss + self.config.bc_regularizer_weight * bc_loss
        total_actor_loss.backward()
        actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'bc_loss': bc_loss.item(),
            'total_actor_loss': total_actor_loss.item()
        }
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select action using the trained policy
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            # Use Actor A for action selection (could also ensemble both actors)
            action, _ = self.dual_actor.actor_a.sample(
                state.unsqueeze(0) if state.dim() == 1 else state,
                deterministic=deterministic
            )
            return action.squeeze(0) if state.dim() == 1 else action
    
    def evaluate_policy(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy
        
        Args:
            env: Environment for evaluation
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle gym environments that return (obs, info)
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Convert state to tensor
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                
                # Select action
                action = self.select_action(state_tensor, deterministic=True)
                
                # Convert action to numpy for environment
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                state = next_state
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_return': np.min(episode_returns),
            'max_return': np.max(episode_returns)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'config': self.config,
            'dual_actor_state_dict': self.dual_actor.state_dict(),
            'dual_critic_state_dict': self.dual_critic.state_dict(),
            'actor_a_optimizer_state_dict': self.actor_a_optimizer.state_dict(),
            'actor_b_optimizer_state_dict': self.actor_b_optimizer.state_dict(),
            'critic_a_optimizer_state_dict': self.critic_a_optimizer.state_dict(),
            'critic_b_optimizer_state_dict': self.critic_b_optimizer.state_dict(),
            'importance_weights': self.importance_weights,
            'training_metrics': self.training_metrics
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step = checkpoint['step']
        self.dual_actor.load_state_dict(checkpoint['dual_actor_state_dict'])
        self.dual_critic.load_state_dict(checkpoint['dual_critic_state_dict'])
        self.actor_a_optimizer.load_state_dict(checkpoint['actor_a_optimizer_state_dict'])
        self.actor_b_optimizer.load_state_dict(checkpoint['actor_b_optimizer_state_dict'])
        self.critic_a_optimizer.load_state_dict(checkpoint['critic_a_optimizer_state_dict'])
        self.critic_b_optimizer.load_state_dict(checkpoint['critic_b_optimizer_state_dict'])
        self.importance_weights = checkpoint.get('importance_weights')
        self.training_metrics = checkpoint.get('training_metrics', [])
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_metrics:
            return {}
        
        recent_metrics = self.training_metrics[-100:]  # Last 100 steps
        
        stats = {
            'total_steps': self.step,
            'recent_actor_a_loss': np.mean([m.actor_a_loss for m in recent_metrics]),
            'recent_actor_b_loss': np.mean([m.actor_b_loss for m in recent_metrics]),
            'recent_critic_a_loss': np.mean([m.critic_a_loss for m in recent_metrics]),
            'recent_critic_b_loss': np.mean([m.critic_b_loss for m in recent_metrics]),
            'recent_bc_loss_a': np.mean([m.bc_loss_a for m in recent_metrics]),
            'recent_bc_loss_b': np.mean([m.bc_loss_b for m in recent_metrics]),
            'recent_importance_weight_mean': np.mean([m.importance_weight_mean for m in recent_metrics]),
            'recent_selection_rate': np.mean([m.selection_rate for m in recent_metrics]),
            'recent_q_value_mean': np.mean([m.q_value_mean for m in recent_metrics]),
            'recent_policy_entropy': np.mean([m.policy_entropy for m in recent_metrics])
        }
        
        return stats


def create_iact_algorithm(config: Union[Dict, IACTConfig]) -> IACTAlgorithm:
    """
    Factory function to create IACT algorithm
    
    Args:
        config: Configuration dictionary or IACTConfig object
        
    Returns:
        IACT algorithm instance
    """
    if isinstance(config, dict):
        config = IACTConfig(**config)
    
    return IACTAlgorithm(config)


# Example usage and testing functions
def test_iact_algorithm():
    """Test IACT algorithm with dummy data"""
    # Create test configuration
    config = IACTConfig(
        state_dim=4,
        action_dim=2,
        batch_size=32,
        device='cpu'
    )
    
    # Create algorithm
    algorithm = create_iact_algorithm(config)
    
    # Create dummy batch
    batch_size = 32
    batch = {
        'states': torch.randn(batch_size, 4),
        'actions': torch.randn(batch_size, 2),
        'rewards': torch.randn(batch_size, 1),
        'next_states': torch.randn(batch_size, 4),
        'dones': torch.zeros(batch_size, 1)
    }
    
    # Perform training step
    metrics = algorithm.train_step(batch)
    
    print("IACT Algorithm Test Results:")
    print(f"Step: {metrics.step}")
    print(f"Actor A Loss: {metrics.actor_a_loss:.4f}")
    print(f"Actor B Loss: {metrics.actor_b_loss:.4f}")
    print(f"Critic A Loss: {metrics.critic_a_loss:.4f}")
    print(f"Critic B Loss: {metrics.critic_b_loss:.4f}")
    print(f"BC Loss A: {metrics.bc_loss_a:.4f}")
    print(f"BC Loss B: {metrics.bc_loss_b:.4f}")
    print(f"Importance Weight Mean: {metrics.importance_weight_mean:.4f}")
    print(f"Selection Rate: {metrics.selection_rate:.4f}")
    print(f"Selected Samples A: {metrics.selected_samples_a}")
    print(f"Selected Samples B: {metrics.selected_samples_b}")
    
    # Test action selection
    test_state = torch.randn(4)
    action = algorithm.select_action(test_state)
    print(f"Test action shape: {action.shape}")
    
    # Test training stats
    stats = algorithm.get_training_stats()
    print(f"Training stats keys: {list(stats.keys())}")
    
    return True


if __name__ == "__main__":
    # Run test
    test_iact_algorithm()
    print("IACT algorithm implementation completed successfully!")