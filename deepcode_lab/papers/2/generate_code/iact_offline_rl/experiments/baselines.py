"""
Baseline Methods Implementation for IACT Offline RL Comparison

This module implements various baseline algorithms for comparison with IACT:
- Behavior Cloning (BC)
- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Advantage Weighted Actor-Critic (AWAC)
- Batch-Constrained Deep Q-Learning (BCQ)

Author: IACT Implementation Team
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.networks import MLP
from models.actor import Actor, create_actor
from models.critic import Critic, create_critic
from data.d4rl_loader import load_d4rl_dataset
from data.replay_buffer import ReplayBuffer
from utils.metrics import MetricsTracker, create_metrics_tracker
from configs.iact_config import get_iact_config
from configs.env_configs import get_env_config


@dataclass
class BaselineConfig:
    """Configuration for baseline algorithms"""
    # Environment
    state_dim: int
    action_dim: int
    max_action: float = 1.0
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = 'relu'
    
    # Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    max_epochs: int = 1000
    eval_freq: int = 50
    
    # Algorithm specific
    gamma: float = 0.99
    tau: float = 0.005
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class BehaviorCloning:
    """
    Behavior Cloning (BC) baseline implementation
    Simple supervised learning on state-action pairs
    """
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create policy network
        self.policy = Actor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation,
                'max_action': config.max_action
            }
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Metrics
        self.training_metrics = defaultdict(list)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for behavior cloning"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        
        # Forward pass
        mean, log_std = self.policy(states)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Compute BC loss (negative log likelihood)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        bc_loss = -log_prob.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()
        
        return {
            'bc_loss': bc_loss.item(),
            'log_prob': log_prob.mean().item()
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using trained policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(state_tensor, deterministic=deterministic)
            return action.cpu().numpy().flatten()
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ConservativeQLearning:
    """
    Conservative Q-Learning (CQL) baseline implementation
    Adds conservative regularization to prevent overestimation
    """
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create networks
        self.actor = Actor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation,
                'max_action': config.max_action
            }
        ).to(self.device)
        
        self.critic1 = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation
            }
        ).to(self.device)
        
        self.critic2 = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation
            }
        ).to(self.device)
        
        # Target networks
        self.target_critic1 = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation
            }
        ).to(self.device)
        
        self.target_critic2 = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation
            }
        ).to(self.device)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.learning_rate
        )
        
        # CQL hyperparameters
        self.cql_alpha = 1.0
        self.cql_temp = 1.0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for CQL"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update critics with CQL regularization
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update target networks
        self._update_targets()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        }
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks with CQL regularization"""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states, with_log_prob=True)
            
            # Compute target Q-values
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.cql_temp * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # TD loss
        td_loss1 = F.mse_loss(current_q1, target_q)
        td_loss2 = F.mse_loss(current_q2, target_q)
        
        # CQL regularization
        cql_loss1 = self._compute_cql_loss(self.critic1, states, actions)
        cql_loss2 = self._compute_cql_loss(self.critic2, states, actions)
        
        # Total critic loss
        critic_loss = td_loss1 + td_loss2 + self.cql_alpha * (cql_loss1 + cql_loss2)
        
        # Backward pass
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _compute_cql_loss(self, critic, states, actions):
        """Compute CQL regularization loss"""
        batch_size = states.shape[0]
        
        # Sample random actions
        random_actions = torch.FloatTensor(batch_size, self.config.action_dim).uniform_(
            -self.config.max_action, self.config.max_action
        ).to(self.device)
        
        # Sample actions from current policy
        policy_actions, _ = self.actor.sample(states)
        
        # Q-values for different action sources
        q_random = critic(states, random_actions)
        q_policy = critic(states, policy_actions)
        q_data = critic(states, actions)
        
        # CQL loss: log-sum-exp of Q-values minus data Q-values
        cat_q = torch.cat([q_random, q_policy], dim=0)
        cql_loss = torch.logsumexp(cat_q / self.cql_temp, dim=0).mean() - q_data.mean()
        
        return cql_loss
    
    def _update_actor(self, states):
        """Update actor network"""
        actions, log_probs = self.actor.sample(states, with_log_prob=True)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        # Actor loss: maximize Q-values
        actor_loss = -q.mean()
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_targets(self):
        """Soft update of target networks"""
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using trained policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)
            return action.cpu().numpy().flatten()


class ImplicitQLearning:
    """
    Implicit Q-Learning (IQL) baseline implementation
    Uses expectile regression for value learning
    """
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create networks
        self.actor = Actor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation,
                'max_action': config.max_action
            }
        ).to(self.device)
        
        self.critic = Critic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config={
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'activation': config.activation
            }
        ).to(self.device)
        
        # Value network for IQL
        self.value_net = MLP(
            input_dim=config.state_dim,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.learning_rate)
        
        # IQL hyperparameters
        self.expectile = 0.8
        self.temperature = 3.0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for IQL"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update value network
        value_loss = self._update_value(states, actions)
        
        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states, actions)
        
        return {
            'value_loss': value_loss,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss
        }
    
    def _update_value(self, states, actions):
        """Update value network using expectile regression"""
        q_values = self.critic(states, actions)
        v_values = self.value_net(states)
        
        # Expectile loss
        diff = q_values - v_values
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        value_loss = (weight * diff.pow(2)).mean()
        
        # Backward pass
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """Update critic network"""
        with torch.no_grad():
            target_v = self.value_net(next_states)
            target_q = rewards + (1 - dones) * self.config.gamma * target_v
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Backward pass
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states, actions):
        """Update actor network using advantage weighting"""
        with torch.no_grad():
            v_values = self.value_net(states)
            q_values = self.critic(states, actions)
            advantages = q_values - v_values
            weights = torch.exp(advantages / self.temperature)
            weights = torch.clamp(weights, max=100.0)  # Clip for stability
        
        # Actor loss: weighted behavior cloning
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        
        actor_loss = -(weights * log_prob).mean()
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using trained policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)
            return action.cpu().numpy().flatten()


class BaselineTrainer:
    """
    Trainer for baseline algorithms
    """
    
    def __init__(self, algorithm, replay_buffer: ReplayBuffer, config: BaselineConfig):
        self.algorithm = algorithm
        self.replay_buffer = replay_buffer
        self.config = config
        
        # Metrics tracking
        self.metrics_tracker = create_metrics_tracker({
            'track_training': True,
            'track_evaluation': True,
            'save_plots': True
        })
        
        # Training state
        self.epoch = 0
        self.best_score = -np.inf
        
    def train(self, num_epochs: int, eval_env=None) -> Dict[str, List[float]]:
        """Train the baseline algorithm"""
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training step
            batch = self.replay_buffer.sample(self.config.batch_size)
            metrics = self.algorithm.train_step(batch)
            
            # Log training metrics
            for key, value in metrics.items():
                self.metrics_tracker.log_metric(f'train/{key}', value, epoch)
            
            # Evaluation
            if epoch % self.config.eval_freq == 0 and eval_env is not None:
                eval_metrics = self.evaluate(eval_env)
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    self.metrics_tracker.log_metric(f'eval/{key}', value, epoch)
                
                # Save best model
                if eval_metrics['return'] > self.best_score:
                    self.best_score = eval_metrics['return']
                    self.save_checkpoint(f'best_model_epoch_{epoch}.pt')
                
                logging.info(f"Epoch {epoch}: Return = {eval_metrics['return']:.2f}")
        
        return self.metrics_tracker.get_all_metrics()
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the algorithm"""
        returns = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.algorithm.select_action(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                
                episode_return += reward
                episode_length += 1
                state = next_state
            
            returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return {
            'return': np.mean(returns),
            'return_std': np.std(returns),
            'episode_length': np.mean(episode_lengths)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        self.algorithm.save_checkpoint(filepath)


class BaselineComparison:
    """
    Framework for comparing baseline methods with IACT
    """
    
    def __init__(self, env_name: str, dataset_type: str = 'medium'):
        self.env_name = env_name
        self.dataset_type = dataset_type
        
        # Load dataset
        self.dataset, self.replay_buffer = load_d4rl_dataset(
            env_name=env_name,
            normalize_states=True,
            normalize_rewards=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Get environment configuration
        self.env_config = get_env_config(env_name)
        self.state_dim = self.env_config['state_dim']
        self.action_dim = self.env_config['action_dim']
        
        # Results storage
        self.results = {}
        
    def run_baseline(self, algorithm_name: str, config: Optional[BaselineConfig] = None) -> Dict[str, Any]:
        """Run a specific baseline algorithm"""
        if config is None:
            config = BaselineConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.env_config.get('max_action', 1.0)
            )
        
        # Create algorithm
        if algorithm_name == 'bc':
            algorithm = BehaviorCloning(config)
        elif algorithm_name == 'cql':
            algorithm = ConservativeQLearning(config)
        elif algorithm_name == 'iql':
            algorithm = ImplicitQLearning(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Create trainer
        trainer = BaselineTrainer(algorithm, self.replay_buffer, config)
        
        # Train
        start_time = time.time()
        metrics = trainer.train(config.max_epochs)
        training_time = time.time() - start_time
        
        # Store results
        result = {
            'algorithm': algorithm_name,
            'env_name': self.env_name,
            'dataset_type': self.dataset_type,
            'metrics': metrics,
            'training_time': training_time,
            'config': asdict(config)
        }
        
        self.results[algorithm_name] = result
        return result
    
    def run_all_baselines(self) -> Dict[str, Any]:
        """Run all baseline algorithms"""
        baselines = ['bc', 'cql', 'iql']
        
        for baseline in baselines:
            logging.info(f"Running {baseline.upper()} baseline...")
            try:
                self.run_baseline(baseline)
                logging.info(f"Completed {baseline.upper()} baseline")
            except Exception as e:
                logging.error(f"Failed to run {baseline.upper()}: {e}")
        
        return self.results
    
    def save_results(self, filepath: str):
        """Save comparison results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def generate_comparison_report(self) -> str:
        """Generate a comparison report"""
        report = f"Baseline Comparison Report\n"
        report += f"Environment: {self.env_name}\n"
        report += f"Dataset: {self.dataset_type}\n"
        report += "=" * 50 + "\n\n"
        
        for algorithm, result in self.results.items():
            report += f"{algorithm.upper()} Results:\n"
            
            # Get final evaluation metrics
            if 'eval/return' in result['metrics']:
                final_return = result['metrics']['eval/return'][-1]
                report += f"  Final Return: {final_return:.2f}\n"
            
            report += f"  Training Time: {result['training_time']:.2f}s\n"
            report += "\n"
        
        return report


def create_baseline_config(env_name: str, algorithm: str, **overrides) -> BaselineConfig:
    """Create baseline configuration for specific environment and algorithm"""
    env_config = get_env_config(env_name)
    
    config = BaselineConfig(
        state_dim=env_config['state_dim'],
        action_dim=env_config['action_dim'],
        max_action=env_config.get('max_action', 1.0)
    )
    
    # Algorithm-specific adjustments
    if algorithm == 'cql':
        config.learning_rate = 1e-4  # CQL typically uses lower learning rate
        config.max_epochs = 2000     # CQL needs more training
    elif algorithm == 'iql':
        config.learning_rate = 3e-4
        config.max_epochs = 1500
    elif algorithm == 'bc':
        config.learning_rate = 1e-3  # BC can use higher learning rate
        config.max_epochs = 500      # BC converges faster
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def main():
    """Main function for running baseline comparisons"""
    parser = argparse.ArgumentParser(description='Run baseline comparisons for IACT')
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2',
                       help='D4RL environment name')
    parser.add_argument('--dataset', type=str, default='medium',
                       help='Dataset type (medium, expert, random)')
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['bc', 'cql', 'iql', 'all'],
                       help='Baseline algorithm to run')
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create baseline comparison
    comparison = BaselineComparison(args.env, args.dataset)
    
    # Run baselines
    if args.algorithm == 'all':
        results = comparison.run_all_baselines()
    else:
        results = {args.algorithm: comparison.run_baseline(args.algorithm)}
    
    # Save results
    results_file = output_dir / f'{args.env}_{args.dataset}_baseline_results.json'
    comparison.save_results(str(results_file))
    
    # Generate and save report
    report = comparison.generate_comparison_report()
    report_file = output_dir / f'{args.env}_{args.dataset}_baseline_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Results saved to {results_file}")
    print(f"Report saved to {report_file}")
    print("\nComparison Report:")
    print(report)


if __name__ == '__main__':
    main()