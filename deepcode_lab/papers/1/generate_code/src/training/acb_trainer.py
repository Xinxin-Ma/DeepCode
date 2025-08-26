"""
ACB-Agent Training Module

This module implements the training algorithm for ACB-Agent as described in Algorithm 1
from Section 3.2 of the paper. It includes the complete training loop with multi-objective
loss computation combining policy gradient, explanation, and diversity losses.

Key Components:
- ACBTrainer: Main training class implementing Algorithm 1
- Multi-objective loss computation (L_policy + λ_explain*L_explain + λ_diverse*L_diverse)
- Experience collection and trajectory processing
- Advantage estimation using GAE (Generalized Advantage Estimation)
- Gradient-based parameter updates

Mathematical Formulation:
- Policy Loss: L_policy = -∑log π_θ(a_t|c_t)A_t
- Explanation Loss: L_explain = BCE(c_t, ĉ_t) 
- Diversity Loss: L_diverse = -H(c_t) (negative entropy)
- Total Loss: L = L_policy + λ_explain*L_explain + λ_diverse*L_diverse
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
import time

# Import our ACB components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.acb_agent import ACBAgent, MultiAgentACBSystem
from models.concept_bottleneck import ConceptBottleneckLayer


class ExperienceBuffer:
    """
    Experience buffer for storing and processing trajectories.
    Supports both single-agent and multi-agent scenarios.
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.clear()
        
    def clear(self):
        """Clear all stored experiences."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.concept_activations = []
        self.dones = []
        self.next_states = []
        
    def add_experience(self, state: torch.Tensor, action: torch.Tensor, 
                      reward: float, value: torch.Tensor, log_prob: torch.Tensor,
                      concept_activation: torch.Tensor, done: bool, 
                      next_state: Optional[torch.Tensor] = None):
        """Add a single experience to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.concept_activations.append(concept_activation)
        self.dones.append(done)
        self.next_states.append(next_state)
        
        # Maintain buffer size
        if len(self.states) > self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.concept_activations.pop(0)
            self.dones.pop(0)
            self.next_states.pop(0)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all experiences as batched tensors."""
        if len(self.states) == 0:
            return {}
            
        return {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'values': torch.stack(self.values),
            'log_probs': torch.stack(self.log_probs),
            'concept_activations': torch.stack(self.concept_activations),
            'dones': torch.tensor(self.dones, dtype=torch.bool),
            'next_states': torch.stack([s for s in self.next_states if s is not None]) if any(s is not None for s in self.next_states) else None
        }
    
    def __len__(self):
        return len(self.states)


class ACBTrainer:
    """
    Main ACB-Agent trainer implementing Algorithm 1 from Section 3.2.
    
    This trainer handles:
    1. Experience collection from environment interactions
    2. Multi-objective loss computation
    3. Gradient-based parameter updates
    4. Training metrics and logging
    """
    
    def __init__(self, 
                 agent: ACBAgent,
                 learning_rate: float = 3e-4,
                 lambda_explain: float = 0.1,
                 lambda_diverse: float = 0.05,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize ACB trainer.
        
        Args:
            agent: ACBAgent instance to train
            learning_rate: Learning rate for optimizer
            lambda_explain: Weight for explanation loss (λ_explain)
            lambda_diverse: Weight for diversity loss (λ_diverse)
            gamma: Discount factor for rewards
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computation
        """
        self.agent = agent
        self.device = device
        
        # Hyperparameters from paper
        self.lambda_explain = lambda_explain
        self.lambda_diverse = lambda_diverse
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Training metrics
        self.training_metrics = {
            'total_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'explanation_loss': deque(maxlen=100),
            'diversity_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'episode_rewards': deque(maxlen=100),
            'concept_alignment': deque(maxlen=100)
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def collect_trajectory(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Collect a complete trajectory from environment interaction.
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary containing trajectory information
        """
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'concept_activations': [],
            'dones': [],
            'explanations': []
        }
        
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle gym environments that return (obs, info)
            
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from agent with explanation
            with torch.no_grad():
                action_info = self.agent.act(state_tensor, 
                                           deterministic=False, 
                                           return_explanation=True)
            
            # Extract action for environment
            action = action_info['actions'].cpu().numpy().squeeze()
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            # Handle both done and truncated (for newer gym versions)
            episode_done = done or truncated
            
            # Store experience
            trajectory['states'].append(state_tensor)
            trajectory['actions'].append(action_info['actions'])
            trajectory['rewards'].append(reward)
            trajectory['values'].append(action_info['values'])
            trajectory['log_probs'].append(action_info['log_probs'])
            trajectory['concept_activations'].append(action_info['concept_activations'])
            trajectory['dones'].append(episode_done)
            trajectory['explanations'].append(action_info.get('explanation', ''))
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if episode_done:
                break
        
        trajectory['total_reward'] = total_reward
        trajectory['steps'] = steps
        
        return trajectory
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                          dones: torch.Tensor, next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            dones: Tensor of done flags
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = torch.tensor(next_value, dtype=torch.float32)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                           advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute PPO policy loss with clipping.
        
        Args:
            log_probs: Current log probabilities
            old_log_probs: Old log probabilities
            advantages: Computed advantages
            
        Returns:
            Policy loss tensor
        """
        # Compute probability ratios
        ratios = torch.exp(log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        # Take minimum for conservative policy update
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def compute_explanation_loss(self, concept_activations: torch.Tensor,
                                target_concepts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute explanation loss L_explain = BCE(c_t, ĉ_t).
        
        Args:
            concept_activations: Current concept activations
            target_concepts: Target concept activations (if available)
            
        Returns:
            Explanation loss tensor
        """
        if target_concepts is None:
            # If no target concepts provided, use self-consistency loss
            # Encourage consistent concept activations for similar states
            batch_size = concept_activations.size(0)
            if batch_size > 1:
                # Compute pairwise consistency loss
                expanded_concepts = concept_activations.unsqueeze(1).expand(-1, batch_size, -1)
                pairwise_diff = torch.abs(expanded_concepts - concept_activations.unsqueeze(0))
                consistency_loss = pairwise_diff.mean()
                return consistency_loss
            else:
                return torch.tensor(0.0, device=concept_activations.device)
        else:
            # Use BCE loss with target concepts
            bce_loss = nn.BCELoss()
            return bce_loss(concept_activations, target_concepts)
    
    def compute_diversity_loss(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss L_diverse = -H(c_t) (negative entropy).
        
        Args:
            concept_activations: Concept activations tensor
            
        Returns:
            Diversity loss tensor
        """
        # Compute entropy of concept activations
        # H(c) = -∑ c_i * log(c_i) - (1-c_i) * log(1-c_i)
        eps = 1e-8  # Small epsilon to avoid log(0)
        c = torch.clamp(concept_activations, eps, 1.0 - eps)
        
        entropy = -(c * torch.log(c) + (1 - c) * torch.log(1 - c))
        entropy = entropy.sum(dim=-1).mean()
        
        # Return negative entropy as diversity loss
        diversity_loss = -entropy
        
        return diversity_loss
    
    def train_step(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step using collected trajectory.
        
        Args:
            trajectory: Dictionary containing trajectory data
            
        Returns:
            Dictionary of training metrics
        """
        # Extract trajectory data
        states = torch.cat(trajectory['states'])
        actions = torch.cat(trajectory['actions'])
        rewards = torch.tensor(trajectory['rewards'], dtype=torch.float32, device=self.device)
        old_values = torch.cat(trajectory['values'])
        old_log_probs = torch.cat(trajectory['log_probs'])
        concept_activations = torch.cat(trajectory['concept_activations'])
        dones = torch.tensor(trajectory['dones'], dtype=torch.bool, device=self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Re-evaluate actions with current policy
        action_info = self.agent.evaluate_actions(states, actions)
        new_log_probs = action_info['log_probs']
        new_values = action_info['values']
        entropy = action_info['entropy']
        new_concept_activations = action_info['concept_activations']
        
        # Compute individual loss components
        policy_loss = self.compute_policy_loss(new_log_probs, old_log_probs, advantages)
        
        value_loss = nn.MSELoss()(new_values.squeeze(), returns)
        
        explanation_loss = self.compute_explanation_loss(new_concept_activations)
        
        diversity_loss = self.compute_diversity_loss(new_concept_activations)
        
        # Compute total loss (Algorithm 1 from paper)
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.lambda_explain * explanation_loss + 
                     self.lambda_diverse * diversity_loss - 
                     self.entropy_coef * entropy)
        
        # Perform gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Compute metrics
        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'explanation_loss': explanation_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'entropy': entropy.item(),
            'episode_reward': trajectory['total_reward'],
            'episode_steps': trajectory['steps']
        }
        
        # Update training metrics
        for key, value in metrics.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)
        
        return metrics
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """
        Train for one complete episode.
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            
        Returns:
            Training metrics for the episode
        """
        # Collect trajectory
        trajectory = self.collect_trajectory(env, max_steps)
        
        # Perform training step
        metrics = self.train_step(trajectory)
        
        return metrics
    
    def get_training_stats(self) -> Dict[str, float]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        stats = {}
        for key, values in self.training_metrics.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_recent'] = values[-1] if len(values) > 0 else 0.0
        
        return stats
    
    def save_checkpoint(self, filepath: str, episode: int, additional_info: Dict = None):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
            additional_info: Additional information to save
        """
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': dict(self.training_metrics),
            'hyperparameters': {
                'lambda_explain': self.lambda_explain,
                'lambda_diverse': self.lambda_diverse,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_loss_coef': self.value_loss_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Loaded checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training metrics
        for key, values in checkpoint['training_metrics'].items():
            if key in self.training_metrics:
                self.training_metrics[key] = deque(values, maxlen=100)
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint


class MultiAgentACBTrainer:
    """
    Trainer for multi-agent ACB systems.
    Extends single-agent training to coordinate multiple ACB agents.
    """
    
    def __init__(self, 
                 multi_agent_system: MultiAgentACBSystem,
                 learning_rate: float = 3e-4,
                 lambda_explain: float = 0.1,
                 lambda_diverse: float = 0.05,
                 coordination_weight: float = 0.1,
                 **kwargs):
        """
        Initialize multi-agent ACB trainer.
        
        Args:
            multi_agent_system: MultiAgentACBSystem instance
            learning_rate: Learning rate for optimizer
            lambda_explain: Weight for explanation loss
            lambda_diverse: Weight for diversity loss
            coordination_weight: Weight for coordination loss
            **kwargs: Additional arguments for ACBTrainer
        """
        self.multi_agent_system = multi_agent_system
        self.coordination_weight = coordination_weight
        
        # Create individual trainers for each agent
        self.agent_trainers = []
        for agent in multi_agent_system.agents:
            trainer = ACBTrainer(agent, learning_rate, lambda_explain, lambda_diverse, **kwargs)
            self.agent_trainers.append(trainer)
        
        # Shared experience buffer for coordination
        self.shared_buffer = ExperienceBuffer()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def compute_coordination_loss(self, agent_concepts: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute coordination loss to encourage agent cooperation.
        
        Args:
            agent_concepts: List of concept activations for each agent
            
        Returns:
            Coordination loss tensor
        """
        if len(agent_concepts) < 2:
            return torch.tensor(0.0)
        
        # Compute pairwise concept similarity
        coordination_loss = 0.0
        num_pairs = 0
        
        for i in range(len(agent_concepts)):
            for j in range(i + 1, len(agent_concepts)):
                # Cosine similarity between concept activations
                cos_sim = nn.functional.cosine_similarity(
                    agent_concepts[i], agent_concepts[j], dim=-1
                ).mean()
                coordination_loss += (1.0 - cos_sim)  # Encourage similarity
                num_pairs += 1
        
        if num_pairs > 0:
            coordination_loss /= num_pairs
        
        return coordination_loss
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """
        Train multi-agent system for one episode.
        
        Args:
            env: Multi-agent environment
            max_steps: Maximum steps per episode
            
        Returns:
            Training metrics for all agents
        """
        # Collect multi-agent trajectory
        states = env.reset()
        if isinstance(states, tuple):
            states = states[0]
        
        episode_metrics = {f'agent_{i}': [] for i in range(len(self.agent_trainers))}
        total_rewards = [0.0] * len(self.agent_trainers)
        
        for step in range(max_steps):
            # Get actions from all agents
            state_tensors = [torch.FloatTensor(state).unsqueeze(0) for state in states]
            
            with torch.no_grad():
                actions_info = self.multi_agent_system.act(state_tensors, 
                                                         deterministic=False, 
                                                         return_explanations=True)
            
            # Extract actions for environment
            actions = [info['actions'].cpu().numpy().squeeze() for info in actions_info]
            
            # Take step in environment
            next_states, rewards, dones, truncated, infos = env.step(actions)
            if isinstance(next_states, tuple):
                next_states = next_states[0]
            
            # Handle episode termination
            episode_done = any(dones) or any(truncated)
            
            # Store experiences for each agent
            for i, (trainer, action_info) in enumerate(zip(self.agent_trainers, actions_info)):
                trainer.experience_buffer.add_experience(
                    state_tensors[i],
                    action_info['actions'],
                    rewards[i],
                    action_info['values'],
                    action_info['log_probs'],
                    action_info['concept_activations'],
                    episode_done,
                    torch.FloatTensor(next_states[i]).unsqueeze(0) if not episode_done else None
                )
                total_rewards[i] += rewards[i]
            
            states = next_states
            
            if episode_done:
                break
        
        # Train each agent
        all_metrics = {}
        agent_concepts = []
        
        for i, trainer in enumerate(self.agent_trainers):
            if len(trainer.experience_buffer) > 0:
                # Get trajectory data
                batch = trainer.experience_buffer.get_batch()
                trajectory = {
                    'states': [batch['states'][j] for j in range(len(batch['states']))],
                    'actions': [batch['actions'][j] for j in range(len(batch['actions']))],
                    'rewards': batch['rewards'].tolist(),
                    'values': [batch['values'][j] for j in range(len(batch['values']))],
                    'log_probs': [batch['log_probs'][j] for j in range(len(batch['log_probs']))],
                    'concept_activations': [batch['concept_activations'][j] for j in range(len(batch['concept_activations']))],
                    'dones': batch['dones'].tolist(),
                    'total_reward': total_rewards[i],
                    'steps': len(batch['states'])
                }
                
                # Train individual agent
                metrics = trainer.train_step(trajectory)
                all_metrics[f'agent_{i}'] = metrics
                
                # Collect concept activations for coordination
                agent_concepts.append(batch['concept_activations'])
                
                # Clear buffer
                trainer.experience_buffer.clear()
        
        # Compute coordination loss and update
        if len(agent_concepts) > 1:
            coord_loss = self.compute_coordination_loss(agent_concepts)
            all_metrics['coordination_loss'] = coord_loss.item()
            
            # Apply coordination loss to all agents
            for trainer in self.agent_trainers:
                trainer.optimizer.zero_grad()
                (self.coordination_weight * coord_loss).backward(retain_graph=True)
                trainer.optimizer.step()
        
        return all_metrics
    
    def save_checkpoint(self, filepath: str, episode: int):
        """Save multi-agent training checkpoint."""
        checkpoint = {
            'episode': episode,
            'num_agents': len(self.agent_trainers),
            'agent_checkpoints': []
        }
        
        for i, trainer in enumerate(self.agent_trainers):
            agent_checkpoint = {
                'agent_state_dict': trainer.agent.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'training_metrics': dict(trainer.training_metrics)
            }
            checkpoint['agent_checkpoints'].append(agent_checkpoint)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Multi-agent checkpoint saved to {filepath}")


# Utility functions for training setup and configuration

def create_default_trainer(state_dim: int, action_dim: int, concept_dim: int = 32,
                          domain: str = 'general', device: str = 'cpu') -> ACBTrainer:
    """
    Create a default ACB trainer with standard configuration.
    
    Args:
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        concept_dim: Concept space dimensionality
        domain: Domain for concept names
        device: Device for computation
        
    Returns:
        Configured ACBTrainer instance
    """
    # Create ACB agent
    agent = ACBAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        concept_dim=concept_dim,
        domain=domain,
        device=device
    )
    
    # Create trainer with paper hyperparameters
    trainer = ACBTrainer(
        agent=agent,
        learning_rate=3e-4,
        lambda_explain=0.1,
        lambda_diverse=0.05,
        device=device
    )
    
    return trainer


def setup_training_environment(trainer: ACBTrainer, log_level: str = 'INFO'):
    """
    Setup training environment with logging and monitoring.
    
    Args:
        trainer: ACBTrainer instance
        log_level: Logging level
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log training configuration
    trainer.logger.info("ACB Training Configuration:")
    trainer.logger.info(f"  λ_explain: {trainer.lambda_explain}")
    trainer.logger.info(f"  λ_diverse: {trainer.lambda_diverse}")
    trainer.logger.info(f"  γ (gamma): {trainer.gamma}")
    trainer.logger.info(f"  GAE λ: {trainer.gae_lambda}")
    trainer.logger.info(f"  Clip ε: {trainer.clip_epsilon}")
    trainer.logger.info(f"  Device: {trainer.device}")


if __name__ == "__main__":
    # Example usage and testing
    print("ACB Trainer Module - Testing Basic Functionality")
    
    # Test trainer creation
    try:
        trainer = create_default_trainer(
            state_dim=10,
            action_dim=4,
            concept_dim=16,
            domain='navigation'
        )
        print("✓ ACB Trainer created successfully")
        
        # Test experience buffer
        buffer = ExperienceBuffer()
        print("✓ Experience buffer created successfully")
        
        # Test loss computation functions
        dummy_concepts = torch.rand(5, 16)  # Batch of 5, 16 concepts
        diversity_loss = trainer.compute_diversity_loss(dummy_concepts)
        explanation_loss = trainer.compute_explanation_loss(dummy_concepts)
        
        print(f"✓ Diversity loss computed: {diversity_loss.item():.4f}")
        print(f"✓ Explanation loss computed: {explanation_loss.item():.4f}")
        
        print("\nACB Trainer module is ready for training!")
        
    except Exception as e:
        print(f"✗ Error in ACB Trainer: {e}")
        import traceback
        traceback.print_exc()