"""
Enhanced Replay Buffer for IACT Offline Reinforcement Learning

This module implements a replay buffer with importance weighting support for the IACT algorithm.
The buffer stores offline RL datasets and provides efficient sampling with importance weights.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Enhanced replay buffer for offline RL with importance weighting support.
    
    Features:
    - Efficient storage of state-action-reward-next_state-done tuples
    - Importance weight storage and sampling
    - Batch sampling with optional importance weighting
    - Memory-efficient tensor operations
    - Support for continuous and discrete action spaces
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        state_dim: int = None,
        action_dim: int = None,
        device: str = 'cpu',
        store_importance_weights: bool = True
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to store tensors on
            store_importance_weights: Whether to store importance weights
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.store_importance_weights = store_importance_weights
        
        # Storage tensors
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.importance_weights = None
        
        # Buffer management
        self.size = 0
        self.ptr = 0
        self.is_full = False
        
        # Statistics
        self.stats = defaultdict(float)
        
        logger.info(f"Initialized ReplayBuffer with capacity {capacity}")
    
    def _initialize_tensors(self, state_dim: int, action_dim: int):
        """Initialize storage tensors based on first sample dimensions."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.bool, device=self.device)
        
        if self.store_importance_weights:
            self.importance_weights = torch.ones((self.capacity, 1), dtype=torch.float32, device=self.device)
        
        logger.info(f"Initialized tensors: state_dim={state_dim}, action_dim={action_dim}")
    
    def add(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: Union[float, np.ndarray, torch.Tensor],
        next_state: Union[np.ndarray, torch.Tensor],
        done: Union[bool, np.ndarray, torch.Tensor],
        importance_weight: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    ):
        """
        Add a single transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            importance_weight: Importance weight for this transition
        """
        # Convert to tensors
        state = self._to_tensor(state)
        action = self._to_tensor(action)
        reward = self._to_tensor(reward).reshape(-1, 1)
        next_state = self._to_tensor(next_state)
        done = self._to_tensor(done, dtype=torch.bool).reshape(-1, 1)
        
        # Initialize tensors if needed
        if self.states is None:
            self._initialize_tensors(state.shape[-1], action.shape[-1])
        
        # Store transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        if self.store_importance_weights and importance_weight is not None:
            importance_weight = self._to_tensor(importance_weight).reshape(-1, 1)
            self.importance_weights[self.ptr] = importance_weight
        
        # Update buffer state
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.is_full = True
        
        # Update statistics
        self.stats['total_added'] += 1
        self.stats['avg_reward'] = (self.stats['avg_reward'] * (self.stats['total_added'] - 1) + 
                                   reward.item()) / self.stats['total_added']
    
    def add_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_states: Union[np.ndarray, torch.Tensor],
        dones: Union[np.ndarray, torch.Tensor],
        importance_weights: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        """
        Add a batch of transitions to the buffer.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            rewards: Batch of rewards [batch_size, 1] or [batch_size]
            next_states: Batch of next states [batch_size, state_dim]
            dones: Batch of done flags [batch_size, 1] or [batch_size]
            importance_weights: Batch of importance weights [batch_size, 1] or [batch_size]
        """
        # Convert to tensors
        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = self._to_tensor(rewards)
        next_states = self._to_tensor(next_states)
        dones = self._to_tensor(dones, dtype=torch.bool)
        
        # Ensure proper shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        batch_size = states.shape[0]
        
        # Initialize tensors if needed
        if self.states is None:
            self._initialize_tensors(states.shape[-1], actions.shape[-1])
        
        # Handle buffer overflow
        if self.ptr + batch_size <= self.capacity:
            # Simple case: batch fits without wrapping
            end_idx = self.ptr + batch_size
            self.states[self.ptr:end_idx] = states
            self.actions[self.ptr:end_idx] = actions
            self.rewards[self.ptr:end_idx] = rewards
            self.next_states[self.ptr:end_idx] = next_states
            self.dones[self.ptr:end_idx] = dones
            
            if self.store_importance_weights and importance_weights is not None:
                importance_weights = self._to_tensor(importance_weights)
                if importance_weights.dim() == 1:
                    importance_weights = importance_weights.unsqueeze(1)
                self.importance_weights[self.ptr:end_idx] = importance_weights
            
            self.ptr = end_idx % self.capacity
        else:
            # Wrap around case
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            # First part
            self.states[self.ptr:] = states[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            self.rewards[self.ptr:] = rewards[:first_part]
            self.next_states[self.ptr:] = next_states[:first_part]
            self.dones[self.ptr:] = dones[:first_part]
            
            # Second part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]
            
            if self.store_importance_weights and importance_weights is not None:
                importance_weights = self._to_tensor(importance_weights)
                if importance_weights.dim() == 1:
                    importance_weights = importance_weights.unsqueeze(1)
                self.importance_weights[self.ptr:] = importance_weights[:first_part]
                self.importance_weights[:second_part] = importance_weights[first_part:]
            
            self.ptr = second_part
            self.is_full = True
        
        # Update size
        self.size = min(self.size + batch_size, self.capacity)
        
        # Update statistics
        self.stats['total_added'] += batch_size
        self.stats['avg_reward'] = (self.stats['avg_reward'] * (self.stats['total_added'] - batch_size) + 
                                   rewards.sum().item()) / self.stats['total_added']
        
        logger.info(f"Added batch of {batch_size} transitions. Buffer size: {self.size}")
    
    def sample(
        self,
        batch_size: int,
        with_importance_weights: bool = True,
        indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            with_importance_weights: Whether to include importance weights
            indices: Specific indices to sample (if None, sample randomly)
        
        Returns:
            Dictionary containing batch of transitions
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if indices is None:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        else:
            indices = indices.to(self.device)
            batch_size = len(indices)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
        if with_importance_weights and self.store_importance_weights:
            batch['importance_weights'] = self.importance_weights[indices]
        
        return batch
    
    def sample_with_importance_sampling(
        self,
        batch_size: int,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Sample transitions using importance weights as probabilities.
        
        Args:
            batch_size: Number of transitions to sample
            temperature: Temperature for softmax sampling (lower = more focused)
        
        Returns:
            Dictionary containing batch of transitions
        """
        if not self.store_importance_weights:
            return self.sample(batch_size)
        
        # Get importance weights and convert to probabilities
        weights = self.importance_weights[:self.size].squeeze()
        probs = torch.softmax(weights / temperature, dim=0)
        
        # Sample indices according to probabilities
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        return self.sample(batch_size, indices=indices)
    
    def update_importance_weights(
        self,
        indices: torch.Tensor,
        new_weights: torch.Tensor
    ):
        """
        Update importance weights for specific transitions.
        
        Args:
            indices: Indices of transitions to update
            new_weights: New importance weights
        """
        if not self.store_importance_weights:
            logger.warning("Buffer not configured to store importance weights")
            return
        
        indices = indices.to(self.device)
        new_weights = self._to_tensor(new_weights)
        
        if new_weights.dim() == 1:
            new_weights = new_weights.unsqueeze(1)
        
        self.importance_weights[indices] = new_weights
        
        logger.debug(f"Updated importance weights for {len(indices)} transitions")
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data in the buffer.
        
        Returns:
            Dictionary containing all transitions
        """
        if self.size == 0:
            return {}
        
        data = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size]
        }
        
        if self.store_importance_weights:
            data['importance_weights'] = self.importance_weights[:self.size]
        
        return data
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        stats = dict(self.stats)
        stats.update({
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'is_full': self.is_full
        })
        
        if self.size > 0:
            stats['reward_std'] = self.rewards[:self.size].std().item()
            stats['reward_min'] = self.rewards[:self.size].min().item()
            stats['reward_max'] = self.rewards[:self.size].max().item()
            
            if self.store_importance_weights:
                weights = self.importance_weights[:self.size]
                stats['importance_weight_mean'] = weights.mean().item()
                stats['importance_weight_std'] = weights.std().item()
                stats['importance_weight_min'] = weights.min().item()
                stats['importance_weight_max'] = weights.max().item()
        
        return stats
    
    def clear(self):
        """Clear the buffer."""
        self.size = 0
        self.ptr = 0
        self.is_full = False
        self.stats.clear()
        
        if self.states is not None:
            self.states.zero_()
            self.actions.zero_()
            self.rewards.zero_()
            self.next_states.zero_()
            self.dones.zero_()
            
            if self.store_importance_weights:
                self.importance_weights.fill_(1.0)
        
        logger.info("Buffer cleared")
    
    def _to_tensor(
        self,
        data: Union[np.ndarray, torch.Tensor, float, int, bool],
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Convert data to tensor on the correct device."""
        if isinstance(data, torch.Tensor):
            tensor = data.to(self.device)
        else:
            tensor = torch.tensor(data, device=self.device)
        
        if dtype is not None:
            tensor = tensor.to(dtype)
        elif tensor.dtype == torch.float64:
            tensor = tensor.float()
        
        return tensor
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
    
    def __repr__(self) -> str:
        return (f"ReplayBuffer(capacity={self.capacity}, size={self.size}, "
                f"state_dim={self.state_dim}, action_dim={self.action_dim})")


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer with importance sampling.
    
    Extends the basic replay buffer with priority-based sampling using
    importance weights as priorities.
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        state_dim: int = None,
        action_dim: int = None,
        device: str = 'cpu',
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to store tensors on
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling step
            epsilon: Small constant to prevent zero priorities
        """
        super().__init__(capacity, state_dim, action_dim, device, store_importance_weights=True)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Priority storage
        self.priorities = torch.ones(capacity, device=device) * epsilon
        self.max_priority = epsilon
        
        logger.info(f"Initialized PrioritizedReplayBuffer with alpha={alpha}, beta={beta}")
    
    def add(self, *args, **kwargs):
        """Add transition with maximum priority."""
        old_ptr = self.ptr
        super().add(*args, **kwargs)
        
        # Set maximum priority for new transition
        self.priorities[old_ptr] = self.max_priority
    
    def add_batch(self, *args, **kwargs):
        """Add batch of transitions with maximum priority."""
        old_ptr = self.ptr
        old_size = self.size
        super().add_batch(*args, **kwargs)
        
        # Set maximum priority for new transitions
        if old_ptr + (self.size - old_size) <= self.capacity:
            self.priorities[old_ptr:old_ptr + (self.size - old_size)] = self.max_priority
        else:
            # Handle wrap-around
            first_part = self.capacity - old_ptr
            self.priorities[old_ptr:] = self.max_priority
            self.priorities[:self.size - old_size - first_part] = self.max_priority
    
    def sample(
        self,
        batch_size: int,
        with_importance_weights: bool = True,
        indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample batch using priority-based sampling."""
        if indices is None:
            # Sample according to priorities
            priorities = self.priorities[:self.size] ** self.alpha
            probs = priorities / priorities.sum()
            
            indices = torch.multinomial(probs, batch_size, replacement=True)
            
            # Compute importance sampling weights
            if with_importance_weights:
                weights = (self.size * probs[indices]) ** (-self.beta)
                weights = weights / weights.max()  # Normalize by max weight
        else:
            indices = indices.to(self.device)
            batch_size = len(indices)
            
            if with_importance_weights:
                priorities = self.priorities[indices] ** self.alpha
                probs = priorities / self.priorities[:self.size].sum()
                weights = (self.size * probs) ** (-self.beta)
                weights = weights / weights.max()
        
        # Get batch
        batch = super().sample(batch_size, with_importance_weights=False, indices=indices)
        
        if with_importance_weights:
            batch['importance_weights'] = weights.unsqueeze(1)
            batch['indices'] = indices
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch
    
    def update_priorities(
        self,
        indices: torch.Tensor,
        priorities: torch.Tensor
    ):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priorities (typically TD errors)
        """
        indices = indices.to(self.device)
        priorities = self._to_tensor(priorities) + self.epsilon
        
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
        
        logger.debug(f"Updated priorities for {len(indices)} transitions")


def create_replay_buffer(
    buffer_type: str = 'standard',
    capacity: int = 1000000,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    device: str = 'cpu',
    **kwargs
) -> ReplayBuffer:
    """
    Factory function to create replay buffer.
    
    Args:
        buffer_type: Type of buffer ('standard' or 'prioritized')
        capacity: Buffer capacity
        state_dim: State dimension
        action_dim: Action dimension
        device: Device to store tensors on
        **kwargs: Additional arguments for specific buffer types
    
    Returns:
        Configured replay buffer
    """
    if buffer_type == 'standard':
        return ReplayBuffer(
            capacity=capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(
            capacity=capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


def load_dataset_to_buffer(
    dataset: Dict[str, np.ndarray],
    buffer: ReplayBuffer,
    normalize_rewards: bool = False,
    importance_weights: Optional[np.ndarray] = None
) -> ReplayBuffer:
    """
    Load a dataset into a replay buffer.
    
    Args:
        dataset: Dictionary with 'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        buffer: Replay buffer to load data into
        normalize_rewards: Whether to normalize rewards
        importance_weights: Optional importance weights for transitions
    
    Returns:
        Loaded replay buffer
    """
    states = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_observations']
    dones = dataset['terminals']
    
    # Normalize rewards if requested
    if normalize_rewards:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        logger.info(f"Normalized rewards: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
    
    # Load data into buffer
    buffer.add_batch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        importance_weights=importance_weights
    )
    
    logger.info(f"Loaded {len(states)} transitions into buffer")
    return buffer