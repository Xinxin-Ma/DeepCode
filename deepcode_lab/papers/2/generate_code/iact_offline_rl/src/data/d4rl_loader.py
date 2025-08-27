"""
D4RL Dataset Loader for Offline Reinforcement Learning

This module provides functionality to load and preprocess D4RL datasets for use with
the IACT algorithm. It handles dataset downloading, normalization, and integration
with the replay buffer system.

Key Features:
- Automatic D4RL dataset downloading and caching
- State and reward normalization options
- Integration with ReplayBuffer for efficient storage
- Support for all major D4RL environments (MuJoCo, AntMaze, etc.)
- Dataset statistics computation and validation
"""

import os
import numpy as np
import torch
import logging
from typing import Dict, Tuple, Optional, List, Any
import pickle
from pathlib import Path

try:
    import gym
    import d4rl
except ImportError:
    logging.warning("D4RL not installed. Install with: pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl")
    gym = None
    d4rl = None

from .replay_buffer import ReplayBuffer, load_dataset_to_buffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class D4RLDatasetLoader:
    """
    D4RL Dataset Loader with preprocessing and normalization capabilities.
    
    This class handles loading D4RL datasets, computing statistics, and preparing
    data for offline RL training with importance weighting support.
    """
    
    def __init__(self, 
                 cache_dir: str = "./d4rl_cache",
                 normalize_states: bool = True,
                 normalize_rewards: bool = True,
                 device: str = "cpu"):
        """
        Initialize D4RL dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
            normalize_states: Whether to normalize states to zero mean, unit variance
            normalize_rewards: Whether to normalize rewards
            device: Device for tensor operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalize_states = normalize_states
        self.normalize_rewards = normalize_rewards
        self.device = device
        
        # Dataset statistics for normalization
        self.state_mean = None
        self.state_std = None
        self.reward_mean = None
        self.reward_std = None
        
        # Supported D4RL environments
        self.supported_envs = {
            # MuJoCo environments
            'halfcheetah': ['halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2',
                           'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2'],
            'hopper': ['hopper-random-v2', 'hopper-medium-v2', 'hopper-expert-v2',
                      'hopper-medium-replay-v2', 'hopper-medium-expert-v2'],
            'walker2d': ['walker2d-random-v2', 'walker2d-medium-v2', 'walker2d-expert-v2',
                        'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2'],
            # AntMaze environments
            'antmaze': ['antmaze-umaze-v2', 'antmaze-umaze-diverse-v2', 'antmaze-medium-play-v2',
                       'antmaze-medium-diverse-v2', 'antmaze-large-play-v2', 'antmaze-large-diverse-v2'],
            # Kitchen environments
            'kitchen': ['kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0'],
            # Adroit environments
            'adroit': ['pen-human-v1', 'pen-cloned-v1', 'pen-expert-v1',
                      'hammer-human-v1', 'hammer-cloned-v1', 'hammer-expert-v1',
                      'door-human-v1', 'door-cloned-v1', 'door-expert-v1',
                      'relocate-human-v1', 'relocate-cloned-v1', 'relocate-expert-v1']
        }
        
        logger.info(f"D4RL Loader initialized with cache_dir: {cache_dir}")
    
    def load_dataset(self, 
                    env_name: str,
                    load_to_buffer: bool = True,
                    buffer_capacity: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Optional[ReplayBuffer]]:
        """
        Load D4RL dataset with optional preprocessing and buffer integration.
        
        Args:
            env_name: D4RL environment name (e.g., 'halfcheetah-medium-v2')
            load_to_buffer: Whether to load data into ReplayBuffer
            buffer_capacity: Buffer capacity (defaults to dataset size)
            
        Returns:
            Tuple of (dataset_dict, replay_buffer)
            dataset_dict contains: states, actions, rewards, next_states, terminals
            replay_buffer is None if load_to_buffer=False
        """
        if gym is None or d4rl is None:
            raise ImportError("D4RL not installed. Install with: pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl")
        
        logger.info(f"Loading D4RL dataset: {env_name}")
        
        # Check cache first
        cache_path = self.cache_dir / f"{env_name}.pkl"
        if cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # Load from D4RL
            logger.info(f"Downloading dataset from D4RL...")
            env = gym.make(env_name)
            dataset = env.get_dataset()
            
            # Cache the dataset
            logger.info(f"Caching dataset to {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
        
        # Preprocess dataset
        processed_dataset = self._preprocess_dataset(dataset, env_name)
        
        # Validate dataset
        self._validate_dataset(processed_dataset, env_name)
        
        # Create replay buffer if requested
        replay_buffer = None
        if load_to_buffer:
            replay_buffer = self._create_replay_buffer(processed_dataset, buffer_capacity)
        
        logger.info(f"Successfully loaded {env_name} with {len(processed_dataset['states'])} transitions")
        
        return processed_dataset, replay_buffer
    
    def _preprocess_dataset(self, dataset: Dict[str, np.ndarray], env_name: str) -> Dict[str, np.ndarray]:
        """
        Preprocess D4RL dataset with normalization and formatting.
        
        Args:
            dataset: Raw D4RL dataset
            env_name: Environment name for specific preprocessing
            
        Returns:
            Processed dataset dictionary
        """
        logger.info("Preprocessing dataset...")
        
        # Extract basic components
        states = dataset['observations'].astype(np.float32)
        actions = dataset['actions'].astype(np.float32)
        rewards = dataset['rewards'].astype(np.float32)
        next_states = dataset['next_observations'].astype(np.float32)
        terminals = dataset['terminals'].astype(bool)
        
        # Handle timeouts (some D4RL datasets have this)
        if 'timeouts' in dataset:
            # Don't treat timeouts as true terminals for value learning
            true_terminals = terminals & (~dataset['timeouts'].astype(bool))
        else:
            true_terminals = terminals
        
        # Normalize states if requested
        if self.normalize_states:
            states, next_states = self._normalize_states(states, next_states)
        
        # Normalize rewards if requested
        if self.normalize_rewards:
            rewards = self._normalize_rewards(rewards, env_name)
        
        # Handle environment-specific preprocessing
        if 'antmaze' in env_name.lower():
            # AntMaze environments need special reward processing
            rewards = self._process_antmaze_rewards(rewards, states, next_states)
        elif 'kitchen' in env_name.lower():
            # Kitchen environments have sparse rewards
            rewards = self._process_kitchen_rewards(rewards)
        
        processed_dataset = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'terminals': true_terminals,
            'raw_terminals': terminals,  # Keep original terminals for reference
        }
        
        # Add dataset statistics
        processed_dataset['stats'] = self._compute_dataset_stats(processed_dataset)
        
        return processed_dataset
    
    def _normalize_states(self, states: np.ndarray, next_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize states to zero mean and unit variance."""
        # Compute statistics from current states
        self.state_mean = np.mean(states, axis=0, keepdims=True)
        self.state_std = np.std(states, axis=0, keepdims=True) + 1e-8  # Add small epsilon
        
        # Normalize both current and next states
        normalized_states = (states - self.state_mean) / self.state_std
        normalized_next_states = (next_states - self.state_mean) / self.state_std
        
        logger.info(f"Normalized states: mean={self.state_mean.mean():.4f}, std={self.state_std.mean():.4f}")
        
        return normalized_states, normalized_next_states
    
    def _normalize_rewards(self, rewards: np.ndarray, env_name: str) -> np.ndarray:
        """Normalize rewards based on environment type."""
        if 'antmaze' in env_name.lower():
            # AntMaze rewards are already 0/1, don't normalize
            return rewards
        
        # For continuous control tasks, normalize to reasonable range
        self.reward_mean = np.mean(rewards)
        self.reward_std = np.std(rewards) + 1e-8
        
        # Use a more conservative normalization to preserve reward structure
        normalized_rewards = rewards / self.reward_std
        
        logger.info(f"Normalized rewards: original_mean={self.reward_mean:.4f}, "
                   f"original_std={self.reward_std:.4f}")
        
        return normalized_rewards
    
    def _process_antmaze_rewards(self, rewards: np.ndarray, states: np.ndarray, next_states: np.ndarray) -> np.ndarray:
        """Process AntMaze rewards (typically sparse 0/1 rewards)."""
        # AntMaze rewards are usually 0 everywhere except at goal
        # Keep them as is, but ensure they're in the right format
        processed_rewards = rewards.copy()
        
        # Log reward statistics
        num_positive = np.sum(rewards > 0)
        logger.info(f"AntMaze rewards: {num_positive}/{len(rewards)} positive rewards")
        
        return processed_rewards
    
    def _process_kitchen_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Process Kitchen environment rewards."""
        # Kitchen rewards are typically sparse and task-based
        processed_rewards = rewards.copy()
        
        # Log reward statistics
        unique_rewards = np.unique(rewards)
        logger.info(f"Kitchen rewards: unique values = {unique_rewards}")
        
        return processed_rewards
    
    def _compute_dataset_stats(self, dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        stats = {}
        
        # Basic statistics
        stats['num_transitions'] = len(dataset['states'])
        stats['state_dim'] = dataset['states'].shape[1]
        stats['action_dim'] = dataset['actions'].shape[1]
        
        # State statistics
        stats['state_mean'] = np.mean(dataset['states'], axis=0)
        stats['state_std'] = np.std(dataset['states'], axis=0)
        stats['state_min'] = np.min(dataset['states'], axis=0)
        stats['state_max'] = np.max(dataset['states'], axis=0)
        
        # Action statistics
        stats['action_mean'] = np.mean(dataset['actions'], axis=0)
        stats['action_std'] = np.std(dataset['actions'], axis=0)
        stats['action_min'] = np.min(dataset['actions'], axis=0)
        stats['action_max'] = np.max(dataset['actions'], axis=0)
        
        # Reward statistics
        stats['reward_mean'] = np.mean(dataset['rewards'])
        stats['reward_std'] = np.std(dataset['rewards'])
        stats['reward_min'] = np.min(dataset['rewards'])
        stats['reward_max'] = np.max(dataset['rewards'])
        stats['reward_sum'] = np.sum(dataset['rewards'])
        
        # Terminal statistics
        stats['num_terminals'] = np.sum(dataset['terminals'])
        stats['terminal_ratio'] = stats['num_terminals'] / stats['num_transitions']
        
        # Episode statistics (approximate)
        stats['approx_num_episodes'] = stats['num_terminals']
        if stats['num_terminals'] > 0:
            stats['avg_episode_length'] = stats['num_transitions'] / stats['num_terminals']
        else:
            stats['avg_episode_length'] = stats['num_transitions']
        
        return stats
    
    def _validate_dataset(self, dataset: Dict[str, np.ndarray], env_name: str):
        """Validate dataset integrity and log warnings for potential issues."""
        stats = dataset['stats']
        
        # Check for NaN or infinite values
        for key in ['states', 'actions', 'rewards', 'next_states']:
            data = dataset[key]
            if np.any(np.isnan(data)):
                logger.warning(f"Found NaN values in {key}")
            if np.any(np.isinf(data)):
                logger.warning(f"Found infinite values in {key}")
        
        # Check data consistency
        num_transitions = len(dataset['states'])
        for key in ['actions', 'rewards', 'next_states', 'terminals']:
            if len(dataset[key]) != num_transitions:
                raise ValueError(f"Inconsistent data length: {key} has {len(dataset[key])} "
                               f"entries, expected {num_transitions}")
        
        # Check for reasonable reward distribution
        if stats['reward_std'] == 0:
            logger.warning("All rewards are identical - this may indicate a problem")
        
        # Environment-specific validation
        if 'antmaze' in env_name.lower():
            if stats['reward_max'] > 1.1 or stats['reward_min'] < -0.1:
                logger.warning("AntMaze rewards outside expected [0, 1] range")
        
        logger.info(f"Dataset validation passed for {env_name}")
        logger.info(f"Dataset stats: {stats['num_transitions']} transitions, "
                   f"{stats['approx_num_episodes']} episodes, "
                   f"avg_length={stats['avg_episode_length']:.1f}")
    
    def _create_replay_buffer(self, dataset: Dict[str, np.ndarray], capacity: Optional[int] = None) -> ReplayBuffer:
        """Create and populate replay buffer with dataset."""
        stats = dataset['stats']
        
        if capacity is None:
            capacity = stats['num_transitions']
        
        # Create replay buffer
        buffer = ReplayBuffer(
            capacity=capacity,
            state_dim=stats['state_dim'],
            action_dim=stats['action_dim'],
            device=self.device,
            store_importance_weights=True  # Enable importance weight storage
        )
        
        # Load dataset into buffer using the existing utility function
        buffer = load_dataset_to_buffer(
            dataset=dataset,
            buffer=buffer,
            normalize_rewards=False,  # Already normalized if requested
            importance_weights=None   # Will be computed later by importance estimator
        )
        
        logger.info(f"Created replay buffer with {buffer.size} transitions")
        
        return buffer
    
    def get_env_info(self, env_name: str) -> Dict[str, Any]:
        """Get environment information without loading the full dataset."""
        if gym is None or d4rl is None:
            raise ImportError("D4RL not installed")
        
        env = gym.make(env_name)
        
        info = {
            'env_name': env_name,
            'state_dim': env.observation_space.shape[0],
            'action_dim': env.action_space.shape[0],
            'action_space_low': env.action_space.low,
            'action_space_high': env.action_space.high,
        }
        
        # Get dataset size without loading full dataset
        try:
            dataset = env.get_dataset()
            info['dataset_size'] = len(dataset['observations'])
            info['has_timeouts'] = 'timeouts' in dataset
        except Exception as e:
            logger.warning(f"Could not get dataset info: {e}")
            info['dataset_size'] = None
        
        env.close()
        return info
    
    def list_available_envs(self) -> Dict[str, List[str]]:
        """List all supported D4RL environments."""
        return self.supported_envs.copy()
    
    def clear_cache(self):
        """Clear cached datasets."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared dataset cache")


def load_d4rl_dataset(env_name: str,
                     normalize_states: bool = True,
                     normalize_rewards: bool = True,
                     cache_dir: str = "./d4rl_cache",
                     device: str = "cpu") -> Tuple[Dict[str, np.ndarray], ReplayBuffer]:
    """
    Convenience function to load D4RL dataset with default settings.
    
    Args:
        env_name: D4RL environment name
        normalize_states: Whether to normalize states
        normalize_rewards: Whether to normalize rewards
        cache_dir: Cache directory for datasets
        device: Device for tensors
        
    Returns:
        Tuple of (dataset_dict, replay_buffer)
    """
    loader = D4RLDatasetLoader(
        cache_dir=cache_dir,
        normalize_states=normalize_states,
        normalize_rewards=normalize_rewards,
        device=device
    )
    
    return loader.load_dataset(env_name, load_to_buffer=True)


def create_d4rl_loader(config: Optional[Dict[str, Any]] = None) -> D4RLDatasetLoader:
    """
    Factory function to create D4RL loader with configuration.
    
    Args:
        config: Configuration dictionary with loader parameters
        
    Returns:
        Configured D4RLDatasetLoader instance
    """
    if config is None:
        config = {}
    
    return D4RLDatasetLoader(
        cache_dir=config.get('cache_dir', './d4rl_cache'),
        normalize_states=config.get('normalize_states', True),
        normalize_rewards=config.get('normalize_rewards', True),
        device=config.get('device', 'cpu')
    )


# Example usage and testing
if __name__ == "__main__":
    # Example: Load HalfCheetah medium dataset
    try:
        loader = D4RLDatasetLoader()
        
        # List available environments
        envs = loader.list_available_envs()
        print("Available environments:")
        for category, env_list in envs.items():
            print(f"  {category}: {env_list[:3]}...")  # Show first 3
        
        # Load a small dataset for testing
        env_name = "halfcheetah-random-v2"  # Smaller dataset for testing
        print(f"\nLoading {env_name}...")
        
        dataset, buffer = loader.load_dataset(env_name)
        
        print(f"Dataset loaded successfully!")
        print(f"States shape: {dataset['states'].shape}")
        print(f"Actions shape: {dataset['actions'].shape}")
        print(f"Rewards shape: {dataset['rewards'].shape}")
        print(f"Buffer size: {buffer.size if buffer else 'None'}")
        
        # Print some statistics
        stats = dataset['stats']
        print(f"\nDataset Statistics:")
        print(f"  Transitions: {stats['num_transitions']}")
        print(f"  Episodes: {stats['approx_num_episodes']}")
        print(f"  Avg episode length: {stats['avg_episode_length']:.1f}")
        print(f"  Reward range: [{stats['reward_min']:.3f}, {stats['reward_max']:.3f}]")
        
    except ImportError as e:
        print(f"D4RL not available: {e}")
        print("This is expected in environments without D4RL installed.")
    except Exception as e:
        print(f"Error during testing: {e}")