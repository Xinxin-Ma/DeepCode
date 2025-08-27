"""
Environment-specific configurations for D4RL benchmarks and other RL environments.
This module provides standardized configurations for different environments used in
offline reinforcement learning experiments.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EnvironmentConfig:
    """Base configuration class for RL environments."""
    name: str
    state_dim: int
    action_dim: int
    action_space_type: str  # 'continuous' or 'discrete'
    action_range: Optional[tuple] = None  # (min, max) for continuous actions
    max_episode_steps: int = 1000
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    normalize_states: bool = True
    normalize_rewards: bool = False
    
    # D4RL specific
    is_d4rl: bool = False
    d4rl_score_range: Optional[tuple] = None  # (min_score, max_score) for normalization
    
    # Environment-specific preprocessing
    state_preprocessing: Optional[str] = None
    reward_preprocessing: Optional[str] = None
    
    # Evaluation settings
    eval_episodes: int = 10
    eval_deterministic: bool = True
    eval_render: bool = False


# D4RL Environment Configurations
D4RL_CONFIGS = {
    # MuJoCo Locomotion Tasks
    'halfcheetah-medium-v2': EnvironmentConfig(
        name='halfcheetah-medium-v2',
        state_dim=17,
        action_dim=6,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 12135.0),  # Approximate range for normalization
        eval_episodes=10
    ),
    
    'halfcheetah-medium-expert-v2': EnvironmentConfig(
        name='halfcheetah-medium-expert-v2',
        state_dim=17,
        action_dim=6,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 12135.0),
        eval_episodes=10
    ),
    
    'halfcheetah-medium-replay-v2': EnvironmentConfig(
        name='halfcheetah-medium-replay-v2',
        state_dim=17,
        action_dim=6,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 12135.0),
        eval_episodes=10
    ),
    
    'walker2d-medium-v2': EnvironmentConfig(
        name='walker2d-medium-v2',
        state_dim=17,
        action_dim=6,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 5000.0),
        eval_episodes=10
    ),
    
    'walker2d-medium-expert-v2': EnvironmentConfig(
        name='walker2d-medium-expert-v2',
        state_dim=17,
        action_dim=6,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 5000.0),
        eval_episodes=10
    ),
    
    'walker2d-medium-replay-v2': EnvironmentConfig(
        name='walker2d-medium-replay-v2',
        state_dim=17,
        action_dim=6,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 5000.0),
        eval_episodes=10
    ),
    
    'hopper-medium-v2': EnvironmentConfig(
        name='hopper-medium-v2',
        state_dim=11,
        action_dim=3,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 3600.0),
        eval_episodes=10
    ),
    
    'hopper-medium-expert-v2': EnvironmentConfig(
        name='hopper-medium-expert-v2',
        state_dim=11,
        action_dim=3,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 3600.0),
        eval_episodes=10
    ),
    
    'hopper-medium-replay-v2': EnvironmentConfig(
        name='hopper-medium-replay-v2',
        state_dim=11,
        action_dim=3,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 3600.0),
        eval_episodes=10
    ),
    
    # AntMaze Tasks
    'antmaze-umaze-v2': EnvironmentConfig(
        name='antmaze-umaze-v2',
        state_dim=29,
        action_dim=8,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=700,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 1.0),  # Success rate based
        eval_episodes=100  # More episodes for sparse reward tasks
    ),
    
    'antmaze-umaze-diverse-v2': EnvironmentConfig(
        name='antmaze-umaze-diverse-v2',
        state_dim=29,
        action_dim=8,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=700,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 1.0),
        eval_episodes=100
    ),
    
    'antmaze-medium-play-v2': EnvironmentConfig(
        name='antmaze-medium-play-v2',
        state_dim=29,
        action_dim=8,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 1.0),
        eval_episodes=100
    ),
    
    'antmaze-medium-diverse-v2': EnvironmentConfig(
        name='antmaze-medium-diverse-v2',
        state_dim=29,
        action_dim=8,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 1.0),
        eval_episodes=100
    ),
    
    'antmaze-large-play-v2': EnvironmentConfig(
        name='antmaze-large-play-v2',
        state_dim=29,
        action_dim=8,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 1.0),
        eval_episodes=100
    ),
    
    'antmaze-large-diverse-v2': EnvironmentConfig(
        name='antmaze-large-diverse-v2',
        state_dim=29,
        action_dim=8,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=1000,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 1.0),
        eval_episodes=100
    ),
    
    # Kitchen Tasks
    'kitchen-complete-v1': EnvironmentConfig(
        name='kitchen-complete-v1',
        state_dim=60,
        action_dim=9,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=280,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 4.0),  # 4 tasks to complete
        eval_episodes=50
    ),
    
    'kitchen-partial-v1': EnvironmentConfig(
        name='kitchen-partial-v1',
        state_dim=60,
        action_dim=9,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=280,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 4.0),
        eval_episodes=50
    ),
    
    'kitchen-mixed-v1': EnvironmentConfig(
        name='kitchen-mixed-v1',
        state_dim=60,
        action_dim=9,
        action_space_type='continuous',
        action_range=(-1.0, 1.0),
        max_episode_steps=280,
        reward_scale=1.0,
        normalize_states=True,
        normalize_rewards=False,
        is_d4rl=True,
        d4rl_score_range=(0.0, 4.0),
        eval_episodes=50
    ),
}


# Environment groups for batch experiments
ENV_GROUPS = {
    'mujoco_locomotion': [
        'halfcheetah-medium-v2',
        'halfcheetah-medium-expert-v2',
        'halfcheetah-medium-replay-v2',
        'walker2d-medium-v2',
        'walker2d-medium-expert-v2',
        'walker2d-medium-replay-v2',
        'hopper-medium-v2',
        'hopper-medium-expert-v2',
        'hopper-medium-replay-v2'
    ],
    
    'antmaze': [
        'antmaze-umaze-v2',
        'antmaze-umaze-diverse-v2',
        'antmaze-medium-play-v2',
        'antmaze-medium-diverse-v2',
        'antmaze-large-play-v2',
        'antmaze-large-diverse-v2'
    ],
    
    'kitchen': [
        'kitchen-complete-v1',
        'kitchen-partial-v1',
        'kitchen-mixed-v1'
    ],
    
    'all_d4rl': list(D4RL_CONFIGS.keys())
}


def get_env_config(env_name: str) -> EnvironmentConfig:
    """
    Get environment configuration for a given environment name.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        EnvironmentConfig object with environment-specific settings
        
    Raises:
        ValueError: If environment name is not supported
    """
    if env_name in D4RL_CONFIGS:
        return D4RL_CONFIGS[env_name]
    else:
        raise ValueError(f"Environment '{env_name}' not supported. "
                        f"Available environments: {list(D4RL_CONFIGS.keys())}")


def get_env_group(group_name: str) -> List[str]:
    """
    Get list of environment names for a given group.
    
    Args:
        group_name: Name of the environment group
        
    Returns:
        List of environment names in the group
        
    Raises:
        ValueError: If group name is not supported
    """
    if group_name in ENV_GROUPS:
        return ENV_GROUPS[group_name]
    else:
        raise ValueError(f"Environment group '{group_name}' not supported. "
                        f"Available groups: {list(ENV_GROUPS.keys())}")


def create_env_config(
    name: str,
    state_dim: int,
    action_dim: int,
    action_space_type: str = 'continuous',
    **kwargs
) -> EnvironmentConfig:
    """
    Create a custom environment configuration.
    
    Args:
        name: Environment name
        state_dim: Dimensionality of state space
        action_dim: Dimensionality of action space
        action_space_type: Type of action space ('continuous' or 'discrete')
        **kwargs: Additional configuration parameters
        
    Returns:
        EnvironmentConfig object
    """
    return EnvironmentConfig(
        name=name,
        state_dim=state_dim,
        action_dim=action_dim,
        action_space_type=action_space_type,
        **kwargs
    )


def get_all_env_names() -> List[str]:
    """Get list of all supported environment names."""
    return list(D4RL_CONFIGS.keys())


def get_env_info(env_name: str) -> Dict[str, Any]:
    """
    Get basic information about an environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Dictionary with environment information
    """
    config = get_env_config(env_name)
    return {
        'name': config.name,
        'state_dim': config.state_dim,
        'action_dim': config.action_dim,
        'action_space_type': config.action_space_type,
        'action_range': config.action_range,
        'max_episode_steps': config.max_episode_steps,
        'is_d4rl': config.is_d4rl,
        'd4rl_score_range': config.d4rl_score_range
    }


def validate_env_config(config: EnvironmentConfig) -> bool:
    """
    Validate an environment configuration.
    
    Args:
        config: EnvironmentConfig to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.state_dim <= 0:
        raise ValueError("state_dim must be positive")
    
    if config.action_dim <= 0:
        raise ValueError("action_dim must be positive")
    
    if config.action_space_type not in ['continuous', 'discrete']:
        raise ValueError("action_space_type must be 'continuous' or 'discrete'")
    
    if config.action_space_type == 'continuous' and config.action_range is None:
        raise ValueError("action_range must be specified for continuous action spaces")
    
    if config.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    
    if config.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    
    return True


# Default configuration for experiments
DEFAULT_EXPERIMENT_CONFIG = {
    'env_name': 'halfcheetah-medium-v2',
    'normalize_states': True,
    'normalize_rewards': False,
    'eval_episodes': 10,
    'eval_deterministic': True,
    'eval_render': False
}


def get_experiment_config(env_name: str, **overrides) -> Dict[str, Any]:
    """
    Get experiment configuration for an environment with optional overrides.
    
    Args:
        env_name: Name of the environment
        **overrides: Configuration parameters to override
        
    Returns:
        Dictionary with experiment configuration
    """
    env_config = get_env_config(env_name)
    
    experiment_config = {
        'env_name': env_config.name,
        'state_dim': env_config.state_dim,
        'action_dim': env_config.action_dim,
        'action_space_type': env_config.action_space_type,
        'action_range': env_config.action_range,
        'max_episode_steps': env_config.max_episode_steps,
        'normalize_states': env_config.normalize_states,
        'normalize_rewards': env_config.normalize_rewards,
        'eval_episodes': env_config.eval_episodes,
        'eval_deterministic': env_config.eval_deterministic,
        'eval_render': env_config.eval_render,
        'is_d4rl': env_config.is_d4rl,
        'd4rl_score_range': env_config.d4rl_score_range
    }
    
    # Apply overrides
    experiment_config.update(overrides)
    
    return experiment_config


if __name__ == "__main__":
    # Example usage and testing
    print("Available environments:")
    for env_name in get_all_env_names():
        print(f"  {env_name}")
    
    print("\nEnvironment groups:")
    for group_name, envs in ENV_GROUPS.items():
        print(f"  {group_name}: {len(envs)} environments")
    
    # Test configuration retrieval
    env_name = 'halfcheetah-medium-v2'
    config = get_env_config(env_name)
    print(f"\nConfiguration for {env_name}:")
    print(f"  State dim: {config.state_dim}")
    print(f"  Action dim: {config.action_dim}")
    print(f"  Action range: {config.action_range}")
    print(f"  Max episode steps: {config.max_episode_steps}")
    print(f"  D4RL score range: {config.d4rl_score_range}")
    
    # Test experiment configuration
    exp_config = get_experiment_config(env_name, eval_episodes=20)
    print(f"\nExperiment configuration:")
    print(f"  Eval episodes: {exp_config['eval_episodes']}")
    print(f"  Normalize states: {exp_config['normalize_states']}")