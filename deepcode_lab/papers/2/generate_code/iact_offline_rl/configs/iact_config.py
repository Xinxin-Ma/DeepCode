"""
IACT Algorithm Configuration
============================

This module provides comprehensive configuration management for the Importance-Aware 
Co-Teaching (IACT) offline reinforcement learning algorithm. It includes hyperparameter
settings, training configurations, and environment-specific adaptations.

Key Components:
- IACTConfig: Main algorithm configuration dataclass
- Default hyperparameters for different environment types
- Configuration validation and factory functions
- Hyperparameter scheduling and adaptation utilities
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch


@dataclass
class IACTConfig:
    """
    Main configuration class for IACT algorithm.
    
    Contains all hyperparameters for:
    - Network architectures (actor, critic)
    - Training parameters (learning rates, batch sizes)
    - IACT-specific settings (importance estimation, co-teaching)
    - Regularization and optimization settings
    """
    
    # Environment Configuration
    state_dim: int = 17  # Default for HalfCheetah
    action_dim: int = 6  # Default for HalfCheetah
    action_space_type: str = "continuous"  # "continuous" or "discrete"
    max_action: float = 1.0
    
    # Network Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "relu"  # "relu", "tanh", "elu"
    layer_norm: bool = True
    dropout_rate: float = 0.1
    
    # Actor Network Specific
    actor_lr: float = 3e-4
    actor_weight_decay: float = 1e-4
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    
    # Critic Network Specific
    critic_lr: float = 3e-4
    critic_weight_decay: float = 1e-4
    double_q: bool = True
    target_update_freq: int = 2
    tau: float = 0.005  # Polyak averaging coefficient
    
    # Training Parameters
    batch_size: int = 256
    max_epochs: int = 1000
    steps_per_epoch: int = 1000
    eval_freq: int = 10
    eval_episodes: int = 10
    
    # IACT-Specific Parameters
    # Importance Estimation (KLIEP)
    n_kernels: int = 100
    kernel_sigma: float = 1.0
    kernel_type: str = "rbf"  # "rbf", "linear", "polynomial"
    importance_update_freq: int = 100  # Update importance weights every N steps
    importance_clip: float = 10.0  # Clip importance weights to prevent instability
    
    # Co-Teaching Parameters
    initial_selection_rate: float = 0.8  # Start with 80% of samples
    final_selection_rate: float = 0.5    # End with 50% of samples
    selection_decay_epochs: int = 500    # Epochs to decay from initial to final
    co_teaching_warmup: int = 50         # Epochs before co-teaching starts
    confidence_threshold: float = 0.1    # Minimum confidence for sample selection
    
    # Behavior Cloning Regularization
    bc_regularization: bool = True
    bc_lambda: float = 0.1  # BC regularization weight
    bc_decay_rate: float = 0.99  # Decay BC weight over time
    bc_min_weight: float = 0.01  # Minimum BC weight
    
    # Optimization Settings
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    gradient_clip: float = 1.0
    lr_scheduler: str = "cosine"  # "cosine", "step", "exponential", "none"
    lr_decay_rate: float = 0.99
    lr_decay_steps: int = 1000
    
    # Regularization
    entropy_regularization: bool = True
    entropy_weight: float = 0.01
    l2_regularization: float = 1e-4
    
    # Logging and Checkpointing
    log_freq: int = 100
    save_freq: int = 100
    max_checkpoints: int = 5
    log_level: str = "INFO"
    
    # Device and Reproducibility
    device: str = "auto"  # "auto", "cpu", "cuda"
    seed: int = 42
    deterministic: bool = False
    
    # Advanced Settings
    use_target_networks: bool = True
    use_double_q: bool = True
    use_layer_norm: bool = True
    use_spectral_norm: bool = False
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self._validate_config()
        self._process_device()
        self._set_derived_parameters()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers >= 1, "num_layers must be at least 1"
        assert 0 < self.initial_selection_rate <= 1.0, "initial_selection_rate must be in (0, 1]"
        assert 0 < self.final_selection_rate <= 1.0, "final_selection_rate must be in (0, 1]"
        assert self.final_selection_rate <= self.initial_selection_rate, \
            "final_selection_rate must be <= initial_selection_rate"
        assert self.bc_lambda >= 0, "bc_lambda must be non-negative"
        assert self.tau > 0 and self.tau <= 1.0, "tau must be in (0, 1]"
    
    def _process_device(self):
        """Process device configuration."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _set_derived_parameters(self):
        """Set parameters derived from other configuration values."""
        # Calculate selection rate decay
        if self.selection_decay_epochs > 0:
            self.selection_decay_rate = (
                (self.final_selection_rate / self.initial_selection_rate) ** 
                (1.0 / self.selection_decay_epochs)
            )
        else:
            self.selection_decay_rate = 1.0
    
    def get_selection_rate(self, epoch: int) -> float:
        """Get current selection rate based on epoch."""
        if epoch < self.co_teaching_warmup:
            return 1.0  # Use all samples during warmup
        
        decay_epoch = epoch - self.co_teaching_warmup
        if decay_epoch >= self.selection_decay_epochs:
            return self.final_selection_rate
        
        rate = self.initial_selection_rate * (self.selection_decay_rate ** decay_epoch)
        return max(rate, self.final_selection_rate)
    
    def get_bc_weight(self, epoch: int) -> float:
        """Get current BC regularization weight based on epoch."""
        if not self.bc_regularization:
            return 0.0
        
        weight = self.bc_lambda * (self.bc_decay_rate ** epoch)
        return max(weight, self.bc_min_weight)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IACTConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Environment-Specific Default Configurations
DEFAULT_CONFIGS = {
    # MuJoCo Locomotion Tasks
    "halfcheetah": IACTConfig(
        state_dim=17,
        action_dim=6,
        max_action=1.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        bc_lambda=0.1,
        n_kernels=100,
        kernel_sigma=1.0,
    ),
    
    "walker2d": IACTConfig(
        state_dim=17,
        action_dim=6,
        max_action=1.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        bc_lambda=0.15,  # Slightly higher BC for Walker2d
        n_kernels=100,
        kernel_sigma=0.8,
    ),
    
    "hopper": IACTConfig(
        state_dim=11,
        action_dim=3,
        max_action=1.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        bc_lambda=0.2,  # Higher BC for Hopper (more unstable)
        n_kernels=80,
        kernel_sigma=0.6,
    ),
    
    "ant": IACTConfig(
        state_dim=111,
        action_dim=8,
        max_action=1.0,
        actor_lr=1e-4,  # Lower LR for high-dimensional Ant
        critic_lr=3e-4,
        bc_lambda=0.05,  # Lower BC for complex Ant dynamics
        n_kernels=150,  # More kernels for high-dimensional state
        kernel_sigma=1.5,
        hidden_dim=512,  # Larger networks for Ant
    ),
    
    # AntMaze Navigation Tasks
    "antmaze": IACTConfig(
        state_dim=29,
        action_dim=8,
        max_action=1.0,
        actor_lr=1e-4,
        critic_lr=3e-4,
        bc_lambda=0.3,  # High BC for sparse reward navigation
        n_kernels=120,
        kernel_sigma=2.0,  # Larger sigma for navigation
        initial_selection_rate=0.9,  # More conservative selection
        final_selection_rate=0.6,
        entropy_weight=0.02,  # Higher entropy for exploration
    ),
    
    # Kitchen Manipulation Tasks
    "kitchen": IACTConfig(
        state_dim=60,
        action_dim=9,
        max_action=1.0,
        actor_lr=1e-4,
        critic_lr=3e-4,
        bc_lambda=0.25,  # High BC for manipulation
        n_kernels=100,
        kernel_sigma=1.2,
        hidden_dim=512,  # Larger networks for manipulation
        initial_selection_rate=0.85,
        final_selection_rate=0.55,
    ),
    
    # Adroit Hand Manipulation
    "adroit": IACTConfig(
        state_dim=24,
        action_dim=24,
        max_action=1.0,
        actor_lr=1e-4,
        critic_lr=3e-4,
        bc_lambda=0.4,  # Very high BC for dexterous manipulation
        n_kernels=100,
        kernel_sigma=0.8,
        hidden_dim=512,
        initial_selection_rate=0.9,
        final_selection_rate=0.7,
        co_teaching_warmup=100,  # Longer warmup for complex tasks
    ),
}

# Dataset Quality Specific Configurations
DATASET_CONFIGS = {
    "random": {
        "bc_lambda": 0.5,  # High BC for random data
        "initial_selection_rate": 0.95,
        "final_selection_rate": 0.8,
        "importance_clip": 5.0,  # Lower clip for noisy data
    },
    
    "medium": {
        "bc_lambda": 0.1,
        "initial_selection_rate": 0.8,
        "final_selection_rate": 0.5,
        "importance_clip": 10.0,
    },
    
    "expert": {
        "bc_lambda": 0.05,  # Low BC for expert data
        "initial_selection_rate": 0.7,
        "final_selection_rate": 0.4,
        "importance_clip": 15.0,  # Higher clip for clean data
    },
    
    "medium-expert": {
        "bc_lambda": 0.08,
        "initial_selection_rate": 0.75,
        "final_selection_rate": 0.45,
        "importance_clip": 12.0,
    },
    
    "medium-replay": {
        "bc_lambda": 0.15,
        "initial_selection_rate": 0.85,
        "final_selection_rate": 0.6,
        "importance_clip": 8.0,
    },
}


def get_iact_config(
    env_name: str, 
    dataset_type: str = "medium",
    **overrides
) -> IACTConfig:
    """
    Get IACT configuration for specific environment and dataset.
    
    Args:
        env_name: Environment name (e.g., "halfcheetah", "walker2d")
        dataset_type: Dataset quality ("random", "medium", "expert", etc.)
        **overrides: Additional configuration overrides
    
    Returns:
        IACTConfig: Configured IACT algorithm settings
    """
    # Get base environment config
    env_key = env_name.lower().split('-')[0]  # Extract base env name
    if env_key not in DEFAULT_CONFIGS:
        # Use halfcheetah as default for unknown environments
        base_config = DEFAULT_CONFIGS["halfcheetah"]
    else:
        base_config = DEFAULT_CONFIGS[env_key]
    
    # Create config dictionary
    config_dict = base_config.to_dict()
    
    # Apply dataset-specific modifications
    if dataset_type in DATASET_CONFIGS:
        dataset_overrides = DATASET_CONFIGS[dataset_type]
        config_dict.update(dataset_overrides)
    
    # Apply user overrides
    config_dict.update(overrides)
    
    return IACTConfig.from_dict(config_dict)


def create_ablation_config(
    base_config: IACTConfig,
    ablation_type: str
) -> IACTConfig:
    """
    Create configuration for ablation studies.
    
    Args:
        base_config: Base IACT configuration
        ablation_type: Type of ablation study
            - "no_importance": Disable importance weighting
            - "no_co_teaching": Disable co-teaching
            - "no_bc": Disable behavior cloning
            - "single_policy": Use single policy instead of dual
    
    Returns:
        IACTConfig: Modified configuration for ablation
    """
    config_dict = base_config.to_dict()
    
    if ablation_type == "no_importance":
        config_dict.update({
            "n_kernels": 0,  # Disable importance estimation
            "importance_update_freq": float('inf'),
        })
    
    elif ablation_type == "no_co_teaching":
        config_dict.update({
            "initial_selection_rate": 1.0,  # Use all samples
            "final_selection_rate": 1.0,
            "co_teaching_warmup": 0,
        })
    
    elif ablation_type == "no_bc":
        config_dict.update({
            "bc_regularization": False,
            "bc_lambda": 0.0,
        })
    
    elif ablation_type == "single_policy":
        config_dict.update({
            "initial_selection_rate": 1.0,  # No sample selection
            "final_selection_rate": 1.0,
            "co_teaching_warmup": 0,
        })
    
    return IACTConfig.from_dict(config_dict)


def get_hyperparameter_grid() -> Dict[str, List[Any]]:
    """
    Get hyperparameter grid for hyperparameter tuning.
    
    Returns:
        Dict: Grid of hyperparameters to search over
    """
    return {
        "actor_lr": [1e-4, 3e-4, 1e-3],
        "critic_lr": [1e-4, 3e-4, 1e-3],
        "bc_lambda": [0.05, 0.1, 0.2, 0.3],
        "initial_selection_rate": [0.7, 0.8, 0.9],
        "final_selection_rate": [0.4, 0.5, 0.6],
        "n_kernels": [50, 100, 150],
        "kernel_sigma": [0.5, 1.0, 1.5, 2.0],
        "hidden_dim": [256, 512],
        "batch_size": [128, 256, 512],
        "entropy_weight": [0.0, 0.01, 0.02],
    }


def validate_config(config: IACTConfig) -> bool:
    """
    Validate IACT configuration.
    
    Args:
        config: IACT configuration to validate
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Basic parameter validation is done in __post_init__
        # Additional validation can be added here
        
        # Check device availability
        if config.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            config.device = "cpu"
        
        # Check parameter combinations
        if config.double_q and not config.use_double_q:
            print("Warning: double_q=True but use_double_q=False")
        
        if config.bc_regularization and config.bc_lambda <= 0:
            print("Warning: BC regularization enabled but bc_lambda <= 0")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def print_config_summary(config: IACTConfig):
    """
    Print a summary of the IACT configuration.
    
    Args:
        config: IACT configuration to summarize
    """
    print("=" * 60)
    print("IACT Algorithm Configuration Summary")
    print("=" * 60)
    
    print(f"Environment: {config.state_dim}D state, {config.action_dim}D action")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    
    print("\nNetwork Architecture:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Activation: {config.activation}")
    print(f"  Layer norm: {config.layer_norm}")
    
    print("\nTraining Parameters:")
    print(f"  Actor LR: {config.actor_lr}")
    print(f"  Critic LR: {config.critic_lr}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    
    print("\nIACT-Specific:")
    print(f"  Importance kernels: {config.n_kernels}")
    print(f"  Kernel sigma: {config.kernel_sigma}")
    print(f"  Selection rate: {config.initial_selection_rate} → {config.final_selection_rate}")
    print(f"  BC lambda: {config.bc_lambda}")
    print(f"  Co-teaching warmup: {config.co_teaching_warmup}")
    
    print("=" * 60)


# Factory function for easy configuration creation
def create_iact_config(
    env_name: str,
    dataset_type: str = "medium",
    **kwargs
) -> IACTConfig:
    """
    Factory function to create IACT configuration.
    
    Args:
        env_name: Environment name
        dataset_type: Dataset quality type
        **kwargs: Additional configuration overrides
    
    Returns:
        IACTConfig: Configured IACT settings
    """
    config = get_iact_config(env_name, dataset_type, **kwargs)
    
    if not validate_config(config):
        raise ValueError("Invalid configuration created")
    
    return config


if __name__ == "__main__":
    # Example usage and testing
    print("Testing IACT Configuration System")
    print("=" * 50)
    
    # Test default configuration
    config = IACTConfig()
    print("Default configuration created successfully")
    print_config_summary(config)
    
    # Test environment-specific configuration
    print("\nTesting environment-specific configs:")
    for env_name in ["halfcheetah", "walker2d", "hopper", "ant"]:
        config = get_iact_config(env_name, "medium")
        print(f"✓ {env_name}: {config.state_dim}D state, {config.action_dim}D action")
    
    # Test ablation configurations
    print("\nTesting ablation configurations:")
    base_config = get_iact_config("halfcheetah", "medium")
    for ablation in ["no_importance", "no_co_teaching", "no_bc", "single_policy"]:
        ablation_config = create_ablation_config(base_config, ablation)
        print(f"✓ {ablation} ablation created")
    
    print("\nAll tests passed! ✓")