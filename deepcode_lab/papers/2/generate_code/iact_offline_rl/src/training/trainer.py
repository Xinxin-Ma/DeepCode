"""
Main Training Pipeline for IACT Algorithm

This module implements the main training coordinator that orchestrates the IACT training process,
including importance weight estimation, co-teaching sample selection, and dual policy training.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time
from collections import defaultdict, deque

from ..algorithms.iact import IACTAlgorithm, create_iact_algorithm, TrainingMetrics
from ..algorithms.importance_estimator import ImportanceEstimator
from ..data.replay_buffer import ReplayBuffer


@dataclass
class TrainingConfig:
    """Configuration for IACT training pipeline"""
    # Training parameters
    max_epochs: int = 1000
    batch_size: int = 256
    eval_frequency: int = 10
    save_frequency: int = 50
    
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    
    # Co-teaching parameters
    initial_selection_rate: float = 0.8
    final_selection_rate: float = 0.5
    selection_decay_epochs: int = 500
    
    # Importance estimation
    importance_update_frequency: int = 100
    importance_estimation_samples: int = 10000
    
    # Regularization
    bc_regularizer_weight: float = 0.1
    target_update_frequency: int = 2
    
    # Evaluation
    eval_episodes: int = 10
    
    # Logging
    log_frequency: int = 10
    tensorboard_log: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch"""
    epoch: int
    actor_a_loss: float
    actor_b_loss: float
    critic_a_loss: float
    critic_b_loss: float
    bc_loss_a: float
    bc_loss_b: float
    selection_rate: float
    importance_weights_mean: float
    importance_weights_std: float
    training_time: float
    eval_return: Optional[float] = None
    eval_success_rate: Optional[float] = None


class IACTTrainer:
    """
    Main trainer for IACT algorithm that coordinates the training process.
    
    This trainer implements the complete IACT training pipeline including:
    - Importance weight estimation using KLIEP
    - Co-teaching sample selection between dual policies
    - Alternating training of dual actor-critic pairs
    - Evaluation and checkpointing
    """
    
    def __init__(
        self,
        algorithm: IACTAlgorithm,
        replay_buffer: ReplayBuffer,
        config: Optional[TrainingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize IACT trainer.
        
        Args:
            algorithm: IACT algorithm instance
            replay_buffer: Replay buffer containing offline dataset
            config: Training configuration
            logger: Logger for training progress
        """
        self.algorithm = algorithm
        self.replay_buffer = replay_buffer
        self.config = config or TrainingConfig()
        self.logger = logger or self._setup_logger()
        
        # Training state
        self.current_epoch = 0
        self.best_eval_return = -np.inf
        self.training_history = []
        self.metrics_buffer = deque(maxlen=100)
        
        # Selection rate scheduling
        self.selection_rate_scheduler = self._create_selection_rate_scheduler()
        
        # Device setup
        self.device = torch.device(self.config.device)
        self.algorithm.to(self.device)
        
        self.logger.info(f"IACT Trainer initialized with device: {self.device}")
        self.logger.info(f"Training configuration: {asdict(self.config)}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger("IACTTrainer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_selection_rate_scheduler(self) -> callable:
        """Create selection rate scheduler for co-teaching"""
        def scheduler(epoch: int) -> float:
            if epoch >= self.config.selection_decay_epochs:
                return self.config.final_selection_rate
            
            # Linear decay from initial to final selection rate
            decay_progress = epoch / self.config.selection_decay_epochs
            rate = (
                self.config.initial_selection_rate - 
                decay_progress * (self.config.initial_selection_rate - self.config.final_selection_rate)
            )
            return max(rate, self.config.final_selection_rate)
        
        return scheduler
    
    def train(
        self,
        env: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None
    ) -> List[EpochMetrics]:
        """
        Main training loop for IACT algorithm.
        
        Args:
            env: Environment for evaluation (optional)
            checkpoint_dir: Directory to save checkpoints (optional)
            
        Returns:
            List of epoch metrics
        """
        self.logger.info("Starting IACT training...")
        self.logger.info(f"Dataset size: {len(self.replay_buffer)}")
        self.logger.info(f"Training for {self.config.max_epochs} epochs")
        
        # Initial importance weight estimation
        self._update_importance_weights()
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Get current selection rate
            current_selection_rate = self.selection_rate_scheduler(epoch)
            
            # Training step
            epoch_metrics = self._train_epoch(current_selection_rate)
            epoch_metrics.training_time = time.time() - epoch_start_time
            
            # Update importance weights periodically
            if epoch % self.config.importance_update_frequency == 0:
                self._update_importance_weights()
            
            # Evaluation
            if env is not None and epoch % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate(env)
                epoch_metrics.eval_return = eval_metrics.get('mean_return', None)
                epoch_metrics.eval_success_rate = eval_metrics.get('success_rate', None)
                
                # Update best model
                if epoch_metrics.eval_return > self.best_eval_return:
                    self.best_eval_return = epoch_metrics.eval_return
                    if checkpoint_dir:
                        self._save_best_checkpoint(checkpoint_dir)
            
            # Save checkpoint
            if checkpoint_dir and epoch % self.config.save_frequency == 0:
                self._save_checkpoint(checkpoint_dir, epoch)
            
            # Logging
            if epoch % self.config.log_frequency == 0:
                self._log_epoch_metrics(epoch_metrics)
            
            self.training_history.append(epoch_metrics)
            self.metrics_buffer.append(epoch_metrics)
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def _train_epoch(self, selection_rate: float) -> EpochMetrics:
        """
        Train for one epoch with co-teaching.
        
        Args:
            selection_rate: Current selection rate for co-teaching
            
        Returns:
            Epoch metrics
        """
        epoch_metrics = {
            'actor_a_losses': [],
            'actor_b_losses': [],
            'critic_a_losses': [],
            'critic_b_losses': [],
            'bc_losses_a': [],
            'bc_losses_b': []
        }
        
        # Number of training steps per epoch
        steps_per_epoch = len(self.replay_buffer) // self.config.batch_size
        
        for step in range(steps_per_epoch):
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.config.batch_size)
            batch = self._batch_to_device(batch)
            
            # Training step with co-teaching
            training_metrics = self.algorithm.train_step(
                batch=batch,
                selection_rate=selection_rate
            )
            
            # Collect metrics
            epoch_metrics['actor_a_losses'].append(training_metrics.actor_a_loss)
            epoch_metrics['actor_b_losses'].append(training_metrics.actor_b_loss)
            epoch_metrics['critic_a_losses'].append(training_metrics.critic_a_loss)
            epoch_metrics['critic_b_losses'].append(training_metrics.critic_b_loss)
            epoch_metrics['bc_losses_a'].append(training_metrics.bc_loss_a)
            epoch_metrics['bc_losses_b'].append(training_metrics.bc_loss_b)
        
        # Aggregate metrics
        return EpochMetrics(
            epoch=self.current_epoch,
            actor_a_loss=np.mean(epoch_metrics['actor_a_losses']),
            actor_b_loss=np.mean(epoch_metrics['actor_b_losses']),
            critic_a_loss=np.mean(epoch_metrics['critic_a_losses']),
            critic_b_loss=np.mean(epoch_metrics['critic_b_losses']),
            bc_loss_a=np.mean(epoch_metrics['bc_losses_a']),
            bc_loss_b=np.mean(epoch_metrics['bc_losses_b']),
            selection_rate=selection_rate,
            importance_weights_mean=self.algorithm.importance_weights.mean().item(),
            importance_weights_std=self.algorithm.importance_weights.std().item(),
            training_time=0.0  # Will be set by caller
        )
    
    def _update_importance_weights(self):
        """Update importance weights using KLIEP estimation"""
        self.logger.info("Updating importance weights...")
        
        # Sample states for importance estimation
        n_samples = min(
            self.config.importance_estimation_samples,
            len(self.replay_buffer)
        )
        
        batch = self.replay_buffer.sample(n_samples)
        states = batch['states']
        
        # Update importance weights in algorithm
        self.algorithm.update_importance_weights(
            states=states,
            behavior_states=states  # In offline RL, behavior states are the dataset states
        )
        
        weights_stats = {
            'mean': self.algorithm.importance_weights.mean().item(),
            'std': self.algorithm.importance_weights.std().item(),
            'min': self.algorithm.importance_weights.min().item(),
            'max': self.algorithm.importance_weights.max().item()
        }
        
        self.logger.info(f"Importance weights updated: {weights_stats}")
    
    def _evaluate(self, env: Any) -> Dict[str, float]:
        """
        Evaluate current policy in environment.
        
        Args:
            env: Environment for evaluation
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating policy...")
        
        eval_metrics = self.algorithm.evaluate_policy(
            env=env,
            num_episodes=self.config.eval_episodes
        )
        
        self.logger.info(f"Evaluation results: {eval_metrics}")
        return eval_metrics
    
    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def _log_epoch_metrics(self, metrics: EpochMetrics):
        """Log epoch metrics"""
        self.logger.info(
            f"Epoch {metrics.epoch:4d} | "
            f"Actor A: {metrics.actor_a_loss:.4f} | "
            f"Actor B: {metrics.actor_b_loss:.4f} | "
            f"Critic A: {metrics.critic_a_loss:.4f} | "
            f"Critic B: {metrics.critic_b_loss:.4f} | "
            f"BC A: {metrics.bc_loss_a:.4f} | "
            f"BC B: {metrics.bc_loss_b:.4f} | "
            f"Selection Rate: {metrics.selection_rate:.3f} | "
            f"Importance μ: {metrics.importance_weights_mean:.3f} | "
            f"Time: {metrics.training_time:.2f}s"
        )
        
        if metrics.eval_return is not None:
            self.logger.info(
                f"Evaluation - Return: {metrics.eval_return:.2f} | "
                f"Success Rate: {metrics.eval_success_rate:.3f}"
            )
    
    def _save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """Save training checkpoint"""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'algorithm_state': self.algorithm.state_dict(),
            'training_history': self.training_history,
            'best_eval_return': self.best_eval_return,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_checkpoint(self, checkpoint_dir: str):
        """Save best model checkpoint"""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        self.algorithm.save_checkpoint(best_path)
        self.logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.algorithm.load_state_dict(checkpoint['algorithm_state'])
        self.training_history = checkpoint['training_history']
        self.best_eval_return = checkpoint['best_eval_return']
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        if not self.training_history:
            return {}
        
        recent_metrics = list(self.metrics_buffer)
        
        summary = {
            'total_epochs': len(self.training_history),
            'best_eval_return': self.best_eval_return,
            'current_epoch': self.current_epoch,
            'recent_actor_a_loss': np.mean([m.actor_a_loss for m in recent_metrics]),
            'recent_actor_b_loss': np.mean([m.actor_b_loss for m in recent_metrics]),
            'recent_critic_a_loss': np.mean([m.critic_a_loss for m in recent_metrics]),
            'recent_critic_b_loss': np.mean([m.critic_b_loss for m in recent_metrics]),
            'recent_bc_loss_a': np.mean([m.bc_loss_a for m in recent_metrics]),
            'recent_bc_loss_b': np.mean([m.bc_loss_b for m in recent_metrics]),
            'current_selection_rate': self.selection_rate_scheduler(self.current_epoch),
            'importance_weights_stats': {
                'mean': self.algorithm.importance_weights.mean().item(),
                'std': self.algorithm.importance_weights.std().item()
            }
        }
        
        return summary


class CoTeachingTrainer:
    """
    Specialized trainer focusing on co-teaching mechanism.
    
    This trainer provides detailed control over the co-teaching process
    and can be used for ablation studies.
    """
    
    def __init__(
        self,
        algorithm: IACTAlgorithm,
        config: Optional[TrainingConfig] = None
    ):
        self.algorithm = algorithm
        self.config = config or TrainingConfig()
        self.co_teaching_stats = defaultdict(list)
    
    def train_with_co_teaching(
        self,
        batch: Dict[str, torch.Tensor],
        selection_rate: float,
        track_selection_stats: bool = True
    ) -> TrainingMetrics:
        """
        Train with detailed co-teaching tracking.
        
        Args:
            batch: Training batch
            selection_rate: Selection rate for co-teaching
            track_selection_stats: Whether to track selection statistics
            
        Returns:
            Training metrics with co-teaching details
        """
        if track_selection_stats:
            # Track selection statistics before training
            self._track_selection_stats(batch, selection_rate)
        
        # Perform training step
        metrics = self.algorithm.train_step(batch, selection_rate)
        
        return metrics
    
    def _track_selection_stats(
        self,
        batch: Dict[str, torch.Tensor],
        selection_rate: float
    ):
        """Track co-teaching selection statistics"""
        states = batch['states']
        actions = batch['actions']
        
        # Get importance weights
        importance_weights = self.algorithm.importance_weights[:len(states)]
        
        # Get confidence scores from both actors
        confidence_a = self.algorithm.dual_actor.actor_a.get_confidence_scores(states, actions)
        confidence_b = self.algorithm.dual_actor.actor_b.get_confidence_scores(states, actions)
        
        # Track statistics
        self.co_teaching_stats['importance_weights'].append(importance_weights.mean().item())
        self.co_teaching_stats['confidence_a'].append(confidence_a.mean().item())
        self.co_teaching_stats['confidence_b'].append(confidence_b.mean().item())
        self.co_teaching_stats['selection_rate'].append(selection_rate)
    
    def get_co_teaching_stats(self) -> Dict[str, List[float]]:
        """Get co-teaching statistics"""
        return dict(self.co_teaching_stats)


def create_trainer(
    state_dim: int,
    action_dim: int,
    replay_buffer: ReplayBuffer,
    config: Optional[Dict] = None,
    training_config: Optional[TrainingConfig] = None
) -> IACTTrainer:
    """
    Factory function to create IACT trainer.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        replay_buffer: Replay buffer with offline data
        config: Algorithm configuration
        training_config: Training configuration
        
    Returns:
        Configured IACT trainer
    """
    # Create algorithm
    algorithm = create_iact_algorithm(config or {
        'state_dim': state_dim,
        'action_dim': action_dim
    })
    
    # Create trainer
    trainer = IACTTrainer(
        algorithm=algorithm,
        replay_buffer=replay_buffer,
        config=training_config or TrainingConfig()
    )
    
    return trainer


def create_co_teaching_trainer(
    algorithm: IACTAlgorithm,
    config: Optional[TrainingConfig] = None
) -> CoTeachingTrainer:
    """
    Factory function to create co-teaching trainer.
    
    Args:
        algorithm: IACT algorithm instance
        config: Training configuration
        
    Returns:
        Configured co-teaching trainer
    """
    return CoTeachingTrainer(algorithm=algorithm, config=config)


# Training utilities
class TrainingLogger:
    """Enhanced logger for IACT training"""
    
    def __init__(self, log_dir: str = "logs"):
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_dir = log_dir
        self.metrics_log = []
    
    def log_metrics(self, metrics: EpochMetrics):
        """Log epoch metrics"""
        self.metrics_log.append(asdict(metrics))
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save metrics to file"""
        import json
        import os
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_log:
                return
            
            epochs = [m['epoch'] for m in self.metrics_log]
            actor_a_losses = [m['actor_a_loss'] for m in self.metrics_log]
            actor_b_losses = [m['actor_b_loss'] for m in self.metrics_log]
            critic_a_losses = [m['critic_a_loss'] for m in self.metrics_log]
            critic_b_losses = [m['critic_b_loss'] for m in self.metrics_log]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].plot(epochs, actor_a_losses, label='Actor A')
            axes[0, 0].plot(epochs, actor_b_losses, label='Actor B')
            axes[0, 0].set_title('Actor Losses')
            axes[0, 0].legend()
            
            axes[0, 1].plot(epochs, critic_a_losses, label='Critic A')
            axes[0, 1].plot(epochs, critic_b_losses, label='Critic B')
            axes[0, 1].set_title('Critic Losses')
            axes[0, 1].legend()
            
            selection_rates = [m['selection_rate'] for m in self.metrics_log]
            axes[1, 0].plot(epochs, selection_rates)
            axes[1, 0].set_title('Selection Rate')
            
            importance_means = [m['importance_weights_mean'] for m in self.metrics_log]
            axes[1, 1].plot(epochs, importance_means)
            axes[1, 1].set_title('Importance Weights Mean')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")