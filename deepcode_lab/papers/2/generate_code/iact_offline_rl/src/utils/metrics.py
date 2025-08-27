"""
Evaluation metrics and logging utilities for IACT offline reinforcement learning.

This module provides comprehensive metrics tracking, evaluation functions, and
logging utilities for the IACT algorithm implementation.
"""

import torch
import numpy as np
import logging
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single training step."""
    # Policy losses
    policy_a_loss: float
    policy_b_loss: float
    critic_a_loss: float
    critic_b_loss: float
    
    # Co-teaching metrics
    selection_rate_a: float
    selection_rate_b: float
    selected_samples_a: int
    selected_samples_b: int
    
    # Importance weights
    mean_importance_weight: float
    std_importance_weight: float
    max_importance_weight: float
    min_importance_weight: float
    
    # Behavior cloning regularization
    bc_loss_a: float
    bc_loss_b: float
    
    # Q-values
    mean_q_value_a: float
    mean_q_value_b: float
    
    # Training step info
    step: int
    epoch: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for policy performance."""
    # Episode statistics
    mean_return: float
    std_return: float
    max_return: float
    min_return: float
    
    # Episode lengths
    mean_episode_length: float
    std_episode_length: float
    
    # Success metrics (if applicable)
    success_rate: float = 0.0
    
    # Normalized scores (D4RL style)
    normalized_score: float = 0.0
    
    # Additional metrics
    num_episodes: int = 0
    total_steps: int = 0
    evaluation_time: float = 0.0
    
    # Per-episode data
    episode_returns: List[float] = None
    episode_lengths: List[int] = None
    
    def __post_init__(self):
        if self.episode_returns is None:
            self.episode_returns = []
        if self.episode_lengths is None:
            self.episode_lengths = []


class MetricsTracker:
    """Comprehensive metrics tracking and logging system."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 save_frequency: int = 100,
                 plot_frequency: int = 1000,
                 max_history: int = 10000):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs and plots
            save_frequency: How often to save metrics to disk
            plot_frequency: How often to generate plots
            max_history: Maximum number of metrics to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.plot_frequency = plot_frequency
        self.max_history = max_history
        
        # Training metrics history
        self.training_metrics: List[TrainingMetrics] = []
        self.evaluation_metrics: List[EvaluationMetrics] = []
        
        # Running averages
        self.running_averages = defaultdict(lambda: deque(maxlen=100))
        
        # Best performance tracking
        self.best_eval_return = float('-inf')
        self.best_eval_metrics = None
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"MetricsTracker initialized with log_dir: {self.log_dir}")
    
    def setup_logging(self):
        """Setup file logging."""
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def log_training_step(self, metrics: TrainingMetrics):
        """Log training step metrics."""
        self.training_metrics.append(metrics)
        
        # Update running averages
        metrics_dict = asdict(metrics)
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                self.running_averages[key].append(value)
        
        # Log to console periodically
        if metrics.step % 100 == 0:
            self._log_training_summary(metrics)
        
        # Save to disk periodically
        if metrics.step % self.save_frequency == 0:
            self.save_metrics()
        
        # Generate plots periodically
        if metrics.step % self.plot_frequency == 0:
            self.plot_training_curves()
        
        # Trim history if needed
        if len(self.training_metrics) > self.max_history:
            self.training_metrics = self.training_metrics[-self.max_history:]
    
    def log_evaluation(self, metrics: EvaluationMetrics, step: int):
        """Log evaluation metrics."""
        metrics.total_steps = step
        self.evaluation_metrics.append(metrics)
        
        # Check if this is the best performance
        if metrics.mean_return > self.best_eval_return:
            self.best_eval_return = metrics.mean_return
            self.best_eval_metrics = metrics
            logger.info(f"New best evaluation return: {metrics.mean_return:.2f}")
        
        # Log evaluation summary
        self._log_evaluation_summary(metrics, step)
        
        # Save evaluation results
        self.save_evaluation_results()
    
    def _log_training_summary(self, metrics: TrainingMetrics):
        """Log training summary to console."""
        avg_policy_loss = (metrics.policy_a_loss + metrics.policy_b_loss) / 2
        avg_critic_loss = (metrics.critic_a_loss + metrics.critic_b_loss) / 2
        avg_selection_rate = (metrics.selection_rate_a + metrics.selection_rate_b) / 2
        
        logger.info(
            f"Step {metrics.step:6d} | "
            f"Policy Loss: {avg_policy_loss:.4f} | "
            f"Critic Loss: {avg_critic_loss:.4f} | "
            f"Selection Rate: {avg_selection_rate:.3f} | "
            f"Importance Weight: {metrics.mean_importance_weight:.3f}±{metrics.std_importance_weight:.3f}"
        )
    
    def _log_evaluation_summary(self, metrics: EvaluationMetrics, step: int):
        """Log evaluation summary to console."""
        logger.info(
            f"Evaluation at step {step:6d} | "
            f"Return: {metrics.mean_return:.2f}±{metrics.std_return:.2f} | "
            f"Episodes: {metrics.num_episodes} | "
            f"Success Rate: {metrics.success_rate:.3f} | "
            f"Normalized Score: {metrics.normalized_score:.3f}"
        )
    
    def get_running_average(self, metric_name: str, window: int = 100) -> float:
        """Get running average of a metric."""
        if metric_name not in self.running_averages:
            return 0.0
        
        values = list(self.running_averages[metric_name])
        if not values:
            return 0.0
        
        return np.mean(values[-window:])
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the latest training metrics."""
        return self.training_metrics[-1] if self.training_metrics else None
    
    def get_best_evaluation(self) -> Optional[EvaluationMetrics]:
        """Get the best evaluation metrics."""
        return self.best_eval_metrics
    
    def save_metrics(self):
        """Save metrics to disk."""
        try:
            # Save training metrics
            training_file = self.log_dir / "training_metrics.json"
            training_data = [asdict(m) for m in self.training_metrics[-1000:]]  # Save last 1000
            with open(training_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # Save evaluation metrics
            eval_file = self.log_dir / "evaluation_metrics.json"
            eval_data = [asdict(m) for m in self.evaluation_metrics]
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
            
            # Save running averages
            avg_file = self.log_dir / "running_averages.json"
            avg_data = {k: list(v) for k, v in self.running_averages.items()}
            with open(avg_file, 'w') as f:
                json.dump(avg_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def save_evaluation_results(self):
        """Save detailed evaluation results."""
        try:
            if not self.evaluation_metrics:
                return
            
            eval_file = self.log_dir / "detailed_evaluation.json"
            eval_data = []
            
            for metrics in self.evaluation_metrics:
                data = asdict(metrics)
                eval_data.append(data)
            
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def plot_training_curves(self):
        """Generate training curve plots."""
        try:
            if len(self.training_metrics) < 10:
                return
            
            # Extract data
            steps = [m.step for m in self.training_metrics]
            policy_losses_a = [m.policy_a_loss for m in self.training_metrics]
            policy_losses_b = [m.policy_b_loss for m in self.training_metrics]
            critic_losses_a = [m.critic_a_loss for m in self.training_metrics]
            critic_losses_b = [m.critic_b_loss for m in self.training_metrics]
            selection_rates_a = [m.selection_rate_a for m in self.training_metrics]
            selection_rates_b = [m.selection_rate_b for m in self.training_metrics]
            importance_weights = [m.mean_importance_weight for m in self.training_metrics]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('IACT Training Curves', fontsize=16)
            
            # Policy losses
            axes[0, 0].plot(steps, policy_losses_a, label='Policy A', alpha=0.7)
            axes[0, 0].plot(steps, policy_losses_b, label='Policy B', alpha=0.7)
            axes[0, 0].set_title('Policy Losses')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Critic losses
            axes[0, 1].plot(steps, critic_losses_a, label='Critic A', alpha=0.7)
            axes[0, 1].plot(steps, critic_losses_b, label='Critic B', alpha=0.7)
            axes[0, 1].set_title('Critic Losses')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Selection rates
            axes[1, 0].plot(steps, selection_rates_a, label='Selection Rate A', alpha=0.7)
            axes[1, 0].plot(steps, selection_rates_b, label='Selection Rate B', alpha=0.7)
            axes[1, 0].set_title('Co-teaching Selection Rates')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Selection Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Importance weights
            axes[1, 1].plot(steps, importance_weights, alpha=0.7)
            axes[1, 1].set_title('Mean Importance Weights')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Importance Weight')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / "training_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot evaluation curves if available
            if self.evaluation_metrics:
                self.plot_evaluation_curves()
                
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")
    
    def plot_evaluation_curves(self):
        """Generate evaluation curve plots."""
        try:
            if len(self.evaluation_metrics) < 2:
                return
            
            # Extract data
            steps = [m.total_steps for m in self.evaluation_metrics]
            returns = [m.mean_return for m in self.evaluation_metrics]
            std_returns = [m.std_return for m in self.evaluation_metrics]
            normalized_scores = [m.normalized_score for m in self.evaluation_metrics]
            success_rates = [m.success_rate for m in self.evaluation_metrics]
            
            # Create subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('IACT Evaluation Results', fontsize=16)
            
            # Returns with error bars
            axes[0].errorbar(steps, returns, yerr=std_returns, alpha=0.7, capsize=3)
            axes[0].set_title('Episode Returns')
            axes[0].set_xlabel('Training Steps')
            axes[0].set_ylabel('Return')
            axes[0].grid(True, alpha=0.3)
            
            # Normalized scores
            axes[1].plot(steps, normalized_scores, marker='o', alpha=0.7)
            axes[1].set_title('Normalized Scores (D4RL)')
            axes[1].set_xlabel('Training Steps')
            axes[1].set_ylabel('Normalized Score')
            axes[1].grid(True, alpha=0.3)
            
            # Success rates
            axes[2].plot(steps, success_rates, marker='s', alpha=0.7)
            axes[2].set_title('Success Rates')
            axes[2].set_xlabel('Training Steps')
            axes[2].set_ylabel('Success Rate')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / "evaluation_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting evaluation curves: {e}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        report = {
            "training_summary": {},
            "evaluation_summary": {},
            "best_performance": {},
            "training_progress": {}
        }
        
        # Training summary
        if self.training_metrics:
            latest = self.training_metrics[-1]
            report["training_summary"] = {
                "total_steps": latest.step,
                "total_epochs": latest.epoch,
                "avg_policy_loss": self.get_running_average("policy_a_loss"),
                "avg_critic_loss": self.get_running_average("critic_a_loss"),
                "avg_selection_rate": self.get_running_average("selection_rate_a"),
                "avg_importance_weight": self.get_running_average("mean_importance_weight")
            }
        
        # Evaluation summary
        if self.evaluation_metrics:
            latest_eval = self.evaluation_metrics[-1]
            report["evaluation_summary"] = {
                "num_evaluations": len(self.evaluation_metrics),
                "latest_return": latest_eval.mean_return,
                "latest_normalized_score": latest_eval.normalized_score,
                "latest_success_rate": latest_eval.success_rate
            }
        
        # Best performance
        if self.best_eval_metrics:
            report["best_performance"] = {
                "best_return": self.best_eval_metrics.mean_return,
                "best_normalized_score": self.best_eval_metrics.normalized_score,
                "best_success_rate": self.best_eval_metrics.success_rate,
                "achieved_at_step": self.best_eval_metrics.total_steps
            }
        
        return report


class D4RLEvaluator:
    """Specialized evaluator for D4RL environments."""
    
    def __init__(self, env_name: str):
        """
        Initialize D4RL evaluator.
        
        Args:
            env_name: Name of the D4RL environment
        """
        self.env_name = env_name
        self.ref_min_score, self.ref_max_score = self._get_d4rl_score_range()
    
    def _get_d4rl_score_range(self) -> Tuple[float, float]:
        """Get the reference score range for D4RL normalization."""
        # D4RL reference scores for normalization
        d4rl_scores = {
            # Gym-MuJoCo
            'halfcheetah-random-v2': (-280.178953, 12135.0),
            'halfcheetah-medium-v2': (-280.178953, 12135.0),
            'halfcheetah-expert-v2': (-280.178953, 12135.0),
            'halfcheetah-medium-expert-v2': (-280.178953, 12135.0),
            'halfcheetah-medium-replay-v2': (-280.178953, 12135.0),
            
            'hopper-random-v2': (-20.272305, 3234.3),
            'hopper-medium-v2': (-20.272305, 3234.3),
            'hopper-expert-v2': (-20.272305, 3234.3),
            'hopper-medium-expert-v2': (-20.272305, 3234.3),
            'hopper-medium-replay-v2': (-20.272305, 3234.3),
            
            'walker2d-random-v2': (1.629008, 4592.3),
            'walker2d-medium-v2': (1.629008, 4592.3),
            'walker2d-expert-v2': (1.629008, 4592.3),
            'walker2d-medium-expert-v2': (1.629008, 4592.3),
            'walker2d-medium-replay-v2': (1.629008, 4592.3),
        }
        
        return d4rl_scores.get(self.env_name, (0.0, 100.0))
    
    def normalize_score(self, score: float) -> float:
        """Normalize score according to D4RL convention."""
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score) * 100.0
    
    def evaluate_policy(self, 
                       policy_fn,
                       env,
                       num_episodes: int = 10,
                       max_episode_steps: int = 1000,
                       render: bool = False) -> EvaluationMetrics:
        """
        Evaluate policy in D4RL environment.
        
        Args:
            policy_fn: Function that takes state and returns action
            env: Environment instance
            num_episodes: Number of episodes to evaluate
            max_episode_steps: Maximum steps per episode
            render: Whether to render episodes
            
        Returns:
            EvaluationMetrics with comprehensive evaluation results
        """
        episode_returns = []
        episode_lengths = []
        success_count = 0
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new gym API
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            for step in range(max_episode_steps):
                if render:
                    env.render()
                
                # Get action from policy
                with torch.no_grad():
                    if isinstance(state, np.ndarray):
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    else:
                        state_tensor = state.unsqueeze(0)
                    
                    action = policy_fn(state_tensor, deterministic=True)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy().flatten()
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]  # Handle new gym API
                
                episode_return += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            # Check for success (environment-specific)
            if 'success' in info:
                success_count += info['success']
            elif episode_return > (self.ref_max_score * 0.8):  # Heuristic success
                success_count += 1
        
        evaluation_time = time.time() - start_time
        
        # Calculate statistics
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        max_return = np.max(episode_returns)
        min_return = np.min(episode_returns)
        
        mean_episode_length = np.mean(episode_lengths)
        std_episode_length = np.std(episode_lengths)
        
        success_rate = success_count / num_episodes
        normalized_score = self.normalize_score(mean_return)
        
        return EvaluationMetrics(
            mean_return=mean_return,
            std_return=std_return,
            max_return=max_return,
            min_return=min_return,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            success_rate=success_rate,
            normalized_score=normalized_score,
            num_episodes=num_episodes,
            total_steps=sum(episode_lengths),
            evaluation_time=evaluation_time,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths
        )


def create_metrics_tracker(config: Dict[str, Any]) -> MetricsTracker:
    """
    Factory function to create a metrics tracker.
    
    Args:
        config: Configuration dictionary with tracker settings
        
    Returns:
        Configured MetricsTracker instance
    """
    return MetricsTracker(
        log_dir=config.get('log_dir', 'logs'),
        save_frequency=config.get('save_frequency', 100),
        plot_frequency=config.get('plot_frequency', 1000),
        max_history=config.get('max_history', 10000)
    )


def create_d4rl_evaluator(env_name: str) -> D4RLEvaluator:
    """
    Factory function to create a D4RL evaluator.
    
    Args:
        env_name: Name of the D4RL environment
        
    Returns:
        Configured D4RLEvaluator instance
    """
    return D4RLEvaluator(env_name)


def compute_policy_divergence(policy_a_probs: torch.Tensor, 
                             policy_b_probs: torch.Tensor) -> float:
    """
    Compute KL divergence between two policy distributions.
    
    Args:
        policy_a_probs: Action probabilities from policy A
        policy_b_probs: Action probabilities from policy B
        
    Returns:
        KL divergence D_KL(A || B)
    """
    # Add small epsilon to prevent log(0)
    eps = 1e-8
    policy_a_probs = policy_a_probs + eps
    policy_b_probs = policy_b_probs + eps
    
    # Compute KL divergence
    kl_div = torch.sum(policy_a_probs * torch.log(policy_a_probs / policy_b_probs))
    return kl_div.item()


def compute_importance_weight_stats(weights: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for importance weights.
    
    Args:
        weights: Tensor of importance weights
        
    Returns:
        Dictionary with weight statistics
    """
    weights_np = weights.detach().cpu().numpy()
    
    return {
        'mean': float(np.mean(weights_np)),
        'std': float(np.std(weights_np)),
        'min': float(np.min(weights_np)),
        'max': float(np.max(weights_np)),
        'median': float(np.median(weights_np)),
        'q25': float(np.percentile(weights_np, 25)),
        'q75': float(np.percentile(weights_np, 75)),
        'effective_sample_size': float(np.sum(weights_np) ** 2 / np.sum(weights_np ** 2))
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing IACT Metrics Module")
    
    # Test metrics tracker
    tracker = MetricsTracker(log_dir="test_logs")
    
    # Create dummy training metrics
    for step in range(100):
        metrics = TrainingMetrics(
            policy_a_loss=np.random.uniform(0.1, 1.0),
            policy_b_loss=np.random.uniform(0.1, 1.0),
            critic_a_loss=np.random.uniform(0.5, 2.0),
            critic_b_loss=np.random.uniform(0.5, 2.0),
            selection_rate_a=np.random.uniform(0.5, 0.9),
            selection_rate_b=np.random.uniform(0.5, 0.9),
            selected_samples_a=int(np.random.uniform(100, 500)),
            selected_samples_b=int(np.random.uniform(100, 500)),
            mean_importance_weight=np.random.uniform(0.8, 1.2),
            std_importance_weight=np.random.uniform(0.1, 0.3),
            max_importance_weight=np.random.uniform(2.0, 5.0),
            min_importance_weight=np.random.uniform(0.1, 0.5),
            bc_loss_a=np.random.uniform(0.01, 0.1),
            bc_loss_b=np.random.uniform(0.01, 0.1),
            mean_q_value_a=np.random.uniform(-10, 10),
            mean_q_value_b=np.random.uniform(-10, 10),
            step=step,
            epoch=step // 10
        )
        tracker.log_training_step(metrics)
    
    # Create dummy evaluation metrics
    eval_metrics = EvaluationMetrics(
        mean_return=100.5,
        std_return=15.2,
        max_return=130.0,
        min_return=75.0,
        mean_episode_length=500.0,
        std_episode_length=50.0,
        success_rate=0.8,
        normalized_score=85.3,
        num_episodes=10,
        total_steps=5000,
        evaluation_time=120.5
    )
    tracker.log_evaluation(eval_metrics, step=100)
    
    # Generate summary report
    report = tracker.generate_summary_report()
    print("Summary Report:")
    print(json.dumps(report, indent=2))
    
    # Test D4RL evaluator
    evaluator = D4RLEvaluator('halfcheetah-medium-v2')
    normalized_score = evaluator.normalize_score(5000.0)
    print(f"Normalized score for return 5000: {normalized_score:.2f}")
    
    # Test importance weight statistics
    dummy_weights = torch.randn(1000).abs() + 0.5
    weight_stats = compute_importance_weight_stats(dummy_weights)
    print("Importance weight statistics:")
    print(json.dumps(weight_stats, indent=2))
    
    print("Metrics module testing completed successfully!")