#!/usr/bin/env python3
"""
Main D4RL Benchmark Evaluation Script for IACT Algorithm

This script runs comprehensive benchmarks on D4RL datasets using the IACT algorithm,
including training, evaluation, and comparison with baseline methods.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import gym
import d4rl

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.iact import create_iact_algorithm, IACTAlgorithm
from training.trainer import create_trainer, IACTTrainer
from data.d4rl_loader import load_d4rl_dataset, create_d4rl_loader
from data.replay_buffer import ReplayBuffer
from configs.iact_config import get_iact_config, IACTConfig
from configs.env_configs import get_env_config, get_env_group, get_experiment_config
from utils.metrics import create_metrics_tracker, create_d4rl_evaluator, MetricsTracker, D4RLEvaluator


class D4RLBenchmarkRunner:
    """Main benchmark runner for IACT on D4RL datasets."""
    
    def __init__(self, 
                 results_dir: str = "results",
                 log_level: str = "INFO",
                 device: Optional[str] = None):
        """
        Initialize benchmark runner.
        
        Args:
            results_dir: Directory to save results
            log_level: Logging level
            device: Device to use (cuda/cpu)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized D4RL Benchmark Runner on device: {self.device}")
        
        # Results storage
        self.benchmark_results = {}
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        log_file = self.results_dir / "benchmark.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_single_experiment(self, 
                            env_name: str,
                            config_overrides: Optional[Dict] = None,
                            num_seeds: int = 3) -> Dict:
        """
        Run IACT experiment on a single D4RL environment.
        
        Args:
            env_name: D4RL environment name
            config_overrides: Configuration overrides
            num_seeds: Number of random seeds to run
            
        Returns:
            Dictionary with experiment results
        """
        self.logger.info(f"Starting experiment on {env_name}")
        
        # Get environment configuration
        env_config = get_env_config(env_name)
        experiment_config = get_experiment_config(env_name, **(config_overrides or {}))
        
        # Create IACT configuration
        iact_config = get_iact_config(
            env_name=env_name,
            dataset_type=self._get_dataset_type(env_name),
            state_dim=env_config.state_dim,
            action_dim=env_config.action_dim,
            device=str(self.device),
            **experiment_config
        )
        
        # Results for this environment
        env_results = {
            'env_name': env_name,
            'config': iact_config.__dict__,
            'seeds': {}
        }
        
        # Run multiple seeds
        for seed in range(num_seeds):
            self.logger.info(f"Running seed {seed + 1}/{num_seeds} for {env_name}")
            
            try:
                seed_results = self._run_single_seed(env_name, iact_config, seed)
                env_results['seeds'][seed] = seed_results
                
            except Exception as e:
                self.logger.error(f"Error in seed {seed} for {env_name}: {e}")
                env_results['seeds'][seed] = {'error': str(e)}
        
        # Aggregate results across seeds
        env_results['aggregated'] = self._aggregate_seed_results(env_results['seeds'])
        
        self.logger.info(f"Completed experiment on {env_name}")
        return env_results
    
    def _run_single_seed(self, env_name: str, config: IACTConfig, seed: int) -> Dict:
        """Run single seed experiment."""
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create experiment directory
        exp_dir = self.results_dir / env_name / f"seed_{seed}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.logger.info(f"Loading dataset for {env_name}")
        dataset, replay_buffer = load_d4rl_dataset(
            env_name=env_name,
            normalize_states=config.normalize_states,
            normalize_rewards=config.normalize_rewards,
            device=str(self.device)
        )
        
        # Create environment for evaluation
        env = gym.make(env_name)
        
        # Create IACT algorithm
        algorithm = create_iact_algorithm(config)
        
        # Create metrics tracker
        metrics_config = {
            'log_dir': str(exp_dir),
            'save_frequency': config.eval_frequency,
            'plot_frequency': config.eval_frequency * 5,
            'max_history': config.max_epochs
        }
        metrics_tracker = create_metrics_tracker(metrics_config)
        
        # Create D4RL evaluator
        evaluator = create_d4rl_evaluator(env_name)
        
        # Create trainer
        trainer = create_trainer(
            algorithm=algorithm,
            replay_buffer=replay_buffer,
            metrics_tracker=metrics_tracker,
            config=config
        )
        
        # Training loop
        self.logger.info(f"Starting training for {env_name}, seed {seed}")
        start_time = time.time()
        
        best_score = -np.inf
        best_epoch = 0
        
        for epoch in range(config.max_epochs):
            # Training step
            train_metrics = trainer.train_epoch(epoch)
            
            # Evaluation
            if epoch % config.eval_frequency == 0:
                eval_metrics = evaluator.evaluate_policy(
                    policy_fn=lambda state: algorithm.select_action(
                        torch.FloatTensor(state).to(self.device), 
                        deterministic=True
                    ).cpu().numpy(),
                    env=env,
                    num_episodes=config.eval_episodes,
                    max_episode_steps=config.max_episode_steps
                )
                
                # Log evaluation
                metrics_tracker.log_evaluation(eval_metrics, epoch)
                
                # Check for best performance
                if eval_metrics.normalized_score > best_score:
                    best_score = eval_metrics.normalized_score
                    best_epoch = epoch
                    
                    # Save best model
                    best_model_path = exp_dir / "best_model.pt"
                    algorithm.save_checkpoint(str(best_model_path))
                
                self.logger.info(
                    f"Epoch {epoch}: Score = {eval_metrics.normalized_score:.3f}, "
                    f"Return = {eval_metrics.mean_return:.3f}"
                )
            
            # Early stopping check
            if epoch - best_epoch > config.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        # Final evaluation with best model
        if best_epoch < epoch:
            algorithm.load_checkpoint(str(exp_dir / "best_model.pt"))
        
        final_eval = evaluator.evaluate_policy(
            policy_fn=lambda state: algorithm.select_action(
                torch.FloatTensor(state).to(self.device), 
                deterministic=True
            ).cpu().numpy(),
            env=env,
            num_episodes=config.final_eval_episodes,
            max_episode_steps=config.max_episode_steps
        )
        
        # Compile results
        seed_results = {
            'seed': seed,
            'training_time': training_time,
            'best_epoch': best_epoch,
            'best_score': best_score,
            'final_evaluation': {
                'normalized_score': final_eval.normalized_score,
                'mean_return': final_eval.mean_return,
                'std_return': final_eval.std_return,
                'success_rate': final_eval.success_rate,
                'mean_episode_length': final_eval.mean_episode_length
            },
            'training_history': metrics_tracker.get_history()
        }
        
        # Save seed results
        with open(exp_dir / "results.json", 'w') as f:
            json.dump(seed_results, f, indent=2, default=str)
        
        env.close()
        return seed_results
    
    def _aggregate_seed_results(self, seed_results: Dict) -> Dict:
        """Aggregate results across multiple seeds."""
        valid_seeds = [r for r in seed_results.values() if 'error' not in r]
        
        if not valid_seeds:
            return {'error': 'No valid seeds'}
        
        # Extract metrics
        scores = [r['final_evaluation']['normalized_score'] for r in valid_seeds]
        returns = [r['final_evaluation']['mean_return'] for r in valid_seeds]
        success_rates = [r['final_evaluation']['success_rate'] for r in valid_seeds]
        training_times = [r['training_time'] for r in valid_seeds]
        
        aggregated = {
            'num_seeds': len(valid_seeds),
            'normalized_score': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'mean_return': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns)
            },
            'success_rate': {
                'mean': np.mean(success_rates),
                'std': np.std(success_rates),
                'min': np.min(success_rates),
                'max': np.max(success_rates)
            },
            'training_time': {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'total': np.sum(training_times)
            }
        }
        
        return aggregated
    
    def run_benchmark_suite(self, 
                          env_groups: Optional[List[str]] = None,
                          specific_envs: Optional[List[str]] = None,
                          num_seeds: int = 3,
                          config_overrides: Optional[Dict] = None) -> Dict:
        """
        Run complete benchmark suite.
        
        Args:
            env_groups: Environment groups to run (e.g., ['locomotion', 'antmaze'])
            specific_envs: Specific environments to run
            num_seeds: Number of seeds per environment
            config_overrides: Global configuration overrides
            
        Returns:
            Complete benchmark results
        """
        self.logger.info("Starting D4RL benchmark suite")
        
        # Determine environments to run
        environments = []
        
        if specific_envs:
            environments.extend(specific_envs)
        
        if env_groups:
            for group in env_groups:
                try:
                    group_envs = get_env_group(group)
                    environments.extend(group_envs)
                except KeyError:
                    self.logger.warning(f"Unknown environment group: {group}")
        
        # Default to locomotion tasks if nothing specified
        if not environments:
            environments = get_env_group('locomotion')
        
        # Remove duplicates
        environments = list(set(environments))
        
        self.logger.info(f"Running benchmark on {len(environments)} environments: {environments}")
        
        # Run experiments
        suite_results = {
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
            'environments': environments,
            'num_seeds': num_seeds,
            'config_overrides': config_overrides or {},
            'results': {}
        }
        
        for env_name in environments:
            try:
                env_results = self.run_single_experiment(
                    env_name=env_name,
                    config_overrides=config_overrides,
                    num_seeds=num_seeds
                )
                suite_results['results'][env_name] = env_results
                
            except Exception as e:
                self.logger.error(f"Failed to run experiment on {env_name}: {e}")
                suite_results['results'][env_name] = {'error': str(e)}
        
        # Save complete results
        results_file = self.results_dir / f"benchmark_results_{suite_results['timestamp']}.json"
        with open(results_file, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        
        # Generate summary
        self._generate_benchmark_summary(suite_results)
        
        self.logger.info(f"Benchmark suite completed. Results saved to {results_file}")
        return suite_results
    
    def _generate_benchmark_summary(self, results: Dict):
        """Generate benchmark summary report."""
        summary_file = self.results_dir / "benchmark_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("IACT D4RL Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Environments: {len(results['environments'])}\n")
            f.write(f"Seeds per environment: {results['num_seeds']}\n\n")
            
            # Results table
            f.write("Results Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Environment':<25} {'Score':<15} {'Return':<15} {'Success Rate':<15}\n")
            f.write("-" * 80 + "\n")
            
            for env_name, env_results in results['results'].items():
                if 'error' in env_results:
                    f.write(f"{env_name:<25} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}\n")
                else:
                    agg = env_results['aggregated']
                    score = f"{agg['normalized_score']['mean']:.3f}±{agg['normalized_score']['std']:.3f}"
                    ret = f"{agg['mean_return']['mean']:.1f}±{agg['mean_return']['std']:.1f}"
                    success = f"{agg['success_rate']['mean']:.3f}±{agg['success_rate']['std']:.3f}"
                    
                    f.write(f"{env_name:<25} {score:<15} {ret:<15} {success:<15}\n")
            
            f.write("-" * 80 + "\n")
        
        self.logger.info(f"Summary report saved to {summary_file}")
    
    def _get_dataset_type(self, env_name: str) -> str:
        """Determine dataset type from environment name."""
        if 'random' in env_name:
            return 'random'
        elif 'medium' in env_name and 'expert' in env_name:
            return 'medium-expert'
        elif 'medium' in env_name and 'replay' in env_name:
            return 'medium-replay'
        elif 'medium' in env_name:
            return 'medium'
        elif 'expert' in env_name:
            return 'expert'
        else:
            return 'medium'  # default


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Run IACT D4RL Benchmark")
    
    # Environment selection
    parser.add_argument('--env-groups', nargs='+', 
                       choices=['locomotion', 'antmaze', 'kitchen', 'adroit'],
                       help='Environment groups to run')
    parser.add_argument('--envs', nargs='+',
                       help='Specific environments to run')
    
    # Experiment settings
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of seeds per environment')
    parser.add_argument('--max-epochs', type=int, default=1000,
                       help='Maximum training epochs')
    parser.add_argument('--eval-frequency', type=int, default=50,
                       help='Evaluation frequency')
    
    # System settings
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Configuration overrides
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate')
    parser.add_argument('--importance-weight', type=float,
                       help='Importance weighting factor')
    
    args = parser.parse_args()
    
    # Create configuration overrides
    config_overrides = {}
    if args.max_epochs:
        config_overrides['max_epochs'] = args.max_epochs
    if args.eval_frequency:
        config_overrides['eval_frequency'] = args.eval_frequency
    if args.batch_size:
        config_overrides['batch_size'] = args.batch_size
    if args.learning_rate:
        config_overrides['actor_lr'] = args.learning_rate
        config_overrides['critic_lr'] = args.learning_rate
    if args.importance_weight:
        config_overrides['importance_weight'] = args.importance_weight
    
    # Create benchmark runner
    runner = D4RLBenchmarkRunner(
        results_dir=args.results_dir,
        log_level=args.log_level,
        device=args.device
    )
    
    # Run benchmark
    try:
        results = runner.run_benchmark_suite(
            env_groups=args.env_groups,
            specific_envs=args.envs,
            num_seeds=args.seeds,
            config_overrides=config_overrides
        )
        
        print("\nBenchmark completed successfully!")
        print(f"Results saved to: {runner.results_dir}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    main()