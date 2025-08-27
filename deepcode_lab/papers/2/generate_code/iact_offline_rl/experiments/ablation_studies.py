#!/usr/bin/env python3
"""
Ablation Studies for IACT (Importance-Aware Co-Teaching) Algorithm

This module implements comprehensive ablation studies to analyze the contribution
of different components in the IACT algorithm:
1. Importance estimation (KLIEP vs uniform weighting)
2. Co-teaching mechanism (dual vs single policy)
3. Sample selection strategies (importance-based vs random)
4. Behavior cloning regularization (with vs without)
5. Selection rate schedules (dynamic vs fixed)

Author: IACT Research Team
"""

import os
import sys
import argparse
import logging
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.iact import IACTAlgorithm, create_iact_algorithm
from src.training.trainer import IACTTrainer
from src.data.d4rl_loader import load_d4rl_dataset
from src.utils.metrics import MetricsTracker
from configs.iact_config import IACTConfig, get_iact_config, create_ablation_config
from configs.env_configs import get_env_config, get_experiment_config


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    ablation_type: str
    env_name: str
    dataset_type: str
    seed: int
    final_score: float
    training_time: float
    convergence_epoch: int
    metrics_history: Dict[str, List[float]]
    config_used: Dict[str, Any]


class AblationStudyRunner:
    """
    Comprehensive ablation study runner for IACT algorithm.
    
    Systematically evaluates the contribution of different components:
    - Importance estimation methods
    - Co-teaching mechanisms
    - Sample selection strategies
    - Regularization techniques
    """
    
    def __init__(self, 
                 results_dir: str = "ablation_results",
                 device: str = "auto"):
        """
        Initialize ablation study runner.
        
        Args:
            results_dir: Directory to save ablation results
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Results storage
        self.ablation_results: List[AblationResult] = []
        
        # Setup logging
        self._setup_logging()
        
        # Define ablation configurations
        self.ablation_types = {
            'no_importance': 'Disable importance estimation (uniform weights)',
            'no_co_teaching': 'Disable co-teaching (single policy)',
            'random_selection': 'Random sample selection instead of importance-based',
            'no_bc_regularization': 'Disable behavior cloning regularization',
            'fixed_selection_rate': 'Fixed selection rate instead of dynamic',
            'no_dual_critics': 'Single critic instead of dual critics',
            'baseline_sac': 'Standard SAC without IACT components',
            'full_iact': 'Complete IACT algorithm (control condition)'
        }
        
        self.logger.info(f"Initialized AblationStudyRunner on device: {self.device}")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
    
    def _setup_logging(self):
        """Setup logging for ablation studies."""
        log_file = self.results_dir / "ablation_study.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_single_ablation(self,
                           ablation_type: str,
                           env_name: str,
                           dataset_type: str = "medium",
                           seed: int = 42,
                           max_epochs: int = 1000,
                           eval_frequency: int = 50) -> AblationResult:
        """
        Run a single ablation experiment.
        
        Args:
            ablation_type: Type of ablation to perform
            env_name: D4RL environment name
            dataset_type: Dataset quality ('random', 'medium', 'expert', 'medium-expert')
            seed: Random seed for reproducibility
            max_epochs: Maximum training epochs
            eval_frequency: Evaluation frequency in epochs
            
        Returns:
            AblationResult containing experiment results
        """
        self.logger.info(f"Starting ablation: {ablation_type} on {env_name}-{dataset_type} (seed={seed})")
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        start_time = time.time()
        
        try:
            # Load dataset
            full_env_name = f"{env_name}-{dataset_type}-v2"
            dataset, replay_buffer = load_d4rl_dataset(
                env_name=full_env_name,
                normalize_states=True,
                normalize_rewards=True,
                device=self.device
            )
            
            # Get base configuration
            base_config = get_iact_config(
                env_name=env_name,
                dataset_type=dataset_type,
                state_dim=dataset['observations'].shape[1],
                action_dim=dataset['actions'].shape[1],
                device=self.device
            )
            
            # Create ablation-specific configuration
            config = create_ablation_config(base_config, ablation_type)
            
            # Create algorithm
            algorithm = create_iact_algorithm(config)
            
            # Setup metrics tracking
            metrics_config = {
                'track_training_metrics': True,
                'track_evaluation_metrics': True,
                'save_frequency': eval_frequency,
                'log_frequency': 10
            }
            metrics_tracker = MetricsTracker(metrics_config)
            
            # Create trainer
            trainer = IACTTrainer(
                algorithm=algorithm,
                replay_buffer=replay_buffer,
                metrics_tracker=metrics_tracker,
                config=config
            )
            
            # Training loop with evaluation
            best_score = -np.inf
            convergence_epoch = max_epochs
            metrics_history = defaultdict(list)
            
            for epoch in range(max_epochs):
                # Training step
                train_metrics = trainer.train_epoch()
                
                # Log training metrics
                for key, value in train_metrics.items():
                    metrics_history[f"train_{key}"].append(value)
                
                # Periodic evaluation
                if epoch % eval_frequency == 0:
                    eval_metrics = trainer.evaluate()
                    current_score = eval_metrics.get('normalized_score', eval_metrics.get('episode_return', 0))
                    
                    # Log evaluation metrics
                    for key, value in eval_metrics.items():
                        metrics_history[f"eval_{key}"].append(value)
                    
                    # Check for improvement
                    if current_score > best_score:
                        best_score = current_score
                        convergence_epoch = epoch
                    
                    self.logger.info(
                        f"Epoch {epoch}: Score={current_score:.3f}, "
                        f"Best={best_score:.3f} (epoch {convergence_epoch})"
                    )
                
                # Early stopping check
                if epoch - convergence_epoch > 200:  # No improvement for 200 epochs
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            training_time = time.time() - start_time
            
            # Create result
            result = AblationResult(
                ablation_type=ablation_type,
                env_name=env_name,
                dataset_type=dataset_type,
                seed=seed,
                final_score=best_score,
                training_time=training_time,
                convergence_epoch=convergence_epoch,
                metrics_history=dict(metrics_history),
                config_used=asdict(config)
            )
            
            self.logger.info(
                f"Completed ablation {ablation_type}: "
                f"Score={best_score:.3f}, Time={training_time:.1f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ablation {ablation_type}: {str(e)}")
            # Return failed result
            return AblationResult(
                ablation_type=ablation_type,
                env_name=env_name,
                dataset_type=dataset_type,
                seed=seed,
                final_score=-np.inf,
                training_time=time.time() - start_time,
                convergence_epoch=-1,
                metrics_history={},
                config_used={}
            )
    
    def run_comprehensive_ablation(self,
                                 environments: List[str] = None,
                                 dataset_types: List[str] = None,
                                 seeds: List[int] = None,
                                 ablation_types: List[str] = None) -> Dict[str, List[AblationResult]]:
        """
        Run comprehensive ablation study across multiple configurations.
        
        Args:
            environments: List of environment names to test
            dataset_types: List of dataset types to test
            seeds: List of random seeds for multiple runs
            ablation_types: List of ablation types to perform
            
        Returns:
            Dictionary mapping ablation types to lists of results
        """
        # Default configurations
        if environments is None:
            environments = ['halfcheetah', 'hopper', 'walker2d']
        if dataset_types is None:
            dataset_types = ['medium', 'medium-expert']
        if seeds is None:
            seeds = [42, 123, 456]
        if ablation_types is None:
            ablation_types = list(self.ablation_types.keys())
        
        self.logger.info("Starting comprehensive ablation study")
        self.logger.info(f"Environments: {environments}")
        self.logger.info(f"Dataset types: {dataset_types}")
        self.logger.info(f"Seeds: {seeds}")
        self.logger.info(f"Ablation types: {ablation_types}")
        
        all_results = defaultdict(list)
        total_experiments = len(environments) * len(dataset_types) * len(seeds) * len(ablation_types)
        experiment_count = 0
        
        for env_name in environments:
            for dataset_type in dataset_types:
                for seed in seeds:
                    for ablation_type in ablation_types:
                        experiment_count += 1
                        self.logger.info(
                            f"Experiment {experiment_count}/{total_experiments}: "
                            f"{ablation_type} on {env_name}-{dataset_type} (seed={seed})"
                        )
                        
                        result = self.run_single_ablation(
                            ablation_type=ablation_type,
                            env_name=env_name,
                            dataset_type=dataset_type,
                            seed=seed
                        )
                        
                        all_results[ablation_type].append(result)
                        self.ablation_results.append(result)
                        
                        # Save intermediate results
                        self._save_results()
        
        self.logger.info("Comprehensive ablation study completed")
        return dict(all_results)
    
    def _save_results(self):
        """Save current ablation results to file."""
        results_file = self.results_dir / "ablation_results.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.ablation_results:
            result_dict = asdict(result)
            # Convert numpy arrays to lists for JSON serialization
            for key, value in result_dict['metrics_history'].items():
                if isinstance(value, np.ndarray):
                    result_dict['metrics_history'][key] = value.tolist()
            serializable_results.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved {len(self.ablation_results)} results to {results_file}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze ablation study results and generate summary statistics.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.ablation_results:
            self.logger.warning("No results to analyze")
            return {}
        
        analysis = {
            'summary_statistics': {},
            'component_importance': {},
            'environment_analysis': {},
            'dataset_analysis': {}
        }
        
        # Group results by ablation type
        results_by_type = defaultdict(list)
        for result in self.ablation_results:
            results_by_type[result.ablation_type].append(result)
        
        # Summary statistics for each ablation type
        for ablation_type, results in results_by_type.items():
            scores = [r.final_score for r in results if r.final_score != -np.inf]
            if scores:
                analysis['summary_statistics'][ablation_type] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'num_successful_runs': len(scores),
                    'num_total_runs': len(results)
                }
        
        # Component importance analysis (relative to full IACT)
        if 'full_iact' in results_by_type:
            full_iact_scores = [r.final_score for r in results_by_type['full_iact'] 
                              if r.final_score != -np.inf]
            full_iact_mean = np.mean(full_iact_scores) if full_iact_scores else 0
            
            for ablation_type, results in results_by_type.items():
                if ablation_type != 'full_iact':
                    scores = [r.final_score for r in results if r.final_score != -np.inf]
                    if scores and full_iact_mean > 0:
                        ablation_mean = np.mean(scores)
                        importance = (full_iact_mean - ablation_mean) / full_iact_mean
                        analysis['component_importance'][ablation_type] = {
                            'performance_drop': importance,
                            'absolute_drop': full_iact_mean - ablation_mean,
                            'description': self.ablation_types.get(ablation_type, 'Unknown')
                        }
        
        # Environment-specific analysis
        env_results = defaultdict(lambda: defaultdict(list))
        for result in self.ablation_results:
            env_results[result.env_name][result.ablation_type].append(result.final_score)
        
        for env_name, ablation_scores in env_results.items():
            analysis['environment_analysis'][env_name] = {}
            for ablation_type, scores in ablation_scores.items():
                valid_scores = [s for s in scores if s != -np.inf]
                if valid_scores:
                    analysis['environment_analysis'][env_name][ablation_type] = np.mean(valid_scores)
        
        # Dataset quality analysis
        dataset_results = defaultdict(lambda: defaultdict(list))
        for result in self.ablation_results:
            dataset_results[result.dataset_type][result.ablation_type].append(result.final_score)
        
        for dataset_type, ablation_scores in dataset_results.items():
            analysis['dataset_analysis'][dataset_type] = {}
            for ablation_type, scores in ablation_scores.items():
                valid_scores = [s for s in scores if s != -np.inf]
                if valid_scores:
                    analysis['dataset_analysis'][dataset_type][ablation_type] = np.mean(valid_scores)
        
        # Save analysis
        analysis_file = self.results_dir / "ablation_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Analysis saved to {analysis_file}")
        return analysis
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive ablation study report.
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_results()
        
        report = []
        report.append("=" * 80)
        report.append("IACT ABLATION STUDY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total experiments: {len(self.ablation_results)}")
        report.append(f"Ablation types tested: {len(set(r.ablation_type for r in self.ablation_results))}")
        report.append(f"Environments tested: {len(set(r.env_name for r in self.ablation_results))}")
        report.append("")
        
        # Component importance ranking
        if 'component_importance' in analysis:
            report.append("COMPONENT IMPORTANCE RANKING")
            report.append("-" * 40)
            importance_items = list(analysis['component_importance'].items())
            importance_items.sort(key=lambda x: x[1]['performance_drop'], reverse=True)
            
            for i, (component, data) in enumerate(importance_items, 1):
                report.append(f"{i}. {component}")
                report.append(f"   Performance drop: {data['performance_drop']:.3f} ({data['performance_drop']*100:.1f}%)")
                report.append(f"   Description: {data['description']}")
                report.append("")
        
        # Environment-specific results
        if 'environment_analysis' in analysis:
            report.append("ENVIRONMENT-SPECIFIC RESULTS")
            report.append("-" * 40)
            for env_name, results in analysis['environment_analysis'].items():
                report.append(f"{env_name.upper()}:")
                for ablation_type, score in results.items():
                    report.append(f"  {ablation_type}: {score:.3f}")
                report.append("")
        
        # Dataset quality analysis
        if 'dataset_analysis' in analysis:
            report.append("DATASET QUALITY ANALYSIS")
            report.append("-" * 40)
            for dataset_type, results in analysis['dataset_analysis'].items():
                report.append(f"{dataset_type.upper()}:")
                for ablation_type, score in results.items():
                    report.append(f"  {ablation_type}: {score:.3f}")
                report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.results_dir / "ablation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Report saved to {report_file}")
        return report_text


def main():
    """Main function for running ablation studies."""
    parser = argparse.ArgumentParser(description="IACT Ablation Studies")
    parser.add_argument("--env", type=str, default="halfcheetah",
                       help="Environment name")
    parser.add_argument("--dataset", type=str, default="medium",
                       help="Dataset type")
    parser.add_argument("--ablation", type=str, default="all",
                       help="Ablation type or 'all' for comprehensive study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds")
    parser.add_argument("--results-dir", type=str, default="ablation_results",
                       help="Results directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Computing device")
    parser.add_argument("--max-epochs", type=int, default=1000,
                       help="Maximum training epochs")
    
    args = parser.parse_args()
    
    # Create ablation runner
    runner = AblationStudyRunner(
        results_dir=args.results_dir,
        device=args.device
    )
    
    if args.ablation == "all":
        # Run comprehensive ablation study
        environments = [args.env] if args.env != "all" else ['halfcheetah', 'hopper', 'walker2d']
        dataset_types = [args.dataset] if args.dataset != "all" else ['medium', 'medium-expert']
        
        runner.run_comprehensive_ablation(
            environments=environments,
            dataset_types=dataset_types,
            seeds=args.seeds
        )
    else:
        # Run single ablation
        result = runner.run_single_ablation(
            ablation_type=args.ablation,
            env_name=args.env,
            dataset_type=args.dataset,
            seed=args.seeds[0],
            max_epochs=args.max_epochs
        )
        runner.ablation_results.append(result)
        runner._save_results()
    
    # Generate and print report
    report = runner.generate_report()
    print(report)


if __name__ == "__main__":
    main()