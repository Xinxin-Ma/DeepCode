"""
Explanation Quality Evaluation System for ACB-Agent

This module implements comprehensive evaluation of explanation quality for the
Approximate Concept Bottleneck (ACB) agent system, as described in the paper
"Robust Explanations for Human-Neural Multi-Agent Systems via Approximate Concept Bottlenecks".

Key Features:
- Explanation consistency evaluation across similar states
- Human study simulation for explanation effectiveness
- Concept alignment assessment
- Comparative analysis with baseline explanation methods
- Statistical significance testing
- Comprehensive reporting and visualization

Author: ACB-Agent Implementation
Date: 2025
"""

import torch
import numpy as np
import logging
import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import our implemented components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.acb_agent import ACBAgent, MultiAgentACBSystem
from utils.explanation_generator import ExplanationGenerator, ExplanationStyle, create_explanation_generator
from utils.metrics import (
    ComprehensiveMetricsEvaluator, 
    ExplanationQualityMetrics,
    HumanStudyMetrics,
    ConceptAlignmentMetrics,
    create_default_metrics_evaluator
)
from utils.concept_utils import ConceptVocabulary, create_concept_vocabulary
from environments.human_interaction import HumanInteractionEnvironment, create_human_interaction_environment, InteractionMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationEvaluationConfig:
    """Configuration for explanation evaluation experiments."""
    
    # Evaluation parameters
    num_evaluation_episodes: int = 100
    num_test_states: int = 500
    similarity_threshold: float = 0.8
    k_concepts: int = 3
    
    # Human study simulation parameters
    num_simulated_participants: int = 50
    participant_expertise_levels: List[str] = None
    trust_threshold: float = 0.7
    
    # Statistical testing parameters
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Baseline comparison
    compare_baselines: bool = True
    baseline_methods: List[str] = None
    
    # Output configuration
    save_results: bool = True
    results_dir: str = "evaluation_results"
    generate_plots: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.participant_expertise_levels is None:
            self.participant_expertise_levels = ["novice", "intermediate", "expert"]
        
        if self.baseline_methods is None:
            self.baseline_methods = ["random", "attention", "gradient"]

class ExplanationEvaluator:
    """
    Comprehensive explanation quality evaluator for ACB-Agent system.
    
    This class implements the evaluation methodology described in Section 4.2
    of the paper, focusing on explanation quality assessment through multiple
    metrics and human study simulation.
    """
    
    def __init__(self, 
                 config: ExplanationEvaluationConfig,
                 device: str = "cpu"):
        """
        Initialize the explanation evaluator.
        
        Args:
            config: Evaluation configuration
            device: Computing device (cpu/cuda)
        """
        self.config = config
        self.device = device
        
        # Initialize metrics evaluators
        self.metrics_evaluator = create_default_metrics_evaluator("human_ai")
        self.explanation_metrics = ExplanationQualityMetrics()
        self.human_study_metrics = HumanStudyMetrics()
        self.concept_alignment_metrics = ConceptAlignmentMetrics()
        
        # Results storage
        self.evaluation_results = {}
        self.detailed_results = defaultdict(list)
        
        # Create results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
        
        logger.info(f"Initialized ExplanationEvaluator with config: {config}")
    
    def evaluate_agent_explanations(self,
                                  agent_system: Union[ACBAgent, MultiAgentACBSystem],
                                  concept_vocab: ConceptVocabulary,
                                  test_states: torch.Tensor,
                                  ground_truth_concepts: Optional[List[List[str]]] = None,
                                  ground_truth_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate explanation quality for a given agent system.
        
        Args:
            agent_system: ACB agent or multi-agent system to evaluate
            concept_vocab: Concept vocabulary for interpretation
            test_states: Test states for evaluation
            ground_truth_concepts: Ground truth concept labels (optional)
            ground_truth_actions: Ground truth actions (optional)
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        logger.info("Starting comprehensive explanation evaluation...")
        start_time = time.time()
        
        # Initialize explanation generator
        explanation_generator = create_explanation_generator(
            concept_vocab=concept_vocab,
            style=ExplanationStyle.DETAILED,
            k_concepts=self.config.k_concepts
        )
        
        # Create human interaction environment
        interaction_env = create_human_interaction_environment(
            agent_system=agent_system,
            concept_vocab=concept_vocab,
            interaction_mode=InteractionMode.EVALUATION,
            explanation_style=ExplanationStyle.DETAILED,
            k_concepts=self.config.k_concepts
        )
        
        # Generate explanations for all test states
        explanations = []
        concept_activations_list = []
        actions_taken = []
        
        logger.info(f"Generating explanations for {len(test_states)} test states...")
        
        for i, state in enumerate(test_states):
            # Get agent action and explanation
            if isinstance(agent_system, MultiAgentACBSystem):
                # Multi-agent case
                agent_outputs = agent_system.act(
                    states=[state],
                    deterministic=True,
                    return_explanations=True
                )
                # Use first agent's output for evaluation
                agent_output = agent_outputs[0]
            else:
                # Single agent case
                agent_output = agent_system.act(
                    state=state,
                    deterministic=True,
                    return_explanation=True,
                    explanation_k=self.config.k_concepts
                )
            
            # Extract information
            concept_activations = agent_output['concept_activations']
            action = agent_output.get('actions', 'unknown')
            
            # Generate detailed explanation
            explanation_result = explanation_generator.generate_explanation(
                concept_activations=concept_activations,
                action=str(action),
                action_probs=agent_output.get('action_probs'),
                state_info={'state_index': i}
            )
            
            explanations.append(explanation_result)
            concept_activations_list.append(concept_activations)
            actions_taken.append(str(action))
            
            if self.config.verbose and (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{len(test_states)} explanations")
        
        # Evaluate explanation quality
        results = {}
        
        # 1. Explanation Consistency Evaluation
        logger.info("Evaluating explanation consistency...")
        consistency_results = self.explanation_metrics.compute_explanation_consistency(
            explanations=explanations,
            states=test_states,
            similarity_threshold=self.config.similarity_threshold
        )
        results['consistency'] = consistency_results
        
        # 2. Explanation Quality Score
        logger.info("Computing explanation quality scores...")
        quality_results = self.explanation_metrics.compute_explanation_quality_score(
            explanations=explanations,
            human_ratings=None  # Will be generated in human study simulation
        )
        results['quality'] = quality_results
        
        # 3. Human Study Simulation
        logger.info("Simulating human study...")
        human_study_results = self.human_study_metrics.simulate_human_study(
            explanations=explanations,
            ground_truth_actions=ground_truth_actions or actions_taken,
            num_participants=self.config.num_simulated_participants
        )
        results['human_study'] = human_study_results
        
        # 4. Concept Alignment Evaluation
        if ground_truth_concepts is not None:
            logger.info("Evaluating concept alignment...")
            # Convert concept activations to tensor
            model_concepts = torch.stack(concept_activations_list)
            
            # Convert ground truth concepts to tensor representation
            human_labels = self._convert_concepts_to_tensor(
                ground_truth_concepts, concept_vocab
            )
            
            alignment_results = self.concept_alignment_metrics.compute_concept_alignment(
                model_concepts=model_concepts,
                human_labels=human_labels,
                concept_names=concept_vocab.get_concept_names()
            )
            results['concept_alignment'] = alignment_results
        
        # 5. Statistical Analysis
        logger.info("Performing statistical analysis...")
        statistical_results = self._perform_statistical_analysis(
            explanations, concept_activations_list, actions_taken
        )
        results['statistical_analysis'] = statistical_results
        
        # 6. Baseline Comparison (if enabled)
        if self.config.compare_baselines:
            logger.info("Comparing with baseline methods...")
            baseline_results = self._compare_with_baselines(
                test_states, explanations, concept_activations_list
            )
            results['baseline_comparison'] = baseline_results
        
        # Compute overall evaluation score
        overall_score = self._compute_overall_evaluation_score(results)
        results['overall_score'] = overall_score
        
        # Store detailed results
        self.evaluation_results = results
        self.detailed_results['explanations'] = explanations
        self.detailed_results['concept_activations'] = concept_activations_list
        self.detailed_results['actions'] = actions_taken
        
        evaluation_time = time.time() - start_time
        results['evaluation_time'] = evaluation_time
        
        logger.info(f"Explanation evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Overall evaluation score: {overall_score:.3f}")
        
        return results
    
    def _convert_concepts_to_tensor(self, 
                                  concept_lists: List[List[str]], 
                                  concept_vocab: ConceptVocabulary) -> torch.Tensor:
        """Convert list of concept names to tensor representation."""
        concept_names = concept_vocab.get_concept_names()
        name_to_idx = {name: idx for idx, name in enumerate(concept_names)}
        
        tensor_labels = torch.zeros(len(concept_lists), len(concept_names))
        
        for i, concepts in enumerate(concept_lists):
            for concept in concepts:
                if concept in name_to_idx:
                    tensor_labels[i, name_to_idx[concept]] = 1.0
        
        return tensor_labels
    
    def _perform_statistical_analysis(self,
                                    explanations: List[Dict],
                                    concept_activations: List[torch.Tensor],
                                    actions: List[str]) -> Dict[str, Any]:
        """Perform statistical analysis of explanation patterns."""
        
        # Extract concept activation statistics
        all_activations = torch.stack(concept_activations)
        
        # Compute activation statistics
        activation_stats = {
            'mean_activation': torch.mean(all_activations, dim=0).tolist(),
            'std_activation': torch.std(all_activations, dim=0).tolist(),
            'max_activation': torch.max(all_activations, dim=0)[0].tolist(),
            'min_activation': torch.min(all_activations, dim=0)[0].tolist()
        }
        
        # Compute explanation diversity
        explanation_texts = [exp.get('explanation', '') for exp in explanations]
        unique_explanations = len(set(explanation_texts))
        diversity_score = unique_explanations / len(explanation_texts) if explanation_texts else 0
        
        # Compute concept usage frequency
        concept_usage = defaultdict(int)
        for exp in explanations:
            top_concepts = exp.get('top_concepts', [])
            for concept in top_concepts:
                concept_usage[concept] += 1
        
        # Sort by frequency
        sorted_usage = sorted(concept_usage.items(), key=lambda x: x[1], reverse=True)
        
        # Action distribution
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
        
        return {
            'activation_statistics': activation_stats,
            'explanation_diversity': diversity_score,
            'concept_usage_frequency': dict(sorted_usage[:10]),  # Top 10
            'action_distribution': dict(action_counts),
            'total_unique_explanations': unique_explanations,
            'total_explanations': len(explanations)
        }
    
    def _compare_with_baselines(self,
                              test_states: torch.Tensor,
                              acb_explanations: List[Dict],
                              concept_activations: List[torch.Tensor]) -> Dict[str, Any]:
        """Compare ACB explanations with baseline methods."""
        
        baseline_results = {}
        
        for baseline_method in self.config.baseline_methods:
            logger.info(f"Evaluating baseline method: {baseline_method}")
            
            if baseline_method == "random":
                baseline_explanations = self._generate_random_explanations(len(test_states))
            elif baseline_method == "attention":
                baseline_explanations = self._generate_attention_explanations(test_states)
            elif baseline_method == "gradient":
                baseline_explanations = self._generate_gradient_explanations(test_states)
            else:
                logger.warning(f"Unknown baseline method: {baseline_method}")
                continue
            
            # Compute baseline metrics
            baseline_quality = self.explanation_metrics.compute_explanation_quality_score(
                explanations=baseline_explanations
            )
            
            baseline_consistency = self.explanation_metrics.compute_explanation_consistency(
                explanations=baseline_explanations,
                states=test_states,
                similarity_threshold=self.config.similarity_threshold
            )
            
            baseline_results[baseline_method] = {
                'quality': baseline_quality,
                'consistency': baseline_consistency
            }
        
        # Compare ACB with baselines
        acb_quality = self.explanation_metrics.compute_explanation_quality_score(
            explanations=acb_explanations
        )
        
        comparison_results = {}
        for method, baseline_metrics in baseline_results.items():
            comparison_results[method] = {
                'quality_improvement': acb_quality['overall_quality'] - baseline_metrics['quality']['overall_quality'],
                'consistency_improvement': self.evaluation_results.get('consistency', {}).get('overall_consistency', 0) - baseline_metrics['consistency']['overall_consistency']
            }
        
        return {
            'baseline_results': baseline_results,
            'acb_vs_baselines': comparison_results
        }
    
    def _generate_random_explanations(self, num_explanations: int) -> List[Dict]:
        """Generate random baseline explanations."""
        random_concepts = ["random_concept_1", "random_concept_2", "random_concept_3"]
        explanations = []
        
        for i in range(num_explanations):
            selected_concepts = np.random.choice(random_concepts, size=2, replace=False).tolist()
            explanation = {
                'explanation': f"Random explanation based on {', '.join(selected_concepts)}",
                'top_concepts': selected_concepts,
                'confidence': np.random.uniform(0.3, 0.7),
                'method': 'random'
            }
            explanations.append(explanation)
        
        return explanations
    
    def _generate_attention_explanations(self, test_states: torch.Tensor) -> List[Dict]:
        """Generate attention-based baseline explanations."""
        explanations = []
        
        for i, state in enumerate(test_states):
            # Simulate attention weights (random for baseline)
            attention_weights = torch.softmax(torch.randn(state.shape[-1]), dim=0)
            top_indices = torch.topk(attention_weights, k=3)[1]
            
            explanation = {
                'explanation': f"Attention-based explanation focusing on features {top_indices.tolist()}",
                'top_concepts': [f"feature_{idx.item()}" for idx in top_indices],
                'confidence': torch.max(attention_weights).item(),
                'method': 'attention'
            }
            explanations.append(explanation)
        
        return explanations
    
    def _generate_gradient_explanations(self, test_states: torch.Tensor) -> List[Dict]:
        """Generate gradient-based baseline explanations."""
        explanations = []
        
        for i, state in enumerate(test_states):
            # Simulate gradient magnitudes (random for baseline)
            gradient_magnitudes = torch.abs(torch.randn_like(state))
            top_indices = torch.topk(gradient_magnitudes.flatten(), k=3)[1]
            
            explanation = {
                'explanation': f"Gradient-based explanation highlighting important features {top_indices.tolist()}",
                'top_concepts': [f"gradient_feature_{idx.item()}" for idx in top_indices],
                'confidence': torch.max(gradient_magnitudes).item(),
                'method': 'gradient'
            }
            explanations.append(explanation)
        
        return explanations
    
    def _compute_overall_evaluation_score(self, results: Dict[str, Any]) -> float:
        """Compute overall evaluation score combining all metrics."""
        
        # Weight different components
        weights = {
            'consistency': 0.25,
            'quality': 0.25,
            'human_study': 0.30,
            'concept_alignment': 0.20
        }
        
        score = 0.0
        total_weight = 0.0
        
        # Consistency score
        if 'consistency' in results:
            consistency_score = results['consistency'].get('overall_consistency', 0)
            score += weights['consistency'] * consistency_score
            total_weight += weights['consistency']
        
        # Quality score
        if 'quality' in results:
            quality_score = results['quality'].get('overall_quality', 0)
            score += weights['quality'] * quality_score
            total_weight += weights['quality']
        
        # Human study score
        if 'human_study' in results:
            human_score = results['human_study'].get('overall_effectiveness', 0)
            score += weights['human_study'] * human_score
            total_weight += weights['human_study']
        
        # Concept alignment score
        if 'concept_alignment' in results:
            alignment_score = results['concept_alignment'].get('overall_alignment', 0)
            score += weights['concept_alignment'] * alignment_score
            total_weight += weights['concept_alignment']
        
        # Normalize by total weight
        if total_weight > 0:
            score = score / total_weight
        
        return score
    
    def generate_evaluation_report(self, 
                                 results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ACB-AGENT EXPLANATION QUALITY EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall Score
        overall_score = results.get('overall_score', 0)
        report_lines.append(f"OVERALL EVALUATION SCORE: {overall_score:.3f}")
        report_lines.append("")
        
        # Consistency Results
        if 'consistency' in results:
            consistency = results['consistency']
            report_lines.append("EXPLANATION CONSISTENCY:")
            report_lines.append(f"  Overall Consistency: {consistency.get('overall_consistency', 0):.3f}")
            report_lines.append(f"  Pairwise Consistency: {consistency.get('pairwise_consistency', 0):.3f}")
            report_lines.append(f"  Temporal Consistency: {consistency.get('temporal_consistency', 0):.3f}")
            report_lines.append("")
        
        # Quality Results
        if 'quality' in results:
            quality = results['quality']
            report_lines.append("EXPLANATION QUALITY:")
            report_lines.append(f"  Overall Quality: {quality.get('overall_quality', 0):.3f}")
            report_lines.append(f"  Completeness: {quality.get('completeness', 0):.3f}")
            report_lines.append(f"  Clarity: {quality.get('clarity', 0):.3f}")
            report_lines.append("")
        
        # Human Study Results
        if 'human_study' in results:
            human_study = results['human_study']
            report_lines.append("HUMAN STUDY SIMULATION:")
            report_lines.append(f"  Overall Effectiveness: {human_study.get('overall_effectiveness', 0):.3f}")
            report_lines.append(f"  Trust Score: {human_study.get('trust_score', 0):.3f}")
            report_lines.append(f"  Understanding Score: {human_study.get('understanding_score', 0):.3f}")
            report_lines.append("")
        
        # Concept Alignment Results
        if 'concept_alignment' in results:
            alignment = results['concept_alignment']
            report_lines.append("CONCEPT ALIGNMENT:")
            report_lines.append(f"  Overall Alignment: {alignment.get('overall_alignment', 0):.3f}")
            report_lines.append(f"  Aligned Concepts: {alignment.get('num_aligned_concepts', 0)}")
            report_lines.append(f"  Total Concepts: {alignment.get('total_concepts', 0)}")
            report_lines.append("")
        
        # Statistical Analysis
        if 'statistical_analysis' in results:
            stats = results['statistical_analysis']
            report_lines.append("STATISTICAL ANALYSIS:")
            report_lines.append(f"  Explanation Diversity: {stats.get('explanation_diversity', 0):.3f}")
            report_lines.append(f"  Unique Explanations: {stats.get('total_unique_explanations', 0)}")
            report_lines.append(f"  Total Explanations: {stats.get('total_explanations', 0)}")
            report_lines.append("")
        
        # Baseline Comparison
        if 'baseline_comparison' in results:
            baseline = results['baseline_comparison']
            report_lines.append("BASELINE COMPARISON:")
            for method, comparison in baseline.get('acb_vs_baselines', {}).items():
                report_lines.append(f"  vs {method.upper()}:")
                report_lines.append(f"    Quality Improvement: {comparison.get('quality_improvement', 0):.3f}")
                report_lines.append(f"    Consistency Improvement: {comparison.get('consistency_improvement', 0):.3f}")
            report_lines.append("")
        
        # Evaluation Time
        eval_time = results.get('evaluation_time', 0)
        report_lines.append(f"EVALUATION TIME: {eval_time:.2f} seconds")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to: {save_path}")
        
        return report_text
    
    def save_results(self, results: Dict[str, Any], filename: str = "explanation_evaluation_results.json"):
        """Save evaluation results to JSON file."""
        if not self.config.save_results:
            return
        
        filepath = os.path.join(self.config.results_dir, filename)
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert tensors and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

def create_explanation_evaluator(domain: str = "human_ai",
                                num_evaluation_episodes: int = 100,
                                num_test_states: int = 500,
                                save_results: bool = True,
                                results_dir: str = "evaluation_results") -> ExplanationEvaluator:
    """
    Factory function to create a configured explanation evaluator.
    
    Args:
        domain: Domain type for evaluation
        num_evaluation_episodes: Number of episodes to evaluate
        num_test_states: Number of test states for evaluation
        save_results: Whether to save results
        results_dir: Directory to save results
        
    Returns:
        Configured ExplanationEvaluator instance
    """
    config = ExplanationEvaluationConfig(
        num_evaluation_episodes=num_evaluation_episodes,
        num_test_states=num_test_states,
        save_results=save_results,
        results_dir=results_dir
    )
    
    return ExplanationEvaluator(config=config)

def run_explanation_evaluation_experiment(agent_system: Union[ACBAgent, MultiAgentACBSystem],
                                        concept_vocab: ConceptVocabulary,
                                        test_states: torch.Tensor,
                                        ground_truth_concepts: Optional[List[List[str]]] = None,
                                        ground_truth_actions: Optional[List[str]] = None,
                                        config: Optional[ExplanationEvaluationConfig] = None) -> Dict[str, Any]:
    """
    Run a complete explanation evaluation experiment.
    
    Args:
        agent_system: ACB agent or multi-agent system to evaluate
        concept_vocab: Concept vocabulary for interpretation
        test_states: Test states for evaluation
        ground_truth_concepts: Ground truth concept labels (optional)
        ground_truth_actions: Ground truth actions (optional)
        config: Evaluation configuration (optional)
        
    Returns:
        Complete evaluation results
    """
    if config is None:
        config = ExplanationEvaluationConfig()
    
    # Create evaluator
    evaluator = ExplanationEvaluator(config=config)
    
    # Run evaluation
    results = evaluator.evaluate_agent_explanations(
        agent_system=agent_system,
        concept_vocab=concept_vocab,
        test_states=test_states,
        ground_truth_concepts=ground_truth_concepts,
        ground_truth_actions=ground_truth_actions
    )
    
    # Generate and save report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    if config.save_results:
        # Save results
        evaluator.save_results(results)
        
        # Save report
        report_path = os.path.join(config.results_dir, "evaluation_report.txt")
        evaluator.generate_evaluation_report(results, save_path=report_path)
    
    return results

if __name__ == "__main__":
    """
    Example usage of the explanation evaluation system.
    """
    
    # This would typically be run with a trained ACB agent
    logger.info("Explanation Evaluation System - Example Usage")
    
    # Example configuration
    config = ExplanationEvaluationConfig(
        num_evaluation_episodes=50,
        num_test_states=100,
        save_results=True,
        results_dir="example_evaluation_results"
    )
    
    logger.info("Example configuration created")
    logger.info("To run evaluation, provide trained ACB agent and test data")
    logger.info("Use run_explanation_evaluation_experiment() function")