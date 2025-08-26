"""
Evaluation Metrics for ACB-Agent System

This module implements comprehensive evaluation metrics for the ACB-Agent system,
including task performance metrics, explanation quality metrics, concept alignment
metrics, and human study evaluation metrics as specified in the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics computation"""
    alignment_threshold: float = 0.6  # τ_align from Section 3.3
    explanation_quality_threshold: float = 0.7
    performance_window: int = 100  # Episodes for moving average
    concept_diversity_weight: float = 0.1
    human_study_confidence: float = 0.95
    batch_size: int = 32


class TaskPerformanceMetrics:
    """
    Task Performance Metrics for Multi-Agent Coordination
    Measures how well ACB-Agent maintains task performance while providing explanations
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.episode_rewards = deque(maxlen=config.performance_window)
        self.episode_lengths = deque(maxlen=config.performance_window)
        self.success_rates = deque(maxlen=config.performance_window)
        self.convergence_data = []
        
    def update_episode_metrics(self, 
                             episode_reward: float,
                             episode_length: int,
                             success: bool) -> None:
        """Update metrics with new episode data"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.success_rates.append(1.0 if success else 0.0)
        
    def compute_performance_metrics(self) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
        if not self.episode_rewards:
            return {}
            
        metrics = {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'success_rate': np.mean(self.success_rates),
            'reward_stability': 1.0 / (1.0 + np.std(self.episode_rewards)),
            'sample_efficiency': np.mean(self.episode_rewards) / np.mean(self.episode_lengths)
        }
        
        # Compute convergence metrics
        if len(self.episode_rewards) >= 10:
            recent_rewards = list(self.episode_rewards)[-10:]
            early_rewards = list(self.episode_rewards)[:10]
            metrics['convergence_improvement'] = np.mean(recent_rewards) - np.mean(early_rewards)
            
        return metrics
    
    def compute_baseline_comparison(self, 
                                  baseline_rewards: List[float],
                                  baseline_success_rates: List[float]) -> Dict[str, float]:
        """Compare performance against non-explainable baselines"""
        if not self.episode_rewards or not baseline_rewards:
            return {}
            
        current_reward = np.mean(self.episode_rewards)
        baseline_reward = np.mean(baseline_rewards)
        current_success = np.mean(self.success_rates)
        baseline_success = np.mean(baseline_success_rates)
        
        return {
            'reward_ratio': current_reward / baseline_reward if baseline_reward > 0 else 0.0,
            'success_ratio': current_success / baseline_success if baseline_success > 0 else 0.0,
            'performance_gap': abs(current_reward - baseline_reward) / baseline_reward if baseline_reward > 0 else float('inf'),
            'within_5_percent': abs(current_reward - baseline_reward) / baseline_reward < 0.05 if baseline_reward > 0 else False
        }


class ConceptAlignmentMetrics:
    """
    Concept Alignment Metrics from Section 3.3
    Measures alignment between learned concepts and human-interpretable labels
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.alignment_history = []
        
    def compute_concept_alignment(self,
                                model_concepts: torch.Tensor,
                                human_labels: torch.Tensor,
                                concept_names: List[str]) -> Dict[str, Any]:
        """
        Compute concept-human alignment scores
        
        Args:
            model_concepts: [batch_size, concept_dim] - Model concept activations
            human_labels: [batch_size, concept_dim] - Human concept labels
            concept_names: List of concept names
            
        Returns:
            Dictionary with alignment metrics
        """
        if model_concepts.shape != human_labels.shape:
            raise ValueError("Model concepts and human labels must have same shape")
            
        batch_size, concept_dim = model_concepts.shape
        
        # Compute per-concept correlations
        concept_correlations = []
        aligned_concepts = []
        
        for i in range(concept_dim):
            model_vals = model_concepts[:, i].cpu().numpy()
            human_vals = human_labels[:, i].cpu().numpy()
            
            # Compute Pearson correlation
            correlation = np.corrcoef(model_vals, human_vals)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
                
            concept_correlations.append(correlation)
            
            # Check if aligned (ρ_i > τ_align)
            if correlation > self.config.alignment_threshold:
                aligned_concepts.append(i)
        
        # Overall alignment metrics
        alignment_score = len(aligned_concepts) / concept_dim
        mean_correlation = np.mean(concept_correlations)
        
        # Per-concept alignment details
        concept_alignment_details = {
            concept_names[i]: {
                'correlation': concept_correlations[i],
                'aligned': concept_correlations[i] > self.config.alignment_threshold,
                'activation_mean': float(model_concepts[:, i].mean()),
                'human_label_mean': float(human_labels[:, i].mean())
            }
            for i in range(min(len(concept_names), concept_dim))
        }
        
        alignment_metrics = {
            'alignment_score': alignment_score,
            'mean_correlation': mean_correlation,
            'num_aligned_concepts': len(aligned_concepts),
            'total_concepts': concept_dim,
            'concept_correlations': concept_correlations,
            'aligned_concept_indices': aligned_concepts,
            'concept_details': concept_alignment_details
        }
        
        self.alignment_history.append(alignment_metrics)
        return alignment_metrics
    
    def compute_alignment_stability(self) -> Dict[str, float]:
        """Compute stability of concept alignment over time"""
        if len(self.alignment_history) < 2:
            return {'alignment_stability': 1.0}
            
        recent_scores = [h['alignment_score'] for h in self.alignment_history[-10:]]
        stability = 1.0 - np.std(recent_scores)
        
        return {
            'alignment_stability': max(0.0, stability),
            'alignment_trend': np.mean(recent_scores[-5:]) - np.mean(recent_scores[:5]) if len(recent_scores) >= 5 else 0.0
        }


class ExplanationQualityMetrics:
    """
    Explanation Quality Assessment Metrics
    Measures quality and interpretability of generated explanations
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.explanation_history = []
        
    def compute_explanation_consistency(self,
                                      explanations: List[Dict],
                                      states: torch.Tensor,
                                      similarity_threshold: float = 0.8) -> Dict[str, float]:
        """
        Compute explanation consistency across similar states
        
        Args:
            explanations: List of explanation dictionaries
            states: Corresponding state tensors
            similarity_threshold: Threshold for state similarity
            
        Returns:
            Consistency metrics
        """
        if len(explanations) < 2:
            return {'consistency_score': 1.0}
            
        consistency_scores = []
        
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                # Compute state similarity
                state_sim = F.cosine_similarity(
                    states[i].unsqueeze(0), 
                    states[j].unsqueeze(0)
                ).item()
                
                if state_sim > similarity_threshold:
                    # Compare explanations for similar states
                    exp1_concepts = set(explanations[i].get('top_concepts', []))
                    exp2_concepts = set(explanations[j].get('top_concepts', []))
                    
                    if exp1_concepts and exp2_concepts:
                        concept_overlap = len(exp1_concepts.intersection(exp2_concepts))
                        concept_union = len(exp1_concepts.union(exp2_concepts))
                        consistency = concept_overlap / concept_union if concept_union > 0 else 0.0
                        consistency_scores.append(consistency)
        
        return {
            'consistency_score': np.mean(consistency_scores) if consistency_scores else 1.0,
            'num_comparisons': len(consistency_scores)
        }
    
    def compute_explanation_diversity(self, explanations: List[Dict]) -> Dict[str, float]:
        """Compute diversity of explanations across different states"""
        if not explanations:
            return {'diversity_score': 0.0}
            
        all_concepts = set()
        explanation_concepts = []
        
        for exp in explanations:
            concepts = set(exp.get('top_concepts', []))
            explanation_concepts.append(concepts)
            all_concepts.update(concepts)
        
        if not all_concepts:
            return {'diversity_score': 0.0}
            
        # Compute diversity as average pairwise Jaccard distance
        diversity_scores = []
        for i in range(len(explanation_concepts)):
            for j in range(i + 1, len(explanation_concepts)):
                intersection = len(explanation_concepts[i].intersection(explanation_concepts[j]))
                union = len(explanation_concepts[i].union(explanation_concepts[j]))
                jaccard_sim = intersection / union if union > 0 else 0.0
                diversity_scores.append(1.0 - jaccard_sim)  # Jaccard distance
        
        return {
            'diversity_score': np.mean(diversity_scores) if diversity_scores else 0.0,
            'unique_concepts_used': len(all_concepts),
            'avg_concepts_per_explanation': np.mean([len(concepts) for concepts in explanation_concepts])
        }
    
    def compute_explanation_quality_score(self,
                                        explanations: List[Dict],
                                        human_ratings: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Compute overall explanation quality score
        
        Args:
            explanations: List of explanation dictionaries
            human_ratings: Optional human quality ratings (0-1 scale)
            
        Returns:
            Quality metrics
        """
        if not explanations:
            return {'quality_score': 0.0}
            
        # Intrinsic quality metrics
        completeness_scores = []
        clarity_scores = []
        
        for exp in explanations:
            # Completeness: presence of required components
            has_concepts = 'top_concepts' in exp and len(exp['top_concepts']) > 0
            has_explanation = 'explanation' in exp and len(exp['explanation']) > 0
            completeness = (has_concepts + has_explanation) / 2.0
            completeness_scores.append(completeness)
            
            # Clarity: explanation length and concept coverage
            if 'explanation' in exp:
                exp_length = len(exp['explanation'].split())
                clarity = min(1.0, exp_length / 20.0)  # Optimal around 20 words
            else:
                clarity = 0.0
            clarity_scores.append(clarity)
        
        quality_metrics = {
            'completeness_score': np.mean(completeness_scores),
            'clarity_score': np.mean(clarity_scores),
            'intrinsic_quality': (np.mean(completeness_scores) + np.mean(clarity_scores)) / 2.0
        }
        
        # Add human ratings if available
        if human_ratings:
            quality_metrics['human_rating'] = np.mean(human_ratings)
            quality_metrics['human_rating_std'] = np.std(human_ratings)
            quality_metrics['overall_quality'] = (quality_metrics['intrinsic_quality'] + np.mean(human_ratings)) / 2.0
        else:
            quality_metrics['overall_quality'] = quality_metrics['intrinsic_quality']
            
        return quality_metrics


class HumanStudyMetrics:
    """
    Human Study Evaluation Metrics
    Simulates and evaluates human study experiments for explanation effectiveness
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.study_results = []
        
    def simulate_human_study(self,
                           explanations: List[Dict],
                           ground_truth_actions: List[str],
                           num_participants: int = 50) -> Dict[str, Any]:
        """
        Simulate human study for explanation effectiveness
        
        Args:
            explanations: Generated explanations
            ground_truth_actions: Correct actions
            num_participants: Number of simulated participants
            
        Returns:
            Human study metrics
        """
        if not explanations or not ground_truth_actions:
            return {}
            
        # Simulate participant responses
        participant_scores = []
        understanding_scores = []
        trust_scores = []
        
        for participant in range(num_participants):
            # Simulate individual participant performance
            correct_predictions = 0
            understanding_ratings = []
            trust_ratings = []
            
            for i, (exp, gt_action) in enumerate(zip(explanations, ground_truth_actions)):
                # Simulate understanding based on explanation quality
                exp_quality = len(exp.get('top_concepts', [])) / 5.0  # Normalize by max concepts
                understanding = min(1.0, exp_quality + np.random.normal(0, 0.1))
                understanding_ratings.append(max(0.0, understanding))
                
                # Simulate trust based on explanation consistency
                trust = 0.7 + 0.3 * understanding + np.random.normal(0, 0.05)
                trust_ratings.append(max(0.0, min(1.0, trust)))
                
                # Simulate action prediction accuracy
                prediction_prob = 0.5 + 0.4 * understanding  # Base + explanation effect
                if np.random.random() < prediction_prob:
                    correct_predictions += 1
            
            participant_scores.append(correct_predictions / len(explanations))
            understanding_scores.append(np.mean(understanding_ratings))
            trust_scores.append(np.mean(trust_ratings))
        
        # Compute study statistics
        study_metrics = {
            'prediction_accuracy': np.mean(participant_scores),
            'prediction_accuracy_std': np.std(participant_scores),
            'understanding_score': np.mean(understanding_scores),
            'understanding_std': np.std(understanding_scores),
            'trust_score': np.mean(trust_scores),
            'trust_std': np.std(trust_scores),
            'num_participants': num_participants,
            'confidence_interval': self._compute_confidence_interval(participant_scores)
        }
        
        self.study_results.append(study_metrics)
        return study_metrics
    
    def _compute_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Compute confidence interval for scores"""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)
        
        # 95% confidence interval
        margin = 1.96 * std_score / np.sqrt(n)
        return (mean_score - margin, mean_score + margin)
    
    def compute_study_effectiveness(self) -> Dict[str, float]:
        """Compute overall study effectiveness metrics"""
        if not self.study_results:
            return {}
            
        all_accuracies = [result['prediction_accuracy'] for result in self.study_results]
        all_understanding = [result['understanding_score'] for result in self.study_results]
        all_trust = [result['trust_score'] for result in self.study_results]
        
        return {
            'overall_accuracy': np.mean(all_accuracies),
            'overall_understanding': np.mean(all_understanding),
            'overall_trust': np.mean(all_trust),
            'study_consistency': 1.0 - np.std(all_accuracies),  # Lower std = higher consistency
            'effectiveness_score': (np.mean(all_accuracies) + np.mean(all_understanding) + np.mean(all_trust)) / 3.0
        }


class ComprehensiveMetricsEvaluator:
    """
    Comprehensive Metrics Evaluator
    Integrates all metrics for complete system evaluation
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.task_metrics = TaskPerformanceMetrics(self.config)
        self.alignment_metrics = ConceptAlignmentMetrics(self.config)
        self.explanation_metrics = ExplanationQualityMetrics(self.config)
        self.human_study_metrics = HumanStudyMetrics(self.config)
        
        self.evaluation_history = []
        
    def evaluate_system(self,
                       episode_data: Dict[str, Any],
                       concept_data: Dict[str, torch.Tensor],
                       explanation_data: List[Dict],
                       baseline_comparison: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Comprehensive system evaluation
        
        Args:
            episode_data: Episode performance data
            concept_data: Concept activation and alignment data
            explanation_data: Generated explanations
            baseline_comparison: Baseline performance data
            
        Returns:
            Complete evaluation metrics
        """
        evaluation_results = {
            'timestamp': time.time(),
            'episode_id': episode_data.get('episode_id', 0)
        }
        
        # Task Performance Metrics
        if 'reward' in episode_data:
            self.task_metrics.update_episode_metrics(
                episode_data['reward'],
                episode_data.get('length', 0),
                episode_data.get('success', False)
            )
            evaluation_results['task_performance'] = self.task_metrics.compute_performance_metrics()
            
            if baseline_comparison:
                evaluation_results['baseline_comparison'] = self.task_metrics.compute_baseline_comparison(
                    baseline_comparison.get('rewards', []),
                    baseline_comparison.get('success_rates', [])
                )
        
        # Concept Alignment Metrics
        if 'model_concepts' in concept_data and 'human_labels' in concept_data:
            alignment_results = self.alignment_metrics.compute_concept_alignment(
                concept_data['model_concepts'],
                concept_data['human_labels'],
                concept_data.get('concept_names', [])
            )
            evaluation_results['concept_alignment'] = alignment_results
            evaluation_results['alignment_stability'] = self.alignment_metrics.compute_alignment_stability()
        
        # Explanation Quality Metrics
        if explanation_data:
            states = concept_data.get('states', torch.randn(len(explanation_data), 10))  # Dummy states if not provided
            
            consistency_metrics = self.explanation_metrics.compute_explanation_consistency(
                explanation_data, states
            )
            diversity_metrics = self.explanation_metrics.compute_explanation_diversity(explanation_data)
            quality_metrics = self.explanation_metrics.compute_explanation_quality_score(
                explanation_data,
                episode_data.get('human_ratings')
            )
            
            evaluation_results['explanation_quality'] = {
                **consistency_metrics,
                **diversity_metrics,
                **quality_metrics
            }
        
        # Human Study Simulation
        if explanation_data and 'ground_truth_actions' in episode_data:
            study_results = self.human_study_metrics.simulate_human_study(
                explanation_data,
                episode_data['ground_truth_actions']
            )
            evaluation_results['human_study'] = study_results
            evaluation_results['study_effectiveness'] = self.human_study_metrics.compute_study_effectiveness()
        
        # Overall System Score
        evaluation_results['overall_score'] = self._compute_overall_score(evaluation_results)
        
        self.evaluation_history.append(evaluation_results)
        return evaluation_results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute overall system performance score"""
        scores = []
        weights = []
        
        # Task performance (40% weight)
        if 'task_performance' in results:
            task_score = results['task_performance'].get('success_rate', 0.0)
            scores.append(task_score)
            weights.append(0.4)
        
        # Concept alignment (25% weight)
        if 'concept_alignment' in results:
            alignment_score = results['concept_alignment'].get('alignment_score', 0.0)
            scores.append(alignment_score)
            weights.append(0.25)
        
        # Explanation quality (25% weight)
        if 'explanation_quality' in results:
            quality_score = results['explanation_quality'].get('overall_quality', 0.0)
            scores.append(quality_score)
            weights.append(0.25)
        
        # Human study effectiveness (10% weight)
        if 'study_effectiveness' in results:
            study_score = results['study_effectiveness'].get('effectiveness_score', 0.0)
            scores.append(study_score)
            weights.append(0.1)
        
        if not scores:
            return 0.0
            
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_history:
            return {'error': 'No evaluation data available'}
            
        latest_eval = self.evaluation_history[-1]
        
        report = {
            'summary': {
                'total_evaluations': len(self.evaluation_history),
                'latest_overall_score': latest_eval.get('overall_score', 0.0),
                'evaluation_period': {
                    'start': self.evaluation_history[0]['timestamp'],
                    'end': latest_eval['timestamp']
                }
            },
            'performance_trends': self._compute_performance_trends(),
            'latest_metrics': latest_eval,
            'recommendations': self._generate_recommendations(latest_eval)
        }
        
        return report
    
    def _compute_performance_trends(self) -> Dict[str, Any]:
        """Compute performance trends over evaluation history"""
        if len(self.evaluation_history) < 2:
            return {}
            
        overall_scores = [eval_data.get('overall_score', 0.0) for eval_data in self.evaluation_history]
        
        return {
            'score_trend': np.polyfit(range(len(overall_scores)), overall_scores, 1)[0],  # Linear trend
            'score_improvement': overall_scores[-1] - overall_scores[0],
            'best_score': max(overall_scores),
            'worst_score': min(overall_scores),
            'score_stability': 1.0 - np.std(overall_scores)
        }
    
    def _generate_recommendations(self, latest_eval: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        # Task performance recommendations
        if 'task_performance' in latest_eval:
            success_rate = latest_eval['task_performance'].get('success_rate', 0.0)
            if success_rate < 0.7:
                recommendations.append("Consider increasing training episodes or adjusting reward structure")
        
        # Concept alignment recommendations
        if 'concept_alignment' in latest_eval:
            alignment_score = latest_eval['concept_alignment'].get('alignment_score', 0.0)
            if alignment_score < 0.6:
                recommendations.append("Improve concept-human alignment through better concept vocabulary or training")
        
        # Explanation quality recommendations
        if 'explanation_quality' in latest_eval:
            quality_score = latest_eval['explanation_quality'].get('overall_quality', 0.0)
            if quality_score < 0.7:
                recommendations.append("Enhance explanation generation with more diverse concept usage")
        
        # Human study recommendations
        if 'human_study' in latest_eval:
            trust_score = latest_eval['human_study'].get('trust_score', 0.0)
            if trust_score < 0.7:
                recommendations.append("Focus on improving explanation trustworthiness and consistency")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory across all metrics")
            
        return recommendations


# Alias for backward compatibility
ExplanationMetrics = ExplanationQualityMetrics


# Factory Functions

def create_default_metrics_evaluator(domain: str = "multi_agent") -> ComprehensiveMetricsEvaluator:
    """Create a default metrics evaluator with domain-specific configuration"""
    config = MetricsConfig()
    
    # Domain-specific adjustments
    if domain == "multi_agent":
        config.alignment_threshold = 0.6
        config.performance_window = 100
    elif domain == "human_ai_collaboration":
        config.alignment_threshold = 0.7
        config.explanation_quality_threshold = 0.8
    
    return ComprehensiveMetricsEvaluator(config)


def compute_quick_evaluation(agent_results: Dict[str, Any],
                           concept_vocab: Any,
                           explanations: List[Dict]) -> Dict[str, float]:
    """
    Quick evaluation function for rapid assessment
    
    Args:
        agent_results: Agent performance results
        concept_vocab: Concept vocabulary
        explanations: Generated explanations
        
    Returns:
        Quick evaluation metrics
    """
    metrics = {}
    
    # Quick performance metrics
    if 'rewards' in agent_results:
        metrics['avg_reward'] = np.mean(agent_results['rewards'])
        metrics['reward_std'] = np.std(agent_results['rewards'])
    
    # Quick explanation metrics
    if explanations:
        metrics['avg_concepts_per_explanation'] = np.mean([
            len(exp.get('top_concepts', [])) for exp in explanations
        ])
        metrics['explanation_coverage'] = len(set().union(*[
            exp.get('top_concepts', []) for exp in explanations
        ])) / getattr(concept_vocab, 'concept_dim', 1)
    
    return metrics


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ACB-Agent Metrics System...")
    
    # Create metrics evaluator
    evaluator = create_default_metrics_evaluator("multi_agent")
    
    # Simulate evaluation data
    episode_data = {
        'episode_id': 1,
        'reward': 150.0,
        'length': 200,
        'success': True,
        'ground_truth_actions': ['move_left', 'move_right', 'stay']
    }
    
    concept_data = {
        'model_concepts': torch.randn(10, 16),  # 10 samples, 16 concepts
        'human_labels': torch.randn(10, 16),
        'concept_names': [f'concept_{i}' for i in range(16)],
        'states': torch.randn(3, 20)  # 3 states, 20 features
    }
    
    explanation_data = [
        {
            'top_concepts': ['obstacle_detected', 'goal_nearby'],
            'explanation': 'Moving left because obstacle detected and goal is nearby',
            'quality_score': 0.8
        },
        {
            'top_concepts': ['clear_path', 'teammate_coordination'],
            'explanation': 'Moving right due to clear path and teammate coordination',
            'quality_score': 0.9
        },
        {
            'top_concepts': ['optimal_position', 'resource_available'],
            'explanation': 'Staying in place as optimal position with resource available',
            'quality_score': 0.7
        }
    ]
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_system(episode_data, concept_data, explanation_data)
    
    print("Evaluation Results:")
    print(f"Overall Score: {results.get('overall_score', 0.0):.3f}")
    
    if 'task_performance' in results:
        print(f"Task Performance - Success Rate: {results['task_performance'].get('success_rate', 0.0):.3f}")
    
    if 'concept_alignment' in results:
        print(f"Concept Alignment Score: {results['concept_alignment'].get('alignment_score', 0.0):.3f}")
    
    if 'explanation_quality' in results:
        print(f"Explanation Quality: {results['explanation_quality'].get('overall_quality', 0.0):.3f}")
    
    if 'human_study' in results:
        print(f"Human Study Trust Score: {results['human_study'].get('trust_score', 0.0):.3f}")
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print(f"\nRecommendations: {report.get('recommendations', [])}")
    
    print("Metrics system testing completed successfully!")