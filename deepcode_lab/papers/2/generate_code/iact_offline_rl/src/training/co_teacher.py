"""
Co-Teaching Training Logic for IACT Algorithm

This module implements the core co-teaching mechanism where dual policies
teach each other by selecting high-quality samples based on importance weights
and policy confidence. The co-teaching framework helps mitigate the impact
of noisy or low-quality samples in offline RL datasets.

Key Components:
- CoTeacher: Main co-teaching coordinator
- PolicyPair: Manages dual policy interactions
- TeachingMetrics: Tracks co-teaching statistics
- SampleExchange: Handles bidirectional sample selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, deque

# Import dependencies
from ..algorithms.sample_selector import SampleSelector, SelectionMetrics
from .bc_regularizer import BCRegularizer, create_bc_regularizer
from ..utils.metrics import MetricsTracker


class TeachingStrategy(Enum):
    """Co-teaching strategies for sample selection"""
    CONFIDENCE_BASED = "confidence"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class CoTeachingConfig:
    """Configuration for co-teaching mechanism"""
    # Core parameters
    initial_selection_rate: float = 0.8
    final_selection_rate: float = 0.4
    selection_decay_steps: int = 50000
    
    # Teaching strategies
    teaching_strategy: TeachingStrategy = TeachingStrategy.HYBRID
    confidence_weight: float = 0.6
    importance_weight: float = 0.4
    
    # Adaptive parameters
    use_adaptive_rates: bool = True
    adaptation_window: int = 1000
    performance_threshold: float = 0.05
    
    # Sample exchange
    exchange_frequency: int = 100
    min_exchange_samples: int = 32
    max_exchange_samples: int = 512
    
    # Regularization
    use_bc_regularization: bool = True
    bc_weight: float = 0.1
    
    # Monitoring
    track_teaching_quality: bool = True
    log_frequency: int = 500
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TeachingMetrics:
    """Metrics for tracking co-teaching performance"""
    # Selection metrics
    selection_rate_a: float = 0.0
    selection_rate_b: float = 0.0
    avg_selection_rate: float = 0.0
    
    # Quality metrics
    avg_sample_quality_a: float = 0.0
    avg_sample_quality_b: float = 0.0
    quality_improvement: float = 0.0
    
    # Teaching effectiveness
    teaching_agreement: float = 0.0  # How often policies agree on sample quality
    cross_validation_score: float = 0.0
    
    # Performance tracking
    policy_a_loss: float = 0.0
    policy_b_loss: float = 0.0
    bc_loss_a: float = 0.0
    bc_loss_b: float = 0.0
    
    # Timing
    selection_time: float = 0.0
    training_time: float = 0.0
    
    # Adaptive metrics
    adaptation_triggered: bool = False
    rate_adjustment: float = 0.0


class PolicyPair:
    """Manages interactions between dual policies in co-teaching"""
    
    def __init__(self, policy_a, policy_b, config: CoTeachingConfig):
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.config = config
        
        # Performance tracking
        self.performance_history_a = deque(maxlen=config.adaptation_window)
        self.performance_history_b = deque(maxlen=config.adaptation_window)
        
        # Teaching statistics
        self.teaching_stats = {
            'samples_taught_by_a': 0,
            'samples_taught_by_b': 0,
            'total_exchanges': 0,
            'quality_improvements': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def compute_policy_confidence(self, policy, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute policy confidence scores for given state-action pairs"""
        with torch.no_grad():
            # Get policy distribution
            if hasattr(policy, 'get_action_distribution'):
                dist = policy.get_action_distribution(states)
                log_probs = dist.log_prob(actions)
            else:
                # Fallback for different policy interfaces
                log_probs = policy.log_prob(states, actions)
            
            # Convert log probabilities to confidence scores
            confidence = torch.exp(log_probs.clamp(min=-10, max=2))  # Clamp for numerical stability
            
            return confidence
    
    def compute_teaching_quality(self, 
                               selecting_policy, 
                               target_policy,
                               batch: Dict[str, torch.Tensor],
                               importance_weights: torch.Tensor) -> torch.Tensor:
        """Compute teaching quality scores for sample selection"""
        states = batch['states']
        actions = batch['actions']
        
        # Get confidence from selecting policy
        selecting_confidence = self.compute_policy_confidence(selecting_policy, states, actions)
        
        # Get confidence from target policy (for cross-validation)
        target_confidence = self.compute_policy_confidence(target_policy, states, actions)
        
        if self.config.teaching_strategy == TeachingStrategy.CONFIDENCE_BASED:
            quality_scores = selecting_confidence
            
        elif self.config.teaching_strategy == TeachingStrategy.IMPORTANCE_WEIGHTED:
            quality_scores = importance_weights
            
        elif self.config.teaching_strategy == TeachingStrategy.HYBRID:
            # Combine confidence and importance weights
            normalized_confidence = F.normalize(selecting_confidence.unsqueeze(0), dim=1).squeeze(0)
            normalized_importance = F.normalize(importance_weights.unsqueeze(0), dim=1).squeeze(0)
            
            quality_scores = (self.config.confidence_weight * normalized_confidence + 
                            self.config.importance_weight * normalized_importance)
            
        elif self.config.teaching_strategy == TeachingStrategy.ADAPTIVE:
            # Adaptive combination based on recent performance
            if len(self.performance_history_a) > 10 and len(self.performance_history_b) > 10:
                recent_perf_a = np.mean(list(self.performance_history_a)[-10:])
                recent_perf_b = np.mean(list(self.performance_history_b)[-10:])
                
                # Adjust weights based on relative performance
                if recent_perf_a > recent_perf_b:
                    conf_weight = min(0.8, self.config.confidence_weight + 0.1)
                else:
                    conf_weight = max(0.2, self.config.confidence_weight - 0.1)
                
                imp_weight = 1.0 - conf_weight
            else:
                conf_weight = self.config.confidence_weight
                imp_weight = self.config.importance_weight
            
            normalized_confidence = F.normalize(selecting_confidence.unsqueeze(0), dim=1).squeeze(0)
            normalized_importance = F.normalize(importance_weights.unsqueeze(0), dim=1).squeeze(0)
            
            quality_scores = (conf_weight * normalized_confidence + 
                            imp_weight * normalized_importance)
        
        return quality_scores
    
    def update_performance_history(self, policy_name: str, loss: float):
        """Update performance history for adaptive teaching"""
        if policy_name == 'a':
            self.performance_history_a.append(loss)
        elif policy_name == 'b':
            self.performance_history_b.append(loss)


class SampleExchange:
    """Handles bidirectional sample exchange between policies"""
    
    def __init__(self, config: CoTeachingConfig):
        self.config = config
        self.exchange_history = []
        self.logger = logging.getLogger(__name__)
    
    def select_samples_for_policy(self,
                                selecting_policy,
                                target_policy,
                                target_policy_name: str,
                                batch: Dict[str, torch.Tensor],
                                importance_weights: torch.Tensor,
                                quality_scores: torch.Tensor,
                                selection_rate: float) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Select samples for target policy training"""
        batch_size = len(batch['states'])
        num_select = max(
            self.config.min_exchange_samples,
            min(self.config.max_exchange_samples, int(batch_size * selection_rate))
        )
        
        # Select top-k samples based on quality scores
        top_k_indices = torch.topk(quality_scores, num_select, largest=True).indices
        
        # Create selected batch
        selected_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                selected_batch[key] = value[top_k_indices]
            else:
                selected_batch[key] = [value[i] for i in top_k_indices.cpu().numpy()]
        
        # Get selected importance weights
        selected_importance_weights = importance_weights[top_k_indices]
        
        # Log exchange statistics
        self.exchange_history.append({
            'target_policy': target_policy_name,
            'selected_samples': num_select,
            'total_samples': batch_size,
            'selection_rate': selection_rate,
            'avg_quality': quality_scores[top_k_indices].mean().item(),
            'avg_importance': selected_importance_weights.mean().item()
        })
        
        return selected_batch, selected_importance_weights
    
    def get_exchange_statistics(self) -> Dict[str, float]:
        """Get statistics about recent sample exchanges"""
        if not self.exchange_history:
            return {}
        
        recent_exchanges = self.exchange_history[-100:]  # Last 100 exchanges
        
        stats = {
            'avg_selection_rate': np.mean([ex['selection_rate'] for ex in recent_exchanges]),
            'avg_quality_score': np.mean([ex['avg_quality'] for ex in recent_exchanges]),
            'avg_importance_weight': np.mean([ex['avg_importance'] for ex in recent_exchanges]),
            'total_exchanges': len(self.exchange_history)
        }
        
        return stats


class CoTeacher:
    """Main co-teaching coordinator for IACT algorithm"""
    
    def __init__(self, 
                 policy_a,
                 policy_b,
                 sample_selector: SampleSelector,
                 config: CoTeachingConfig = None):
        self.config = config or CoTeachingConfig()
        self.policy_pair = PolicyPair(policy_a, policy_b, self.config)
        self.sample_selector = sample_selector
        self.sample_exchange = SampleExchange(self.config)
        
        # Initialize BC regularizers if enabled
        if self.config.use_bc_regularization:
            self.bc_regularizer_a = create_bc_regularizer(
                regularization_type="kl_divergence",
                initial_weight=self.config.bc_weight,
                final_weight=self.config.bc_weight * 0.5,
                use_importance_weights=True
            )
            self.bc_regularizer_b = create_bc_regularizer(
                regularization_type="kl_divergence",
                initial_weight=self.config.bc_weight,
                final_weight=self.config.bc_weight * 0.5,
                use_importance_weights=True
            )
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Adaptive rate tracking
        self.recent_performance = deque(maxlen=self.config.adaptation_window)
        self.last_adaptation_step = 0
    
    def get_current_selection_rate(self) -> float:
        """Compute current selection rate based on curriculum schedule"""
        if self.step_count >= self.config.selection_decay_steps:
            return self.config.final_selection_rate
        
        # Linear decay from initial to final rate
        progress = self.step_count / self.config.selection_decay_steps
        current_rate = (self.config.initial_selection_rate - 
                       (self.config.initial_selection_rate - self.config.final_selection_rate) * progress)
        
        return current_rate
    
    def adapt_selection_rate(self, current_performance: float) -> Tuple[float, bool]:
        """Adapt selection rate based on recent performance"""
        if not self.config.use_adaptive_rates:
            return self.get_current_selection_rate(), False
        
        self.recent_performance.append(current_performance)
        
        # Check if adaptation should be triggered
        if (len(self.recent_performance) >= self.config.adaptation_window and
            self.step_count - self.last_adaptation_step >= self.config.adaptation_window):
            
            recent_perf = np.mean(list(self.recent_performance)[-50:])
            older_perf = np.mean(list(self.recent_performance)[-100:-50]) if len(self.recent_performance) >= 100 else recent_perf
            
            performance_change = recent_perf - older_perf
            
            base_rate = self.get_current_selection_rate()
            
            if performance_change < -self.config.performance_threshold:
                # Performance degrading, increase selection rate (be more selective)
                adapted_rate = min(0.9, base_rate + 0.05)
                self.last_adaptation_step = self.step_count
                return adapted_rate, True
            elif performance_change > self.config.performance_threshold:
                # Performance improving, can afford to be less selective
                adapted_rate = max(0.1, base_rate - 0.02)
                self.last_adaptation_step = self.step_count
                return adapted_rate, True
        
        return self.get_current_selection_rate(), False
    
    def co_teaching_step(self,
                        batch: Dict[str, torch.Tensor],
                        importance_weights: torch.Tensor,
                        current_performance: float = 0.0) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], TeachingMetrics]:
        """Perform one co-teaching step with bidirectional sample selection"""
        start_time = time.time()
        
        # Get current selection rate (with potential adaptation)
        selection_rate, adaptation_triggered = self.adapt_selection_rate(current_performance)
        
        # Compute teaching quality scores for both directions
        quality_scores_a_to_b = self.policy_pair.compute_teaching_quality(
            self.policy_pair.policy_a, self.policy_pair.policy_b, batch, importance_weights
        )
        quality_scores_b_to_a = self.policy_pair.compute_teaching_quality(
            self.policy_pair.policy_b, self.policy_pair.policy_a, batch, importance_weights
        )
        
        selection_time = time.time()
        
        # Policy A selects samples for Policy B
        batch_for_b, importance_weights_b = self.sample_exchange.select_samples_for_policy(
            self.policy_pair.policy_a, self.policy_pair.policy_b, 'b',
            batch, importance_weights, quality_scores_a_to_b, selection_rate
        )
        
        # Policy B selects samples for Policy A
        batch_for_a, importance_weights_a = self.sample_exchange.select_samples_for_policy(
            self.policy_pair.policy_b, self.policy_pair.policy_a, 'a',
            batch, importance_weights, quality_scores_b_to_a, selection_rate
        )
        
        selection_end_time = time.time()
        
        # Compute teaching metrics
        metrics = self._compute_teaching_metrics(
            batch_for_a, batch_for_b, importance_weights_a, importance_weights_b,
            quality_scores_a_to_b, quality_scores_b_to_a, selection_rate,
            adaptation_triggered, selection_end_time - selection_time,
            time.time() - start_time
        )
        
        # Update step count
        self.step_count += 1
        
        # Log metrics periodically
        if self.step_count % self.config.log_frequency == 0:
            self._log_teaching_metrics(metrics)
        
        return batch_for_a, batch_for_b, metrics
    
    def compute_bc_losses(self,
                         batch_a: Dict[str, torch.Tensor],
                         batch_b: Dict[str, torch.Tensor],
                         importance_weights_a: torch.Tensor,
                         importance_weights_b: torch.Tensor,
                         epoch: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute behavior cloning regularization losses"""
        if not self.config.use_bc_regularization:
            return {}, {}
        
        # Compute BC loss for policy A
        bc_loss_a = self.bc_regularizer_a.compute_bc_loss(
            policy_actions=batch_a['actions'],  # Actions from policy A
            behavior_actions=batch_a['actions'],  # Original behavior actions
            importance_weights=importance_weights_a,
            states=batch_a['states']
        )
        
        # Compute BC loss for policy B
        bc_loss_b = self.bc_regularizer_b.compute_bc_loss(
            policy_actions=batch_b['actions'],  # Actions from policy B
            behavior_actions=batch_b['actions'],  # Original behavior actions
            importance_weights=importance_weights_b,
            states=batch_b['states']
        )
        
        # Update regularization weights
        self.bc_regularizer_a.update_weight(epoch)
        self.bc_regularizer_b.update_weight(epoch)
        
        return bc_loss_a, bc_loss_b
    
    def _compute_teaching_metrics(self,
                                batch_for_a: Dict[str, torch.Tensor],
                                batch_for_b: Dict[str, torch.Tensor],
                                importance_weights_a: torch.Tensor,
                                importance_weights_b: torch.Tensor,
                                quality_scores_a_to_b: torch.Tensor,
                                quality_scores_b_to_a: torch.Tensor,
                                selection_rate: float,
                                adaptation_triggered: bool,
                                selection_time: float,
                                total_time: float) -> TeachingMetrics:
        """Compute comprehensive teaching metrics"""
        
        # Selection rates
        original_batch_size = len(quality_scores_a_to_b)
        actual_rate_a = len(batch_for_a['states']) / original_batch_size
        actual_rate_b = len(batch_for_b['states']) / original_batch_size
        
        # Quality metrics
        avg_quality_a = importance_weights_a.mean().item() if len(importance_weights_a) > 0 else 0.0
        avg_quality_b = importance_weights_b.mean().item() if len(importance_weights_b) > 0 else 0.0
        
        # Teaching agreement (correlation between quality scores)
        if len(quality_scores_a_to_b) > 1 and len(quality_scores_b_to_a) > 1:
            correlation = torch.corrcoef(torch.stack([quality_scores_a_to_b, quality_scores_b_to_a]))[0, 1]
            teaching_agreement = correlation.item() if not torch.isnan(correlation) else 0.0
        else:
            teaching_agreement = 0.0
        
        # Quality improvement (compared to random selection)
        random_quality_a = importance_weights_a.mean().item() if len(importance_weights_a) > 0 else 0.0
        random_quality_b = importance_weights_b.mean().item() if len(importance_weights_b) > 0 else 0.0
        quality_improvement = (avg_quality_a + avg_quality_b) / 2 - (random_quality_a + random_quality_b) / 2
        
        return TeachingMetrics(
            selection_rate_a=actual_rate_a,
            selection_rate_b=actual_rate_b,
            avg_selection_rate=(actual_rate_a + actual_rate_b) / 2,
            avg_sample_quality_a=avg_quality_a,
            avg_sample_quality_b=avg_quality_b,
            quality_improvement=quality_improvement,
            teaching_agreement=teaching_agreement,
            cross_validation_score=teaching_agreement,  # Using agreement as cross-validation proxy
            selection_time=selection_time,
            training_time=total_time,
            adaptation_triggered=adaptation_triggered,
            rate_adjustment=selection_rate - self.get_current_selection_rate() if adaptation_triggered else 0.0
        )
    
    def _log_teaching_metrics(self, metrics: TeachingMetrics):
        """Log teaching metrics"""
        self.logger.info(f"Co-Teaching Step {self.step_count}:")
        self.logger.info(f"  Selection Rates: A={metrics.selection_rate_a:.3f}, B={metrics.selection_rate_b:.3f}")
        self.logger.info(f"  Sample Quality: A={metrics.avg_sample_quality_a:.3f}, B={metrics.avg_sample_quality_b:.3f}")
        self.logger.info(f"  Teaching Agreement: {metrics.teaching_agreement:.3f}")
        self.logger.info(f"  Quality Improvement: {metrics.quality_improvement:.3f}")
        
        if metrics.adaptation_triggered:
            self.logger.info(f"  Adaptive Rate Adjustment: {metrics.rate_adjustment:+.3f}")
    
    def get_teaching_statistics(self) -> Dict[str, Any]:
        """Get comprehensive teaching statistics"""
        exchange_stats = self.sample_exchange.get_exchange_statistics()
        
        stats = {
            'step_count': self.step_count,
            'current_selection_rate': self.get_current_selection_rate(),
            'total_adaptations': self.step_count - self.last_adaptation_step,
            'exchange_statistics': exchange_stats,
            'policy_pair_stats': self.policy_pair.teaching_stats,
            'recent_performance_window': len(self.recent_performance)
        }
        
        return stats
    
    def reset_teaching_state(self):
        """Reset co-teaching state for new training session"""
        self.step_count = 0
        self.last_adaptation_step = 0
        self.recent_performance.clear()
        self.sample_exchange.exchange_history.clear()
        self.policy_pair.performance_history_a.clear()
        self.policy_pair.performance_history_b.clear()
        self.policy_pair.teaching_stats = {
            'samples_taught_by_a': 0,
            'samples_taught_by_b': 0,
            'total_exchanges': 0,
            'quality_improvements': []
        }


def create_co_teacher(policy_a, policy_b, sample_selector: SampleSelector, 
                     config: Dict[str, Any] = None) -> CoTeacher:
    """Factory function to create CoTeacher instance"""
    if config is None:
        config = {}
    
    # Convert dict config to CoTeachingConfig
    co_teaching_config = CoTeachingConfig(**config)
    
    return CoTeacher(policy_a, policy_b, sample_selector, co_teaching_config)


# Utility functions for co-teaching analysis
def analyze_teaching_effectiveness(co_teacher: CoTeacher, 
                                 num_steps: int = 1000) -> Dict[str, float]:
    """Analyze the effectiveness of co-teaching over recent steps"""
    stats = co_teacher.get_teaching_statistics()
    exchange_history = co_teacher.sample_exchange.exchange_history[-num_steps:]
    
    if not exchange_history:
        return {'effectiveness_score': 0.0}
    
    # Compute effectiveness metrics
    avg_quality = np.mean([ex['avg_quality'] for ex in exchange_history])
    quality_variance = np.var([ex['avg_quality'] for ex in exchange_history])
    selection_consistency = 1.0 - np.std([ex['selection_rate'] for ex in exchange_history])
    
    effectiveness_score = (avg_quality * 0.5 + 
                          (1.0 / (1.0 + quality_variance)) * 0.3 + 
                          selection_consistency * 0.2)
    
    return {
        'effectiveness_score': effectiveness_score,
        'avg_quality': avg_quality,
        'quality_variance': quality_variance,
        'selection_consistency': selection_consistency,
        'total_exchanges_analyzed': len(exchange_history)
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Co-Teaching Module for IACT Algorithm")
    print("=====================================")
    
    # Test configuration
    config = CoTeachingConfig(
        initial_selection_rate=0.8,
        final_selection_rate=0.4,
        teaching_strategy=TeachingStrategy.HYBRID,
        use_adaptive_rates=True
    )
    
    print(f"Configuration: {config}")
    print(f"Teaching Strategy: {config.teaching_strategy}")
    print(f"Selection Rate Range: {config.initial_selection_rate} -> {config.final_selection_rate}")
    print(f"Adaptive Rates: {config.use_adaptive_rates}")