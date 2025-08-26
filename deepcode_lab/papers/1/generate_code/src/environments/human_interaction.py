"""
Human-AI Interaction Environment
Implements Algorithm 2 from Section 4.2 of the paper:
"Robust Explanations for Human-Neural Multi-Agent Systems via Approximate Concept Bottlenecks"

This module provides the human-AI interaction protocol that:
1. Extracts concepts from states
2. Generates actions through concept space
3. Creates explanations for human understanding
4. Collects human feedback on explanation quality
5. Enables iterative improvement of explanations
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import random
import json

# Import our components
from ..models.acb_agent import ACBAgent, MultiAgentACBSystem
from ..utils.explanation_generator import ExplanationGenerator, InteractiveExplanationGenerator, ExplanationStyle
from ..utils.concept_utils import ConceptVocabulary, extract_top_k_concepts, compute_concept_alignment


class FeedbackType(Enum):
    """Types of human feedback on explanations"""
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    REQUEST_MORE = "request_more"
    UNCLEAR = "unclear"


class InteractionMode(Enum):
    """Modes of human-AI interaction"""
    REAL_TIME = "real_time"  # Real human interaction
    SIMULATED = "simulated"  # Simulated human responses
    BATCH = "batch"  # Batch processing mode


@dataclass
class HumanFeedback:
    """Structure for human feedback on explanations"""
    feedback_type: FeedbackType
    explanation_quality: float  # 0-1 score
    clarity_score: float  # 0-1 score
    usefulness_score: float  # 0-1 score
    suggested_concepts: Optional[List[str]] = None
    comments: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class InteractionResult:
    """Result of a single human-AI interaction"""
    state: torch.Tensor
    action: Union[int, torch.Tensor]
    concept_activations: torch.Tensor
    explanation: str
    top_concepts: List[Tuple[str, float]]
    human_feedback: Optional[HumanFeedback]
    interaction_time: float
    success: bool


class HumanSimulator:
    """Simulates human responses for testing and evaluation"""
    
    def __init__(self, 
                 concept_vocab: ConceptVocabulary,
                 noise_level: float = 0.1,
                 preference_bias: Optional[Dict[str, float]] = None):
        """
        Initialize human simulator
        
        Args:
            concept_vocab: Concept vocabulary for understanding
            noise_level: Amount of noise in simulated responses
            preference_bias: Bias towards certain concepts
        """
        self.concept_vocab = concept_vocab
        self.noise_level = noise_level
        self.preference_bias = preference_bias or {}
        self.interaction_history = []
        
    def simulate_feedback(self, 
                         explanation: str,
                         top_concepts: List[Tuple[str, float]],
                         ground_truth_concepts: Optional[List[str]] = None) -> HumanFeedback:
        """
        Simulate human feedback on an explanation
        
        Args:
            explanation: Generated explanation text
            top_concepts: Top concepts used in explanation
            ground_truth_concepts: True concepts for comparison
            
        Returns:
            Simulated human feedback
        """
        # Base quality score based on concept relevance
        base_quality = 0.7
        
        # Adjust based on concept preferences
        concept_names = [concept[0] for concept in top_concepts]
        preference_bonus = sum(self.preference_bias.get(name, 0) for name in concept_names) / len(concept_names)
        
        # Adjust based on ground truth alignment if available
        alignment_bonus = 0.0
        if ground_truth_concepts:
            overlap = len(set(concept_names) & set(ground_truth_concepts))
            alignment_bonus = overlap / max(len(ground_truth_concepts), 1) * 0.2
        
        # Add noise
        noise = np.random.normal(0, self.noise_level)
        
        # Calculate scores
        explanation_quality = np.clip(base_quality + preference_bonus + alignment_bonus + noise, 0, 1)
        clarity_score = np.clip(explanation_quality + np.random.normal(0, 0.05), 0, 1)
        usefulness_score = np.clip(explanation_quality + np.random.normal(0, 0.05), 0, 1)
        
        # Determine feedback type based on quality
        if explanation_quality > 0.8:
            feedback_type = FeedbackType.ACCEPT
        elif explanation_quality > 0.6:
            feedback_type = FeedbackType.MODIFY
        elif explanation_quality > 0.4:
            feedback_type = FeedbackType.REQUEST_MORE
        else:
            feedback_type = FeedbackType.REJECT
        
        # Generate suggested concepts for modify feedback
        suggested_concepts = None
        if feedback_type == FeedbackType.MODIFY and ground_truth_concepts:
            # Suggest some ground truth concepts not in current explanation
            missing_concepts = set(ground_truth_concepts) - set(concept_names)
            if missing_concepts:
                suggested_concepts = list(missing_concepts)[:2]
        
        return HumanFeedback(
            feedback_type=feedback_type,
            explanation_quality=explanation_quality,
            clarity_score=clarity_score,
            usefulness_score=usefulness_score,
            suggested_concepts=suggested_concepts,
            comments=f"Simulated feedback: {feedback_type.value}",
            timestamp=None
        )


class HumanInteractionEnvironment:
    """
    Human-AI Interaction Environment implementing Algorithm 2
    
    This environment wraps multi-agent systems to provide human-interpretable
    explanations and collect feedback for improving explanation quality.
    """
    
    def __init__(self,
                 agent_system: Union[ACBAgent, MultiAgentACBSystem],
                 concept_vocab: ConceptVocabulary,
                 explanation_generator: Optional[ExplanationGenerator] = None,
                 interaction_mode: InteractionMode = InteractionMode.SIMULATED,
                 human_simulator: Optional[HumanSimulator] = None,
                 k_concepts: int = 3,
                 explanation_style: ExplanationStyle = ExplanationStyle.SIMPLE,
                 device: torch.device = None):
        """
        Initialize Human-AI Interaction Environment
        
        Args:
            agent_system: ACB agent or multi-agent system
            concept_vocab: Concept vocabulary for explanations
            explanation_generator: Custom explanation generator
            interaction_mode: Mode of interaction (real-time, simulated, batch)
            human_simulator: Simulator for human responses
            k_concepts: Number of top concepts to include in explanations
            explanation_style: Style of explanations to generate
            device: Torch device for computations
        """
        self.agent_system = agent_system
        self.concept_vocab = concept_vocab
        self.interaction_mode = interaction_mode
        self.k_concepts = k_concepts
        self.device = device or torch.device('cpu')
        
        # Initialize explanation generator
        if explanation_generator is None:
            self.explanation_generator = InteractiveExplanationGenerator(
                concept_vocab=concept_vocab,
                style=explanation_style,
                k_concepts=k_concepts
            )
        else:
            self.explanation_generator = explanation_generator
        
        # Initialize human simulator if needed
        if interaction_mode == InteractionMode.SIMULATED:
            if human_simulator is None:
                self.human_simulator = HumanSimulator(concept_vocab)
            else:
                self.human_simulator = human_simulator
        else:
            self.human_simulator = human_simulator
        
        # Interaction history and metrics
        self.interaction_history: List[InteractionResult] = []
        self.feedback_history: List[HumanFeedback] = []
        self.explanation_quality_history: List[float] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def interact_with_human(self, 
                           state: torch.Tensor,
                           ground_truth_concepts: Optional[List[str]] = None,
                           return_detailed_info: bool = False) -> Dict[str, Any]:
        """
        Implement Algorithm 2: Human-AI Interaction Protocol
        
        Algorithm 2 Steps:
        1. Extract concepts: c = CB_φ(s)
        2. Generate action: a = π_θ(c)
        3. Create explanation: E = top_k_concepts(c, k=3)
        4. Present (action, explanation) to human
        5. Collect feedback on explanation quality
        
        Args:
            state: Current environment state
            ground_truth_concepts: True concepts for evaluation
            return_detailed_info: Whether to return detailed interaction info
            
        Returns:
            Dictionary containing interaction results
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Extract concepts c = CB_φ(s)
            # Step 2: Generate action a = π_θ(c)
            if isinstance(self.agent_system, MultiAgentACBSystem):
                # Multi-agent case
                if state.dim() == 1:
                    states = [state]  # Single state for all agents
                else:
                    states = [state[i] for i in range(state.shape[0])]
                
                agent_results = self.agent_system.act(
                    states=states,
                    deterministic=False,
                    return_explanations=True
                )
                
                # Use first agent's results for primary interaction
                primary_result = agent_results[0]
                action = primary_result['action']
                concept_activations = primary_result['concept_activations']
                
            else:
                # Single agent case
                agent_result = self.agent_system.act(
                    state=state,
                    deterministic=False,
                    return_explanation=True,
                    explanation_k=self.k_concepts
                )
                action = agent_result['action']
                concept_activations = agent_result['concept_activations']
            
            # Step 3: Create explanation E = top_k_concepts(c, k=3)
            top_concepts = extract_top_k_concepts(
                concept_activations.unsqueeze(0),
                self.concept_vocab,
                k=self.k_concepts
            )[0]  # Get first (and only) batch element
            
            # Generate detailed explanation
            explanation_result = self.explanation_generator.generate_explanation(
                concept_activations=concept_activations,
                action=str(action.item()) if torch.is_tensor(action) else str(action),
                state_info={'state_tensor': state}
            )
            
            explanation = explanation_result['explanation']
            
            # Step 4 & 5: Present to human and collect feedback
            human_feedback = None
            if self.interaction_mode == InteractionMode.SIMULATED:
                human_feedback = self.human_simulator.simulate_feedback(
                    explanation=explanation,
                    top_concepts=top_concepts,
                    ground_truth_concepts=ground_truth_concepts
                )
            elif self.interaction_mode == InteractionMode.REAL_TIME:
                # In real implementation, this would present to actual human
                human_feedback = self._collect_real_human_feedback(
                    explanation, top_concepts, action
                )
            
            # Record interaction
            interaction_time = time.time() - start_time
            interaction_result = InteractionResult(
                state=state,
                action=action,
                concept_activations=concept_activations,
                explanation=explanation,
                top_concepts=top_concepts,
                human_feedback=human_feedback,
                interaction_time=interaction_time,
                success=True
            )
            
            # Update history
            self.interaction_history.append(interaction_result)
            if human_feedback:
                self.feedback_history.append(human_feedback)
                self.explanation_quality_history.append(human_feedback.explanation_quality)
            
            # Prepare return dictionary
            result = {
                'action': action,
                'explanation': explanation,
                'top_concepts': top_concepts,
                'concept_activations': concept_activations,
                'human_feedback': human_feedback,
                'interaction_time': interaction_time,
                'success': True
            }
            
            if return_detailed_info:
                result.update({
                    'state': state,
                    'explanation_quality_score': human_feedback.explanation_quality if human_feedback else None,
                    'clarity_score': human_feedback.clarity_score if human_feedback else None,
                    'usefulness_score': human_feedback.usefulness_score if human_feedback else None,
                    'interaction_result': interaction_result
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in human interaction: {e}")
            return {
                'action': None,
                'explanation': f"Error: {str(e)}",
                'top_concepts': [],
                'concept_activations': None,
                'human_feedback': None,
                'interaction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _collect_real_human_feedback(self, 
                                   explanation: str, 
                                   top_concepts: List[Tuple[str, float]], 
                                   action: Any) -> HumanFeedback:
        """
        Collect feedback from real human (placeholder for actual implementation)
        
        In a real system, this would:
        1. Display the explanation to the human
        2. Collect their feedback through UI
        3. Return structured feedback
        
        Args:
            explanation: Generated explanation
            top_concepts: Top concepts used
            action: Action taken
            
        Returns:
            Human feedback structure
        """
        # Placeholder implementation - in real system would use GUI/web interface
        print(f"\n=== Human-AI Interaction ===")
        print(f"Action taken: {action}")
        print(f"Explanation: {explanation}")
        print(f"Top concepts: {[f'{name} ({score:.2f})' for name, score in top_concepts]}")
        print("\nPlease provide feedback (simulated for now):")
        
        # For now, return simulated feedback
        return HumanFeedback(
            feedback_type=FeedbackType.ACCEPT,
            explanation_quality=0.8,
            clarity_score=0.8,
            usefulness_score=0.8,
            comments="Placeholder real human feedback"
        )
    
    def batch_interact(self, 
                      states: torch.Tensor,
                      ground_truth_concepts_batch: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Process batch of states through human-AI interaction
        
        Args:
            states: Batch of states [batch_size, state_dim]
            ground_truth_concepts_batch: Ground truth concepts for each state
            
        Returns:
            List of interaction results
        """
        results = []
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            state = states[i]
            ground_truth = ground_truth_concepts_batch[i] if ground_truth_concepts_batch else None
            
            result = self.interact_with_human(
                state=state,
                ground_truth_concepts=ground_truth,
                return_detailed_info=True
            )
            results.append(result)
        
        return results
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about human-AI interactions
        
        Returns:
            Dictionary with interaction statistics
        """
        if not self.interaction_history:
            return {'message': 'No interactions recorded yet'}
        
        # Basic statistics
        total_interactions = len(self.interaction_history)
        successful_interactions = sum(1 for r in self.interaction_history if r.success)
        success_rate = successful_interactions / total_interactions
        
        # Timing statistics
        interaction_times = [r.interaction_time for r in self.interaction_history if r.success]
        avg_interaction_time = np.mean(interaction_times) if interaction_times else 0
        
        # Feedback statistics
        if self.explanation_quality_history:
            avg_explanation_quality = np.mean(self.explanation_quality_history)
            explanation_quality_trend = np.polyfit(
                range(len(self.explanation_quality_history)), 
                self.explanation_quality_history, 
                1
            )[0] if len(self.explanation_quality_history) > 1 else 0
        else:
            avg_explanation_quality = 0
            explanation_quality_trend = 0
        
        # Feedback type distribution
        feedback_types = [f.feedback_type.value for f in self.feedback_history]
        feedback_distribution = {
            ftype.value: feedback_types.count(ftype.value) / len(feedback_types) 
            for ftype in FeedbackType
        } if feedback_types else {}
        
        return {
            'total_interactions': total_interactions,
            'successful_interactions': successful_interactions,
            'success_rate': success_rate,
            'avg_interaction_time': avg_interaction_time,
            'avg_explanation_quality': avg_explanation_quality,
            'explanation_quality_trend': explanation_quality_trend,
            'feedback_distribution': feedback_distribution,
            'recent_quality_scores': self.explanation_quality_history[-10:] if self.explanation_quality_history else []
        }
    
    def improve_explanations_from_feedback(self) -> Dict[str, Any]:
        """
        Analyze feedback to improve explanation generation
        
        Returns:
            Dictionary with improvement suggestions
        """
        if not self.feedback_history:
            return {'message': 'No feedback available for improvement'}
        
        # Analyze feedback patterns
        low_quality_feedback = [f for f in self.feedback_history if f.explanation_quality < 0.6]
        high_quality_feedback = [f for f in self.feedback_history if f.explanation_quality > 0.8]
        
        # Collect suggested concepts
        all_suggested_concepts = []
        for feedback in self.feedback_history:
            if feedback.suggested_concepts:
                all_suggested_concepts.extend(feedback.suggested_concepts)
        
        # Count concept suggestions
        concept_suggestions = {}
        for concept in all_suggested_concepts:
            concept_suggestions[concept] = concept_suggestions.get(concept, 0) + 1
        
        # Sort by frequency
        top_suggested_concepts = sorted(
            concept_suggestions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        improvements = {
            'total_feedback_analyzed': len(self.feedback_history),
            'low_quality_count': len(low_quality_feedback),
            'high_quality_count': len(high_quality_feedback),
            'top_suggested_concepts': top_suggested_concepts,
            'avg_quality_improvement_needed': max(0, 0.8 - np.mean(self.explanation_quality_history)),
            'recommendations': []
        }
        
        # Generate recommendations
        if len(low_quality_feedback) > len(high_quality_feedback):
            improvements['recommendations'].append(
                "Consider adjusting explanation style or concept selection"
            )
        
        if top_suggested_concepts:
            improvements['recommendations'].append(
                f"Consider emphasizing concepts: {[c[0] for c in top_suggested_concepts[:3]]}"
            )
        
        return improvements
    
    def reset_interaction_history(self):
        """Reset interaction history and statistics"""
        self.interaction_history.clear()
        self.feedback_history.clear()
        self.explanation_quality_history.clear()
    
    def save_interaction_log(self, filepath: str):
        """
        Save interaction history to file
        
        Args:
            filepath: Path to save the log
        """
        log_data = {
            'interactions': [],
            'statistics': self.get_interaction_statistics()
        }
        
        for interaction in self.interaction_history:
            interaction_data = {
                'state': interaction.state.tolist() if interaction.state is not None else None,
                'action': interaction.action.item() if torch.is_tensor(interaction.action) else interaction.action,
                'concept_activations': interaction.concept_activations.tolist() if interaction.concept_activations is not None else None,
                'explanation': interaction.explanation,
                'top_concepts': interaction.top_concepts,
                'interaction_time': interaction.interaction_time,
                'success': interaction.success
            }
            
            if interaction.human_feedback:
                interaction_data['human_feedback'] = {
                    'feedback_type': interaction.human_feedback.feedback_type.value,
                    'explanation_quality': interaction.human_feedback.explanation_quality,
                    'clarity_score': interaction.human_feedback.clarity_score,
                    'usefulness_score': interaction.human_feedback.usefulness_score,
                    'suggested_concepts': interaction.human_feedback.suggested_concepts,
                    'comments': interaction.human_feedback.comments
                }
            
            log_data['interactions'].append(interaction_data)
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Interaction log saved to {filepath}")


def create_human_interaction_environment(
    agent_system: Union[ACBAgent, MultiAgentACBSystem],
    concept_vocab: ConceptVocabulary,
    interaction_mode: InteractionMode = InteractionMode.SIMULATED,
    explanation_style: ExplanationStyle = ExplanationStyle.SIMPLE,
    k_concepts: int = 3,
    human_simulator_config: Optional[Dict[str, Any]] = None,
    device: torch.device = None
) -> HumanInteractionEnvironment:
    """
    Factory function to create a configured human interaction environment
    
    Args:
        agent_system: ACB agent or multi-agent system
        concept_vocab: Concept vocabulary
        interaction_mode: Mode of interaction
        explanation_style: Style of explanations
        k_concepts: Number of concepts in explanations
        human_simulator_config: Configuration for human simulator
        device: Torch device
        
    Returns:
        Configured HumanInteractionEnvironment
    """
    # Create human simulator if needed
    human_simulator = None
    if interaction_mode == InteractionMode.SIMULATED:
        simulator_config = human_simulator_config or {}
        human_simulator = HumanSimulator(
            concept_vocab=concept_vocab,
            noise_level=simulator_config.get('noise_level', 0.1),
            preference_bias=simulator_config.get('preference_bias', {})
        )
    
    # Create explanation generator
    explanation_generator = InteractiveExplanationGenerator(
        concept_vocab=concept_vocab,
        style=explanation_style,
        k_concepts=k_concepts
    )
    
    return HumanInteractionEnvironment(
        agent_system=agent_system,
        concept_vocab=concept_vocab,
        explanation_generator=explanation_generator,
        interaction_mode=interaction_mode,
        human_simulator=human_simulator,
        k_concepts=k_concepts,
        explanation_style=explanation_style,
        device=device
    )


# Example usage and testing functions
def test_human_interaction_environment():
    """Test the human interaction environment"""
    import torch
    from ..utils.concept_utils import create_domain_concept_vocabulary
    from ..models.acb_agent import ACBAgent
    
    # Create test components
    state_dim = 10
    action_dim = 4
    concept_dim = 8
    
    concept_vocab = create_domain_concept_vocabulary(concept_dim, "navigation")
    
    agent = ACBAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        concept_dim=concept_dim,
        concept_names=concept_vocab.concept_names,
        domain="navigation"
    )
    
    # Create interaction environment
    env = create_human_interaction_environment(
        agent_system=agent,
        concept_vocab=concept_vocab,
        interaction_mode=InteractionMode.SIMULATED,
        k_concepts=3
    )
    
    # Test single interaction
    test_state = torch.randn(state_dim)
    result = env.interact_with_human(
        state=test_state,
        ground_truth_concepts=["obstacle_nearby", "goal_visible"],
        return_detailed_info=True
    )
    
    print("Test interaction result:")
    print(f"Action: {result['action']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Top concepts: {result['top_concepts']}")
    print(f"Success: {result['success']}")
    
    if result['human_feedback']:
        print(f"Feedback quality: {result['human_feedback'].explanation_quality:.2f}")
    
    # Test batch interaction
    batch_states = torch.randn(3, state_dim)
    batch_results = env.batch_interact(batch_states)
    
    print(f"\nBatch interaction: {len(batch_results)} results")
    
    # Get statistics
    stats = env.get_interaction_statistics()
    print(f"\nInteraction statistics: {stats}")
    
    return env, result, batch_results, stats


if __name__ == "__main__":
    # Run tests
    test_env, test_result, batch_results, stats = test_human_interaction_environment()
    print("Human interaction environment test completed successfully!")