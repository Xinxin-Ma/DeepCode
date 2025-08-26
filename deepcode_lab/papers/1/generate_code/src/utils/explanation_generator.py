"""
Explanation Generator for ACB-Agent
Generates human-interpretable explanations from concept activations
Based on Section 4.2 of the paper: Human-AI Interaction Protocol
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Import from existing concept utilities
try:
    from .concept_utils import ConceptVocabulary, extract_top_k_concepts
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.concept_utils import ConceptVocabulary, extract_top_k_concepts


class ExplanationStyle(Enum):
    """Different styles of explanation generation"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation"""
    k_concepts: int = 3  # Number of top concepts to include
    style: ExplanationStyle = ExplanationStyle.SIMPLE
    include_confidence: bool = True
    include_alternatives: bool = False
    confidence_threshold: float = 0.1
    diversity_penalty: float = 0.1


class ExplanationGenerator:
    """
    Core explanation generator class
    Implements Algorithm 2 from Section 4.2: Human-AI Interaction Protocol
    """
    
    def __init__(self, 
                 concept_vocab: ConceptVocabulary,
                 config: Optional[ExplanationConfig] = None):
        """
        Initialize explanation generator
        
        Args:
            concept_vocab: Vocabulary for concept names and mappings
            config: Configuration for explanation generation
        """
        self.concept_vocab = concept_vocab
        self.config = config or ExplanationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Templates for different explanation styles
        self.templates = {
            ExplanationStyle.SIMPLE: {
                'action': "I chose action '{action}' because of {concepts}.",
                'no_action': "Based on {concepts}, I recommend {action}.",
                'confidence': " (confidence: {confidence:.2f})"
            },
            ExplanationStyle.DETAILED: {
                'action': "My decision to take action '{action}' was primarily influenced by the following factors: {concepts}. These concepts were most strongly activated in the current situation.",
                'no_action': "Given the current situation, I analyzed {concepts} and determined that {action} would be the most appropriate response.",
                'confidence': " I am {confidence:.1%} confident in this assessment."
            },
            ExplanationStyle.CAUSAL: {
                'action': "Action '{action}' was selected because {primary_concept} led to {secondary_concepts}, resulting in this decision.",
                'no_action': "The causal chain {primary_concept} → {secondary_concepts} suggests {action} as the optimal choice.",
                'confidence': " (causal strength: {confidence:.2f})"
            },
            ExplanationStyle.COMPARATIVE: {
                'action': "I chose '{action}' over alternatives because {concepts} were more strongly activated than competing factors.",
                'no_action': "Comparing available options, {concepts} favor {action} over other possibilities.",
                'confidence': " (relative confidence: {confidence:.2f})"
            }
        }
    
    def generate_explanation(self,
                           concept_activations: torch.Tensor,
                           action: Optional[str] = None,
                           action_probs: Optional[torch.Tensor] = None,
                           state_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate human-interpretable explanation from concept activations
        
        Args:
            concept_activations: Tensor of concept activations [concept_dim]
            action: Name of the selected action (optional)
            action_probs: Action probabilities for confidence estimation
            state_info: Additional state information for context
            
        Returns:
            Dictionary containing explanation components
        """
        # Extract top-k concepts
        if concept_activations.dim() > 1:
            concept_activations = concept_activations.squeeze(0)
            
        top_concepts = self._extract_top_concepts(concept_activations)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            top_concepts, action, action_probs
        )
        
        # Compute explanation quality metrics
        quality_metrics = self._compute_explanation_quality(
            concept_activations, top_concepts
        )
        
        # Generate alternative explanations if requested
        alternatives = []
        if self.config.include_alternatives:
            alternatives = self._generate_alternative_explanations(
                concept_activations, action, top_concepts
            )
        
        return {
            'explanation': explanation_text,
            'top_concepts': top_concepts,
            'quality_metrics': quality_metrics,
            'alternatives': alternatives,
            'config': self.config,
            'raw_activations': concept_activations.detach().cpu().numpy()
        }
    
    def _extract_top_concepts(self, concept_activations: torch.Tensor) -> List[Tuple[str, float]]:
        """Extract top-k activated concepts with names and scores"""
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(
            concept_activations, 
            min(self.config.k_concepts, len(concept_activations))
        )
        
        # Filter by confidence threshold
        mask = top_k_values >= self.config.confidence_threshold
        filtered_values = top_k_values[mask]
        filtered_indices = top_k_indices[mask]
        
        # Map to concept names
        top_concepts = []
        for idx, value in zip(filtered_indices, filtered_values):
            concept_name = self.concept_vocab.get_concept_name(idx.item())
            top_concepts.append((concept_name, value.item()))
        
        return top_concepts
    
    def _generate_explanation_text(self,
                                 top_concepts: List[Tuple[str, float]],
                                 action: Optional[str],
                                 action_probs: Optional[torch.Tensor]) -> str:
        """Generate natural language explanation text"""
        if not top_concepts:
            return "No significant concepts were activated for this decision."
        
        # Format concepts for text
        concept_text = self._format_concepts_for_text(top_concepts)
        
        # Get appropriate template
        template_dict = self.templates[self.config.style]
        
        # Choose template based on whether action is provided
        if action:
            template = template_dict['action']
            explanation = template.format(action=action, concepts=concept_text)
        else:
            template = template_dict['no_action']
            explanation = template.format(action="taking appropriate action", concepts=concept_text)
        
        # Add confidence if available and requested
        if self.config.include_confidence and action_probs is not None:
            confidence = self._compute_confidence(action_probs)
            confidence_text = template_dict['confidence'].format(confidence=confidence)
            explanation += confidence_text
        
        return explanation
    
    def _format_concepts_for_text(self, top_concepts: List[Tuple[str, float]]) -> str:
        """Format concept list for natural language"""
        if not top_concepts:
            return "no clear factors"
        
        if len(top_concepts) == 1:
            name, score = top_concepts[0]
            if self.config.include_confidence:
                return f"{name} (activation: {score:.2f})"
            return name
        
        # Multiple concepts
        concept_strs = []
        for name, score in top_concepts:
            if self.config.include_confidence:
                concept_strs.append(f"{name} ({score:.2f})")
            else:
                concept_strs.append(name)
        
        if len(concept_strs) == 2:
            return f"{concept_strs[0]} and {concept_strs[1]}"
        else:
            return f"{', '.join(concept_strs[:-1])}, and {concept_strs[-1]}"
    
    def _compute_confidence(self, action_probs: torch.Tensor) -> float:
        """Compute confidence score from action probabilities"""
        if action_probs is None:
            return 0.0
        
        # Use max probability as confidence
        max_prob = torch.max(action_probs).item()
        return max_prob
    
    def _compute_explanation_quality(self,
                                   concept_activations: torch.Tensor,
                                   top_concepts: List[Tuple[str, float]]) -> Dict[str, float]:
        """Compute quality metrics for the explanation"""
        if not top_concepts:
            return {
                'coverage': 0.0,
                'diversity': 0.0,
                'clarity': 0.0,
                'overall_quality': 0.0
            }
        
        # Coverage: how much of total activation is explained
        total_activation = torch.sum(concept_activations).item()
        explained_activation = sum(score for _, score in top_concepts)
        coverage = explained_activation / max(total_activation, 1e-8)
        
        # Diversity: entropy of selected concepts
        if len(top_concepts) > 1:
            scores = torch.tensor([score for _, score in top_concepts])
            probs = scores / torch.sum(scores)
            diversity = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        else:
            diversity = 0.0
        
        # Clarity: inverse of concept score variance (more uniform = clearer)
        if len(top_concepts) > 1:
            scores = [score for _, score in top_concepts]
            clarity = 1.0 / (np.var(scores) + 1e-8)
        else:
            clarity = 1.0
        
        # Overall quality (weighted combination)
        overall_quality = 0.4 * coverage + 0.3 * diversity + 0.3 * min(clarity, 1.0)
        
        return {
            'coverage': coverage,
            'diversity': diversity,
            'clarity': min(clarity, 1.0),
            'overall_quality': overall_quality
        }
    
    def _generate_alternative_explanations(self,
                                         concept_activations: torch.Tensor,
                                         action: Optional[str],
                                         primary_concepts: List[Tuple[str, float]]) -> List[str]:
        """Generate alternative explanation perspectives"""
        alternatives = []
        
        # Alternative 1: Focus on different top concepts
        if len(concept_activations) > self.config.k_concepts:
            # Get next set of concepts
            sorted_indices = torch.argsort(concept_activations, descending=True)
            alt_indices = sorted_indices[self.config.k_concepts:self.config.k_concepts*2]
            
            alt_concepts = []
            for idx in alt_indices:
                if concept_activations[idx] >= self.config.confidence_threshold:
                    concept_name = self.concept_vocab.get_concept_name(idx.item())
                    alt_concepts.append((concept_name, concept_activations[idx].item()))
            
            if alt_concepts:
                alt_text = self._format_concepts_for_text(alt_concepts)
                alternatives.append(f"Alternatively, this decision could be explained by {alt_text}.")
        
        # Alternative 2: Negative explanation (what didn't influence)
        low_activation_concepts = []
        for i, activation in enumerate(concept_activations):
            if activation < 0.1:  # Low activation threshold
                concept_name = self.concept_vocab.get_concept_name(i)
                low_activation_concepts.append(concept_name)
        
        if low_activation_concepts and len(low_activation_concepts) <= 3:
            neg_text = ', '.join(low_activation_concepts[:3])
            alternatives.append(f"This decision was not influenced by {neg_text}.")
        
        return alternatives


class BatchExplanationGenerator:
    """Generator for batch explanation processing"""
    
    def __init__(self, base_generator: ExplanationGenerator):
        self.base_generator = base_generator
    
    def generate_batch_explanations(self,
                                  concept_activations_batch: torch.Tensor,
                                  actions_batch: Optional[List[str]] = None,
                                  action_probs_batch: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of concept activations
        
        Args:
            concept_activations_batch: Batch of concept activations [batch_size, concept_dim]
            actions_batch: List of action names for each sample
            action_probs_batch: Batch of action probabilities [batch_size, action_dim]
            
        Returns:
            List of explanation dictionaries
        """
        batch_size = concept_activations_batch.shape[0]
        explanations = []
        
        for i in range(batch_size):
            concept_activations = concept_activations_batch[i]
            action = actions_batch[i] if actions_batch else None
            action_probs = action_probs_batch[i] if action_probs_batch is not None else None
            
            explanation = self.base_generator.generate_explanation(
                concept_activations, action, action_probs
            )
            explanations.append(explanation)
        
        return explanations


class InteractiveExplanationGenerator:
    """
    Interactive explanation generator for human-AI interaction
    Implements the full Algorithm 2 from Section 4.2
    """
    
    def __init__(self, base_generator: ExplanationGenerator):
        self.base_generator = base_generator
        self.interaction_history = []
    
    def interact_with_human(self,
                          state: torch.Tensor,
                          concept_activations: torch.Tensor,
                          action: str,
                          action_probs: torch.Tensor) -> Dict[str, Any]:
        """
        Full human-AI interaction protocol from Algorithm 2
        
        Args:
            state: Current state tensor
            concept_activations: Concept activations from CB layer
            action: Selected action
            action_probs: Action probability distribution
            
        Returns:
            Interaction result with explanation and feedback interface
        """
        # Step 1: Generate explanation
        explanation_result = self.base_generator.generate_explanation(
            concept_activations, action, action_probs
        )
        
        # Step 2: Create interaction package
        interaction = {
            'timestamp': torch.tensor(len(self.interaction_history)),
            'state_summary': self._summarize_state(state),
            'action': action,
            'explanation': explanation_result['explanation'],
            'top_concepts': explanation_result['top_concepts'],
            'confidence': self._extract_confidence(action_probs),
            'alternatives': explanation_result.get('alternatives', []),
            'feedback_interface': self._create_feedback_interface(explanation_result)
        }
        
        # Step 3: Store interaction
        self.interaction_history.append(interaction)
        
        return interaction
    
    def _summarize_state(self, state: torch.Tensor) -> str:
        """Create human-readable state summary"""
        # Simple state summarization - can be extended
        state_norm = torch.norm(state).item()
        state_mean = torch.mean(state).item()
        return f"State complexity: {state_norm:.2f}, average value: {state_mean:.2f}"
    
    def _extract_confidence(self, action_probs: torch.Tensor) -> float:
        """Extract confidence from action probabilities"""
        return torch.max(action_probs).item()
    
    def _create_feedback_interface(self, explanation_result: Dict) -> Dict[str, Any]:
        """Create interface for collecting human feedback"""
        return {
            'explanation_quality_scale': list(range(1, 6)),  # 1-5 rating
            'concept_relevance_questions': [
                f"How relevant is '{concept}' to this decision?"
                for concept, _ in explanation_result['top_concepts']
            ],
            'improvement_suggestions': {
                'more_concepts': 'Would you like to see more contributing factors?',
                'different_style': 'Would a different explanation style be clearer?',
                'more_detail': 'Would you like more detailed reasoning?'
            }
        }
    
    def process_human_feedback(self, feedback: Dict[str, Any]) -> None:
        """Process feedback from human and update explanation generation"""
        if not self.interaction_history:
            return
        
        # Update the last interaction with feedback
        last_interaction = self.interaction_history[-1]
        last_interaction['human_feedback'] = feedback
        
        # Simple adaptation based on feedback
        if feedback.get('explanation_quality', 0) < 3:
            # Low quality rating - suggest more detailed style
            self.base_generator.config.style = ExplanationStyle.DETAILED
            self.base_generator.config.k_concepts = min(
                self.base_generator.config.k_concepts + 1, 5
            )
        
        if feedback.get('request_more_concepts', False):
            self.base_generator.config.k_concepts = min(
                self.base_generator.config.k_concepts + 1, 5
            )


def create_explanation_generator(concept_vocab: ConceptVocabulary,
                               style: ExplanationStyle = ExplanationStyle.SIMPLE,
                               k_concepts: int = 3) -> ExplanationGenerator:
    """
    Factory function to create explanation generator with common configurations
    
    Args:
        concept_vocab: Concept vocabulary
        style: Explanation style
        k_concepts: Number of concepts to include
        
    Returns:
        Configured ExplanationGenerator
    """
    config = ExplanationConfig(
        k_concepts=k_concepts,
        style=style,
        include_confidence=True,
        include_alternatives=False
    )
    
    return ExplanationGenerator(concept_vocab, config)


def create_interactive_explanation_system(concept_vocab: ConceptVocabulary) -> InteractiveExplanationGenerator:
    """
    Create complete interactive explanation system
    
    Args:
        concept_vocab: Concept vocabulary
        
    Returns:
        Interactive explanation generator
    """
    base_generator = create_explanation_generator(
        concept_vocab, 
        style=ExplanationStyle.DETAILED,
        k_concepts=3
    )
    
    return InteractiveExplanationGenerator(base_generator)


# Example usage and testing functions
def test_explanation_generator():
    """Test function for explanation generator"""
    # Create mock concept vocabulary
    concept_names = [
        "obstacle_nearby", "goal_visible", "teammate_close", 
        "resource_available", "threat_detected", "path_clear"
    ]
    
    from .concept_utils import ConceptVocabulary
    vocab = ConceptVocabulary(concept_names, "navigation")
    
    # Create generator
    generator = create_explanation_generator(vocab, ExplanationStyle.DETAILED)
    
    # Test with mock concept activations
    concept_activations = torch.tensor([0.8, 0.3, 0.9, 0.1, 0.2, 0.7])
    action_probs = torch.tensor([0.1, 0.7, 0.2])
    
    # Generate explanation
    result = generator.generate_explanation(
        concept_activations, 
        action="move_forward",
        action_probs=action_probs
    )
    
    print("Generated Explanation:")
    print(result['explanation'])
    print(f"Top concepts: {result['top_concepts']}")
    print(f"Quality metrics: {result['quality_metrics']}")
    
    return result


if __name__ == "__main__":
    # Run test
    test_result = test_explanation_generator()