"""
ACB-Agent: Approximate Concept Bottleneck Agent for Explainable Multi-Agent RL

This module implements the main ACB-Agent class that integrates concept bottleneck
layers with policy networks to provide explainable reinforcement learning agents
for human-AI collaboration scenarios.

Key Features:
- Integrates ConceptBottleneckLayer with ActorCriticACB policy network
- Provides explainable action selection through concept activations
- Supports multi-agent coordination with interpretable explanations
- Implements the core ACB-Agent architecture from the paper

Author: Implementation based on "Robust Explanations for Human-Neural Multi-Agent Systems via Approximate Concept Bottlenecks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import our implemented components
from .concept_bottleneck import ConceptBottleneckLayer, ConceptEncoder, create_default_concept_names
from .policy_network import ACBPolicyNetwork, ValueNetwork, ActorCriticACB


class ACBAgent(nn.Module):
    """
    Main ACB-Agent class that combines concept bottleneck layers with policy networks
    to create an explainable reinforcement learning agent.
    
    The agent follows the architecture: State -> ConceptBottleneck -> Policy -> Action
    This ensures all decisions are made through interpretable concept activations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        concept_dim: int = 64,
        hidden_dim: int = 128,
        concept_names: Optional[List[str]] = None,
        domain: str = "multi_agent",
        dropout_rate: float = 0.1,
        use_concept_encoder: bool = False,
        encoder_layers: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize ACB-Agent with concept bottleneck and policy components.
        
        Args:
            state_dim: Dimension of input state features
            action_dim: Number of possible actions
            concept_dim: Dimension of concept space (number of concepts)
            hidden_dim: Hidden layer dimension for policy network
            concept_names: List of human-interpretable concept names
            domain: Domain for default concept name generation
            dropout_rate: Dropout rate for regularization
            use_concept_encoder: Whether to use multi-layer concept encoder
            encoder_layers: Number of layers in concept encoder
            device: Device to run the model on
        """
        super(ACBAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.domain = domain
        
        # Generate concept names if not provided
        if concept_names is None:
            concept_names = create_default_concept_names(concept_dim, domain)
        self.concept_names = concept_names
        
        # Initialize concept bottleneck layer
        if use_concept_encoder:
            self.concept_bottleneck = ConceptEncoder(
                input_dim=state_dim,
                concept_dim=concept_dim,
                hidden_dim=hidden_dim,
                num_layers=encoder_layers,
                dropout_rate=dropout_rate,
                concept_names=concept_names
            )
        else:
            self.concept_bottleneck = ConceptBottleneckLayer(
                input_dim=state_dim,
                concept_dim=concept_dim,
                dropout_rate=dropout_rate,
                concept_names=concept_names
            )
        
        # Initialize actor-critic policy network
        self.actor_critic = ActorCriticACB(
            concept_dim=concept_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # Move to device
        self.to(device)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete ACB-Agent architecture.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            action_probs: Action probability distribution [batch_size, action_dim]
            concept_activations: Concept activations [batch_size, concept_dim]
        """
        # Transform state to concept space
        concept_activations = self.concept_bottleneck(state)
        
        # Get action probabilities from concepts
        action_probs = self.actor_critic.actor(concept_activations)
        
        return action_probs, concept_activations
    
    def act(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False,
        return_explanation: bool = True,
        explanation_k: int = 3
    ) -> Dict[str, Union[torch.Tensor, List[str], str]]:
        """
        Select action with optional explanation generation.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            deterministic: Whether to use deterministic action selection
            return_explanation: Whether to generate explanation
            explanation_k: Number of top concepts to include in explanation
            
        Returns:
            Dictionary containing:
            - actions: Selected actions [batch_size]
            - log_probs: Log probabilities of selected actions [batch_size]
            - values: State values [batch_size]
            - concept_activations: Concept activations [batch_size, concept_dim]
            - explanation: Human-readable explanation (if requested)
            - top_concepts: Top-k activated concepts (if requested)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Get concept activations
            concept_activations = self.concept_bottleneck(state)
            
            # Get actions, log_probs, and values from actor-critic
            actions, log_probs, values = self.actor_critic.act(
                concept_activations, deterministic=deterministic
            )
            
            result = {
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
                'concept_activations': concept_activations
            }
            
            # Generate explanation if requested
            if return_explanation:
                explanation_data = self.generate_explanation(
                    concept_activations, k=explanation_k
                )
                result.update(explanation_data)
            
            return result
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for policy gradient updates.
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size]
            
        Returns:
            Dictionary containing:
            - log_probs: Log probabilities [batch_size]
            - values: State values [batch_size]
            - entropy: Policy entropy [batch_size]
            - concept_activations: Concept activations [batch_size, concept_dim]
        """
        # Get concept activations
        concept_activations = self.concept_bottleneck(states)
        
        # Evaluate actions through actor-critic
        log_probs, values, entropy = self.actor_critic.evaluate_actions(
            concept_activations, actions
        )
        
        return {
            'log_probs': log_probs,
            'values': values,
            'entropy': entropy,
            'concept_activations': concept_activations
        }
    
    def generate_explanation(
        self, 
        concept_activations: torch.Tensor, 
        k: int = 3
    ) -> Dict[str, Union[List[Tuple[str, float]], str]]:
        """
        Generate human-interpretable explanations from concept activations.
        
        Args:
            concept_activations: Concept activation tensor [batch_size, concept_dim]
            k: Number of top concepts to include in explanation
            
        Returns:
            Dictionary containing:
            - top_concepts: List of (concept_name, activation_value) tuples
            - explanation: Natural language explanation string
        """
        # Handle batch dimension - use first sample for explanation
        if concept_activations.dim() > 1:
            concepts = concept_activations[0]  # [concept_dim]
        else:
            concepts = concept_activations
        
        # Get top-k activated concepts
        top_k_values, top_k_indices = torch.topk(concepts, k)
        
        top_concepts = []
        for i in range(k):
            concept_idx = top_k_indices[i].item()
            concept_value = top_k_values[i].item()
            concept_name = self.concept_names[concept_idx]
            top_concepts.append((concept_name, concept_value))
        
        # Generate natural language explanation
        if len(top_concepts) > 0:
            concept_list = [f"{name} ({value:.3f})" for name, value in top_concepts]
            explanation = f"Action selected based on: {', '.join(concept_list)}"
        else:
            explanation = "Action selected based on low concept activations"
        
        return {
            'top_concepts': top_concepts,
            'explanation': explanation
        }
    
    def get_concept_scores(self, state: torch.Tensor) -> Dict[str, float]:
        """
        Get interpretable concept scores for a given state.
        
        Args:
            state: Input state tensor [state_dim] or [1, state_dim]
            
        Returns:
            Dictionary mapping concept names to activation scores
        """
        return self.concept_bottleneck.get_concept_scores(state)
    
    def get_top_k_concepts(self, state: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k activated concepts for a given state.
        
        Args:
            state: Input state tensor [state_dim] or [1, state_dim]
            k: Number of top concepts to return
            
        Returns:
            List of (concept_name, activation_value) tuples sorted by activation
        """
        return self.concept_bottleneck.get_top_k_concepts(state, k)
    
    def compute_concept_diversity_loss(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute concept diversity regularization loss.
        
        Args:
            concept_activations: Concept activation tensor [batch_size, concept_dim]
            
        Returns:
            Diversity loss (negative entropy)
        """
        return self.concept_bottleneck.compute_concept_diversity(concept_activations)
    
    def get_parameters_by_component(self) -> Dict[str, List[torch.nn.Parameter]]:
        """
        Get parameters grouped by component for differential learning rates.
        
        Returns:
            Dictionary with parameter groups:
            - concept_bottleneck: Parameters of concept bottleneck layer
            - actor: Parameters of actor network
            - critic: Parameters of critic network
        """
        return {
            'concept_bottleneck': list(self.concept_bottleneck.parameters()),
            'actor': list(self.actor_critic.actor.parameters()),
            'critic': list(self.actor_critic.critic.parameters())
        }
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[Dict] = None):
        """
        Save model checkpoint with additional training information.
        
        Args:
            filepath: Path to save checkpoint
            additional_info: Additional information to save (e.g., training stats)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'concept_dim': self.concept_dim,
                'hidden_dim': self.hidden_dim,
                'concept_names': self.concept_names,
                'domain': self.domain
            }
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = "cpu") -> 'ACBAgent':
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded ACBAgent instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        # Create agent with saved configuration
        agent = cls(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            concept_dim=config['concept_dim'],
            hidden_dim=config['hidden_dim'],
            concept_names=config['concept_names'],
            domain=config['domain'],
            device=device
        )
        
        # Load state dict
        agent.load_state_dict(checkpoint['model_state_dict'])
        
        return agent
    
    def set_training_mode(self, training: bool = True):
        """
        Set training mode for all components.
        
        Args:
            training: Whether to set training mode (True) or evaluation mode (False)
        """
        self.train(training)
        self.concept_bottleneck.train(training)
        self.actor_critic.train(training)
    
    def get_model_info(self) -> Dict[str, Union[int, str, List[str]]]:
        """
        Get model architecture information.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'concept_dim': self.concept_dim,
            'hidden_dim': self.hidden_dim,
            'domain': self.domain,
            'concept_names': self.concept_names,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }


class MultiAgentACBSystem(nn.Module):
    """
    Multi-agent system with multiple ACB-Agents for coordination tasks.
    
    This class manages multiple ACB agents and provides coordination mechanisms
    for multi-agent reinforcement learning scenarios.
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        concept_dim: int = 64,
        hidden_dim: int = 128,
        shared_concepts: bool = True,
        concept_names: Optional[List[str]] = None,
        domain: str = "multi_agent",
        device: str = "cpu"
    ):
        """
        Initialize multi-agent ACB system.
        
        Args:
            num_agents: Number of agents in the system
            state_dim: Dimension of state space for each agent
            action_dim: Dimension of action space for each agent
            concept_dim: Dimension of concept space
            hidden_dim: Hidden layer dimension
            shared_concepts: Whether agents share concept vocabulary
            concept_names: Shared concept names (if shared_concepts=True)
            domain: Domain for concept generation
            device: Device to run models on
        """
        super(MultiAgentACBSystem, self).__init__()
        
        self.num_agents = num_agents
        self.shared_concepts = shared_concepts
        self.device = device
        
        # Create agents
        self.agents = nn.ModuleList()
        
        for i in range(num_agents):
            agent_concept_names = concept_names if shared_concepts else None
            agent_domain = f"{domain}_agent_{i}" if not shared_concepts else domain
            
            agent = ACBAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                concept_dim=concept_dim,
                hidden_dim=hidden_dim,
                concept_names=agent_concept_names,
                domain=agent_domain,
                device=device
            )
            self.agents.append(agent)
        
        self.to(device)
    
    def act(
        self, 
        states: List[torch.Tensor], 
        deterministic: bool = False,
        return_explanations: bool = True
    ) -> List[Dict]:
        """
        Get actions for all agents.
        
        Args:
            states: List of state tensors for each agent
            deterministic: Whether to use deterministic action selection
            return_explanations: Whether to generate explanations
            
        Returns:
            List of action dictionaries for each agent
        """
        results = []
        
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            result = agent.act(
                state, 
                deterministic=deterministic,
                return_explanation=return_explanations
            )
            result['agent_id'] = i
            results.append(result)
        
        return results
    
    def get_concept_alignment(self) -> Dict[str, float]:
        """
        Compute concept alignment between agents (if using shared concepts).
        
        Returns:
            Dictionary with alignment metrics
        """
        if not self.shared_concepts or self.num_agents < 2:
            return {'alignment_score': 1.0, 'message': 'Single agent or non-shared concepts'}
        
        # Compare concept activations across agents for same states
        # This is a simplified alignment measure
        alignment_scores = []
        
        # Generate random states for alignment testing
        test_states = torch.randn(10, self.agents[0].state_dim, device=self.device)
        
        for state in test_states:
            state_batch = state.unsqueeze(0)  # Add batch dimension
            
            # Get concept activations from all agents
            activations = []
            for agent in self.agents:
                concepts = agent.concept_bottleneck(state_batch)
                activations.append(concepts.squeeze(0))  # Remove batch dimension
            
            # Compute pairwise correlations
            for i in range(len(activations)):
                for j in range(i + 1, len(activations)):
                    corr = torch.corrcoef(torch.stack([activations[i], activations[j]]))[0, 1]
                    if not torch.isnan(corr):
                        alignment_scores.append(corr.item())
        
        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        
        return {
            'alignment_score': avg_alignment,
            'num_comparisons': len(alignment_scores),
            'message': f'Average concept alignment across {self.num_agents} agents'
        }
    
    def get_system_info(self) -> Dict:
        """
        Get information about the multi-agent system.
        
        Returns:
            Dictionary with system information
        """
        return {
            'num_agents': self.num_agents,
            'shared_concepts': self.shared_concepts,
            'agent_configs': [agent.get_model_info() for agent in self.agents],
            'total_parameters': sum(sum(p.numel() for p in agent.parameters()) for agent in self.agents)
        }


# Utility functions for ACB-Agent usage

def create_acb_agent_from_config(config: Dict) -> ACBAgent:
    """
    Create ACB-Agent from configuration dictionary.
    
    Args:
        config: Configuration dictionary with agent parameters
        
    Returns:
        Configured ACBAgent instance
    """
    return ACBAgent(**config)


def batch_explain_actions(
    agent: ACBAgent, 
    states: torch.Tensor, 
    actions: torch.Tensor,
    k: int = 3
) -> List[Dict]:
    """
    Generate explanations for a batch of state-action pairs.
    
    Args:
        agent: ACBAgent instance
        states: Batch of states [batch_size, state_dim]
        actions: Batch of actions [batch_size]
        k: Number of top concepts per explanation
        
    Returns:
        List of explanation dictionaries
    """
    explanations = []
    
    with torch.no_grad():
        concept_activations = agent.concept_bottleneck(states)
        
        for i in range(states.size(0)):
            explanation = agent.generate_explanation(
                concept_activations[i:i+1], k=k
            )
            explanation['state_idx'] = i
            explanation['action'] = actions[i].item()
            explanations.append(explanation)
    
    return explanations


if __name__ == "__main__":
    # Example usage and testing
    print("ACB-Agent Implementation Test")
    print("=" * 50)
    
    # Test single agent
    agent = ACBAgent(
        state_dim=10,
        action_dim=4,
        concept_dim=8,
        hidden_dim=64,
        domain="test"
    )
    
    print(f"Agent created: {agent.get_model_info()}")
    
    # Test forward pass
    test_state = torch.randn(1, 10)
    result = agent.act(test_state, return_explanation=True)
    
    print(f"Action: {result['actions'].item()}")
    print(f"Explanation: {result['explanation']}")
    print(f"Top concepts: {result['top_concepts']}")
    
    # Test multi-agent system
    multi_system = MultiAgentACBSystem(
        num_agents=3,
        state_dim=10,
        action_dim=4,
        concept_dim=8
    )
    
    print(f"\nMulti-agent system: {multi_system.get_system_info()}")
    
    # Test multi-agent actions
    test_states = [torch.randn(1, 10) for _ in range(3)]
    multi_results = multi_system.act(test_states)
    
    for i, result in enumerate(multi_results):
        print(f"Agent {i}: Action {result['actions'].item()}, {result['explanation']}")
    
    print("\nACB-Agent implementation completed successfully!")