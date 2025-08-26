"""
Concept Bottleneck Layer Implementation for ACB-Agent

This module implements the core concept bottleneck layer that constrains agent decisions
through interpretable concept space, following the mathematical formulation:
c_t = σ(W_c * φ(s_t) + b_c) (Equation 2)

The concept bottleneck layer transforms raw state features into bounded [0,1] concept
activations that can be interpreted by humans and used for explainable decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConceptBottleneckLayer(nn.Module):
    """
    Core concept bottleneck layer that transforms state features into interpretable concepts.
    
    This layer constrains agent decisions through interpretable concept space by:
    1. Linear transformation: input_features -> concept_space
    2. Sigmoid activation for [0,1] bounded concept activations
    3. Dropout for regularization
    
    Mathematical formulation: c_t = σ(W_c * φ(s_t) + b_c)
    """
    
    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        activation_threshold: float = 0.5,
        concept_sparsity: float = 0.0
    ):
        """
        Initialize concept bottleneck layer.
        
        Args:
            input_dim: Dimension of input state features
            concept_dim: Number of concept dimensions
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
            activation_threshold: Threshold for concept activation
            concept_sparsity: L1 regularization weight for concept sparsity
        """
        super(ConceptBottleneckLayer, self).__init__()
        
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.dropout_rate = dropout_rate
        self.activation_threshold = activation_threshold
        self.concept_sparsity = concept_sparsity
        
        # Core transformation layer: φ(s_t) -> c_t
        self.concept_transform = nn.Linear(input_dim, concept_dim)
        
        # Optional batch normalization
        self.batch_norm = nn.BatchNorm1d(concept_dim) if use_batch_norm else None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.concept_transform.weight)
        nn.init.zeros_(self.concept_transform.bias)
        
        logger.info(f"Initialized ConceptBottleneckLayer: {input_dim} -> {concept_dim}")
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Transform state features to concept activations.
        
        Args:
            state_features: Input state features [batch_size, input_dim]
            
        Returns:
            concept_activations: Bounded concept activations [batch_size, concept_dim]
        """
        # Linear transformation
        concept_logits = self.concept_transform(state_features)
        
        # Optional batch normalization
        if self.batch_norm is not None:
            concept_logits = self.batch_norm(concept_logits)
        
        # Sigmoid activation for [0,1] bounded activations
        concept_activations = torch.sigmoid(concept_logits)
        
        # Optional dropout during training
        if self.dropout is not None and self.training:
            concept_activations = self.dropout(concept_activations)
        
        return concept_activations
    
    def get_concept_scores(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get interpretable concept scores for a single state.
        
        Args:
            state: Single state tensor [input_dim] or batch [batch_size, input_dim]
            
        Returns:
            concept_scores: Concept activation scores [concept_dim] or [batch_size, concept_dim]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            concept_scores = self.forward(state)
        
        return concept_scores.squeeze(0) if concept_scores.size(0) == 1 else concept_scores
    
    def get_active_concepts(
        self, 
        state: torch.Tensor, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get indices and values of active concepts above threshold.
        
        Args:
            state: Input state tensor
            threshold: Activation threshold (uses default if None)
            
        Returns:
            active_indices: Indices of active concepts
            active_values: Values of active concepts
        """
        threshold = threshold or self.activation_threshold
        concept_scores = self.get_concept_scores(state)
        
        if concept_scores.dim() == 1:
            active_mask = concept_scores > threshold
            active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
            active_values = concept_scores[active_indices]
        else:
            active_mask = concept_scores > threshold
            active_indices = torch.nonzero(active_mask, as_tuple=False)
            active_values = concept_scores[active_mask]
        
        return active_indices, active_values
    
    def compute_sparsity_loss(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity regularization loss for concept activations.
        
        Args:
            concept_activations: Concept activation tensor
            
        Returns:
            sparsity_loss: L1 regularization loss
        """
        if self.concept_sparsity <= 0:
            return torch.tensor(0.0, device=concept_activations.device)
        
        return self.concept_sparsity * torch.mean(torch.abs(concept_activations))
    
    def get_concept_statistics(self, concept_activations: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics for concept activations.
        
        Args:
            concept_activations: Concept activation tensor [batch_size, concept_dim]
            
        Returns:
            stats: Dictionary of concept statistics
        """
        with torch.no_grad():
            stats = {
                'mean_activation': torch.mean(concept_activations).item(),
                'std_activation': torch.std(concept_activations).item(),
                'max_activation': torch.max(concept_activations).item(),
                'min_activation': torch.min(concept_activations).item(),
                'active_concepts_ratio': torch.mean(
                    (concept_activations > self.activation_threshold).float()
                ).item(),
                'entropy': -torch.sum(
                    concept_activations * torch.log(concept_activations + 1e-8)
                ).item() / concept_activations.numel()
            }
        
        return stats


class ConceptEncoder(nn.Module):
    """
    Multi-layer concept encoder for complex state feature processing.
    
    This encoder processes raw state features through multiple layers before
    the concept bottleneck transformation, allowing for more complex feature
    extraction while maintaining interpretability at the concept level.
    """
    
    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        encoder_layers: List[int] = [256, 128],
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = False
    ):
        """
        Initialize concept encoder with multiple layers.
        
        Args:
            input_dim: Input state feature dimension
            concept_dim: Output concept dimension
            encoder_layers: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
            use_batch_norm: Whether to use batch normalization
        """
        super(ConceptEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.encoder_layers = encoder_layers
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(encoder_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Concept bottleneck layer
        self.concept_bottleneck = ConceptBottleneckLayer(
            input_dim=prev_dim,
            concept_dim=concept_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ConceptEncoder: {input_dim} -> {encoder_layers} -> {concept_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Encode state features to concept activations.
        
        Args:
            state_features: Input state features [batch_size, input_dim]
            
        Returns:
            concept_activations: Concept activations [batch_size, concept_dim]
        """
        # Encode through multiple layers
        encoded_features = self.encoder(state_features)
        
        # Transform to concept space
        concept_activations = self.concept_bottleneck(encoded_features)
        
        return concept_activations
    
    def get_concept_scores(self, state: torch.Tensor) -> torch.Tensor:
        """Get concept scores for interpretability."""
        return self.concept_bottleneck.get_concept_scores(self.encoder(state))
    
    def get_active_concepts(
        self, 
        state: torch.Tensor, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get active concepts above threshold."""
        encoded_state = self.encoder(state)
        return self.concept_bottleneck.get_active_concepts(encoded_state, threshold)
    
    def compute_sparsity_loss(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        return self.concept_bottleneck.compute_sparsity_loss(concept_activations)


class AdaptiveConceptBottleneck(nn.Module):
    """
    Adaptive concept bottleneck that can adjust concept importance dynamically.
    
    This variant allows for dynamic weighting of concepts based on their
    relevance to the current task or context, enabling more flexible
    concept-based decision making.
    """
    
    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        context_dim: int = 0,
        dropout_rate: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize adaptive concept bottleneck.
        
        Args:
            input_dim: Input state feature dimension
            concept_dim: Number of concepts
            context_dim: Context feature dimension (0 for no context)
            dropout_rate: Dropout rate
            temperature: Temperature for concept attention
        """
        super(AdaptiveConceptBottleneck, self).__init__()
        
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.context_dim = context_dim
        self.temperature = temperature
        
        # Base concept bottleneck
        self.concept_bottleneck = ConceptBottleneckLayer(
            input_dim=input_dim,
            concept_dim=concept_dim,
            dropout_rate=dropout_rate
        )
        
        # Concept attention mechanism
        if context_dim > 0:
            self.attention_net = nn.Sequential(
                nn.Linear(context_dim, concept_dim),
                nn.Tanh(),
                nn.Linear(concept_dim, concept_dim)
            )
        
        logger.info(f"Initialized AdaptiveConceptBottleneck: {input_dim} -> {concept_dim}")
    
    def forward(
        self, 
        state_features: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional context-based adaptation.
        
        Args:
            state_features: Input state features
            context: Optional context for concept adaptation
            
        Returns:
            adapted_concepts: Context-adapted concept activations
        """
        # Base concept activations
        base_concepts = self.concept_bottleneck(state_features)
        
        # Apply context-based adaptation if available
        if context is not None and self.context_dim > 0:
            attention_weights = torch.softmax(
                self.attention_net(context) / self.temperature, dim=-1
            )
            adapted_concepts = base_concepts * attention_weights
        else:
            adapted_concepts = base_concepts
        
        return adapted_concepts


def create_concept_bottleneck(
    input_dim: int,
    concept_dim: int,
    bottleneck_type: str = 'simple',
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of concept bottlenecks.
    
    Args:
        input_dim: Input feature dimension
        concept_dim: Concept dimension
        bottleneck_type: Type of bottleneck ('simple', 'encoder', 'adaptive')
        **kwargs: Additional arguments for specific bottleneck types
        
    Returns:
        concept_bottleneck: Initialized concept bottleneck module
    """
    if bottleneck_type == 'simple':
        return ConceptBottleneckLayer(input_dim, concept_dim, **kwargs)
    elif bottleneck_type == 'encoder':
        return ConceptEncoder(input_dim, concept_dim, **kwargs)
    elif bottleneck_type == 'adaptive':
        return AdaptiveConceptBottleneck(input_dim, concept_dim, **kwargs)
    else:
        raise ValueError(f"Unknown bottleneck type: {bottleneck_type}")


def create_default_concept_names(concept_dim: int, domain: str = 'general') -> List[str]:
    """
    Generate domain-specific concept names for interpretability.
    
    Args:
        concept_dim: Number of concepts to generate names for
        domain: Domain type ('general', 'navigation', 'resource', 'coordination')
        
    Returns:
        concept_names: List of interpretable concept names
    """
    if domain == 'navigation':
        base_concepts = [
            'obstacle_nearby', 'target_visible', 'path_clear', 'collision_risk',
            'goal_direction', 'agent_proximity', 'boundary_close', 'speed_optimal',
            'turning_needed', 'straight_path', 'crowded_area', 'open_space',
            'target_distance', 'movement_efficiency', 'safety_margin', 'exploration_value'
        ]
    elif domain == 'resource':
        base_concepts = [
            'resource_available', 'high_value_item', 'competition_present', 'allocation_fair',
            'demand_high', 'supply_low', 'efficiency_optimal', 'waste_minimal',
            'priority_urgent', 'capacity_full', 'distribution_needed', 'stockpile_low',
            'sharing_beneficial', 'hoarding_detected', 'cooperation_needed', 'conflict_risk'
        ]
    elif domain == 'coordination':
        base_concepts = [
            'team_aligned', 'communication_clear', 'role_defined', 'task_assigned',
            'synchronization_needed', 'leader_present', 'consensus_reached', 'conflict_detected',
            'cooperation_beneficial', 'individual_action', 'group_decision', 'timing_critical',
            'information_shared', 'strategy_agreed', 'execution_ready', 'adaptation_needed'
        ]
    else:  # general domain
        base_concepts = [
            'high_reward', 'low_risk', 'opportunity_present', 'threat_detected',
            'action_beneficial', 'state_favorable', 'goal_achievable', 'constraint_active',
            'information_available', 'uncertainty_high', 'decision_critical', 'time_pressure',
            'resource_sufficient', 'capability_adequate', 'context_relevant', 'outcome_positive'
        ]
    
    # Extend or truncate to match concept_dim
    if concept_dim <= len(base_concepts):
        return base_concepts[:concept_dim]
    else:
        # Generate additional generic concepts
        concept_names = base_concepts.copy()
        for i in range(len(base_concepts), concept_dim):
            concept_names.append(f"{domain}_concept_{i+1}")
        return concept_names


# Export main classes and functions
__all__ = [
    'ConceptBottleneckLayer',
    'ConceptEncoder', 
    'AdaptiveConceptBottleneck',
    'create_concept_bottleneck',
    'create_default_concept_names'
]