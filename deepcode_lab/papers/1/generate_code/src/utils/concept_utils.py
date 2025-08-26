"""
Concept Utilities for ACB-Agent
Provides concept vocabulary management, processing utilities, and concept-related helper functions.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConceptVocabulary:
    """
    Manages concept vocabularies for different domains and provides concept name mapping.
    """
    
    def __init__(self, concept_names: List[str], domain: str = "general"):
        """
        Initialize concept vocabulary.
        
        Args:
            concept_names: List of human-interpretable concept names
            domain: Domain type for specialized vocabularies
        """
        self.concept_names = concept_names
        self.domain = domain
        self.concept_dim = len(concept_names)
        
        # Create mappings
        self.name_to_idx = {name: idx for idx, name in enumerate(concept_names)}
        self.idx_to_name = {idx: name for idx, name in enumerate(concept_names)}
        
        # Concept categories for better organization
        self.concept_categories = self._create_concept_categories()
        
    def _create_concept_categories(self) -> Dict[str, List[str]]:
        """Create concept categories based on domain and concept names."""
        categories = defaultdict(list)
        
        if self.domain == "navigation":
            for name in self.concept_names:
                if any(word in name.lower() for word in ["obstacle", "wall", "barrier"]):
                    categories["obstacles"].append(name)
                elif any(word in name.lower() for word in ["goal", "target", "destination"]):
                    categories["goals"].append(name)
                elif any(word in name.lower() for word in ["agent", "teammate", "ally"]):
                    categories["agents"].append(name)
                elif any(word in name.lower() for word in ["distance", "proximity", "near", "far"]):
                    categories["spatial"].append(name)
                else:
                    categories["general"].append(name)
                    
        elif self.domain == "resource_allocation":
            for name in self.concept_names:
                if any(word in name.lower() for word in ["resource", "supply", "material"]):
                    categories["resources"].append(name)
                elif any(word in name.lower() for word in ["demand", "need", "requirement"]):
                    categories["demands"].append(name)
                elif any(word in name.lower() for word in ["efficiency", "optimal", "cost"]):
                    categories["optimization"].append(name)
                else:
                    categories["general"].append(name)
                    
        else:  # general domain
            categories["general"] = self.concept_names.copy()
            
        return dict(categories)
    
    def get_concept_name(self, idx: int) -> str:
        """Get concept name by index."""
        return self.idx_to_name.get(idx, f"concept_{idx}")
    
    def get_concept_idx(self, name: str) -> Optional[int]:
        """Get concept index by name."""
        return self.name_to_idx.get(name)
    
    def get_category_concepts(self, category: str) -> List[str]:
        """Get all concepts in a specific category."""
        return self.concept_categories.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all available concept categories."""
        return list(self.concept_categories.keys())


def create_domain_concept_vocabulary(concept_dim: int, domain: str = "general") -> ConceptVocabulary:
    """
    Create domain-specific concept vocabulary.
    
    Args:
        concept_dim: Number of concepts
        domain: Domain type ("navigation", "resource_allocation", "general")
        
    Returns:
        ConceptVocabulary instance
    """
    if domain == "navigation":
        base_concepts = [
            "obstacle_nearby", "clear_path_ahead", "goal_visible", "teammate_close",
            "wall_left", "wall_right", "open_space", "narrow_passage",
            "goal_distance_close", "goal_distance_far", "agent_collision_risk",
            "optimal_path_available", "detour_required", "speed_safe",
            "coordination_needed", "solo_action_optimal"
        ]
    elif domain == "resource_allocation":
        base_concepts = [
            "resource_abundant", "resource_scarce", "high_demand_area", "low_demand_area",
            "efficient_allocation", "waste_detected", "priority_high", "priority_low",
            "capacity_available", "capacity_full", "cost_optimal", "cost_high",
            "supply_chain_stable", "supply_chain_disrupted", "coordination_beneficial",
            "independent_action_better"
        ]
    elif domain == "cooperative_task":
        base_concepts = [
            "task_urgent", "task_routine", "collaboration_needed", "individual_capable",
            "resource_shared", "resource_exclusive", "skill_match", "skill_mismatch",
            "communication_clear", "communication_unclear", "goal_aligned", "goal_conflict",
            "efficiency_high", "efficiency_low", "risk_low", "risk_high"
        ]
    else:  # general domain
        base_concepts = [
            "state_positive", "state_negative", "action_beneficial", "action_risky",
            "environment_stable", "environment_dynamic", "goal_achievable", "goal_difficult",
            "cooperation_helpful", "independence_better", "resource_available", "resource_limited",
            "time_sufficient", "time_pressure", "information_complete", "information_partial"
        ]
    
    # Extend or truncate to match concept_dim
    if len(base_concepts) < concept_dim:
        # Add numbered concepts if we need more
        for i in range(len(base_concepts), concept_dim):
            base_concepts.append(f"{domain}_concept_{i}")
    elif len(base_concepts) > concept_dim:
        # Truncate if we have too many
        base_concepts = base_concepts[:concept_dim]
    
    return ConceptVocabulary(base_concepts, domain)


def process_concept_activations(concept_activations: torch.Tensor, 
                              threshold: float = 0.5) -> Dict[str, Any]:
    """
    Process concept activations to extract meaningful information.
    
    Args:
        concept_activations: Tensor of concept activations [batch_size, concept_dim]
        threshold: Activation threshold for binary concept detection
        
    Returns:
        Dictionary with processed concept information
    """
    batch_size, concept_dim = concept_activations.shape
    
    # Convert to numpy for easier processing
    activations_np = concept_activations.detach().cpu().numpy()
    
    # Binary concept detection
    binary_concepts = (activations_np > threshold).astype(int)
    
    # Statistics
    mean_activations = np.mean(activations_np, axis=0)
    std_activations = np.std(activations_np, axis=0)
    max_activations = np.max(activations_np, axis=0)
    min_activations = np.min(activations_np, axis=0)
    
    # Active concept counts
    active_concept_counts = np.sum(binary_concepts, axis=1)  # per sample
    concept_activation_rates = np.mean(binary_concepts, axis=0)  # per concept
    
    return {
        "activations": activations_np,
        "binary_concepts": binary_concepts,
        "statistics": {
            "mean": mean_activations,
            "std": std_activations,
            "max": max_activations,
            "min": min_activations
        },
        "active_counts": active_concept_counts,
        "activation_rates": concept_activation_rates,
        "batch_size": batch_size,
        "concept_dim": concept_dim
    }


def extract_top_k_concepts(concept_activations: torch.Tensor,
                          concept_vocab: ConceptVocabulary,
                          k: int = 3) -> List[List[Tuple[str, float]]]:
    """
    Extract top-k activated concepts for each sample in batch.
    
    Args:
        concept_activations: Tensor [batch_size, concept_dim]
        concept_vocab: ConceptVocabulary instance
        k: Number of top concepts to extract
        
    Returns:
        List of top-k concepts for each sample [(name, score), ...]
    """
    batch_size = concept_activations.shape[0]
    activations_np = concept_activations.detach().cpu().numpy()
    
    top_k_concepts = []
    
    for i in range(batch_size):
        # Get top-k indices for this sample
        top_indices = np.argsort(activations_np[i])[-k:][::-1]  # Descending order
        
        # Convert to (name, score) tuples
        sample_concepts = []
        for idx in top_indices:
            concept_name = concept_vocab.get_concept_name(idx)
            score = float(activations_np[i, idx])
            sample_concepts.append((concept_name, score))
        
        top_k_concepts.append(sample_concepts)
    
    return top_k_concepts


def compute_concept_diversity(concept_activations: torch.Tensor) -> torch.Tensor:
    """
    Compute concept diversity using entropy measure.
    
    Args:
        concept_activations: Tensor [batch_size, concept_dim]
        
    Returns:
        Diversity loss tensor (negative entropy)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    concept_probs = concept_activations + eps
    
    # Normalize to ensure probabilities sum to 1
    concept_probs = concept_probs / torch.sum(concept_probs, dim=1, keepdim=True)
    
    # Compute entropy: H = -sum(p * log(p))
    entropy = -torch.sum(concept_probs * torch.log(concept_probs), dim=1)
    
    # Return negative entropy as diversity loss (we want to maximize entropy)
    diversity_loss = -torch.mean(entropy)
    
    return diversity_loss


def compute_concept_alignment(model_concepts: torch.Tensor,
                            human_labels: torch.Tensor,
                            threshold: float = 0.6) -> Dict[str, Any]:
    """
    Compute alignment between model concepts and human labels.
    Implementation of Section 3.3 concept alignment procedure.
    
    Args:
        model_concepts: Model concept activations [batch_size, concept_dim]
        human_labels: Human concept labels [batch_size, concept_dim]
        threshold: Alignment threshold (τ_align = 0.6)
        
    Returns:
        Dictionary with alignment metrics
    """
    # Convert to numpy for correlation computation
    model_np = model_concepts.detach().cpu().numpy()
    human_np = human_labels.detach().cpu().numpy()
    
    batch_size, concept_dim = model_np.shape
    
    # Compute correlation for each concept dimension
    correlations = []
    aligned_concepts = []
    
    for i in range(concept_dim):
        # Compute Pearson correlation coefficient
        corr_coef = np.corrcoef(model_np[:, i], human_np[:, i])[0, 1]
        
        # Handle NaN correlations (constant values)
        if np.isnan(corr_coef):
            corr_coef = 0.0
        
        correlations.append(corr_coef)
        
        # Check if aligned
        if abs(corr_coef) > threshold:
            aligned_concepts.append(i)
    
    # Compute alignment score
    alignment_score = len(aligned_concepts) / concept_dim
    
    return {
        "correlations": correlations,
        "aligned_concepts": aligned_concepts,
        "alignment_score": alignment_score,
        "threshold": threshold,
        "num_aligned": len(aligned_concepts),
        "total_concepts": concept_dim,
        "mean_correlation": np.mean(correlations),
        "std_correlation": np.std(correlations)
    }


def generate_concept_explanation(top_concepts: List[Tuple[str, float]],
                               action_name: str = "action",
                               template: str = "default") -> str:
    """
    Generate natural language explanation from top concepts.
    
    Args:
        top_concepts: List of (concept_name, score) tuples
        action_name: Name of the action being explained
        template: Explanation template type
        
    Returns:
        Natural language explanation string
    """
    if not top_concepts:
        return f"Taking {action_name} based on general state assessment."
    
    # Sort by score (highest first)
    sorted_concepts = sorted(top_concepts, key=lambda x: x[1], reverse=True)
    
    if template == "detailed":
        concept_descriptions = []
        for name, score in sorted_concepts:
            concept_descriptions.append(f"{name} (confidence: {score:.2f})")
        
        if len(concept_descriptions) == 1:
            return f"Taking {action_name} because {concept_descriptions[0]}."
        elif len(concept_descriptions) == 2:
            return f"Taking {action_name} because {concept_descriptions[0]} and {concept_descriptions[1]}."
        else:
            main_concepts = ", ".join(concept_descriptions[:-1])
            return f"Taking {action_name} because {main_concepts}, and {concept_descriptions[-1]}."
    
    else:  # default template
        concept_names = [name for name, _ in sorted_concepts]
        
        if len(concept_names) == 1:
            return f"Taking {action_name} because of {concept_names[0]}."
        elif len(concept_names) == 2:
            return f"Taking {action_name} because of {concept_names[0]} and {concept_names[1]}."
        else:
            main_concepts = ", ".join(concept_names[:-1])
            return f"Taking {action_name} because of {main_concepts}, and {concept_names[-1]}."


def save_concept_vocabulary(vocab: ConceptVocabulary, filepath: str):
    """Save concept vocabulary to file."""
    vocab_data = {
        "concept_names": vocab.concept_names,
        "domain": vocab.domain,
        "concept_categories": vocab.concept_categories,
        "name_to_idx": vocab.name_to_idx,
        "idx_to_name": vocab.idx_to_name
    }
    
    with open(filepath, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    logger.info(f"Concept vocabulary saved to {filepath}")


def load_concept_vocabulary(filepath: str) -> ConceptVocabulary:
    """Load concept vocabulary from file."""
    with open(filepath, 'r') as f:
        vocab_data = json.load(f)
    
    vocab = ConceptVocabulary(vocab_data["concept_names"], vocab_data["domain"])
    vocab.concept_categories = vocab_data["concept_categories"]
    
    logger.info(f"Concept vocabulary loaded from {filepath}")
    return vocab


def analyze_concept_usage(concept_activations_history: List[torch.Tensor],
                         concept_vocab: ConceptVocabulary) -> Dict[str, Any]:
    """
    Analyze concept usage patterns over time.
    
    Args:
        concept_activations_history: List of concept activation tensors
        concept_vocab: ConceptVocabulary instance
        
    Returns:
        Dictionary with usage analysis
    """
    if not concept_activations_history:
        return {"error": "No concept activation history provided"}
    
    # Concatenate all activations
    all_activations = torch.cat(concept_activations_history, dim=0)
    batch_size, concept_dim = all_activations.shape
    
    activations_np = all_activations.detach().cpu().numpy()
    
    # Compute usage statistics
    usage_frequency = np.mean(activations_np > 0.5, axis=0)  # How often each concept is active
    average_activation = np.mean(activations_np, axis=0)     # Average activation level
    max_activation = np.max(activations_np, axis=0)          # Maximum activation seen
    activation_variance = np.var(activations_np, axis=0)     # Activation variance
    
    # Find most and least used concepts
    most_used_idx = np.argmax(usage_frequency)
    least_used_idx = np.argmin(usage_frequency)
    
    # Category-wise analysis
    category_usage = {}
    for category, concept_names in concept_vocab.concept_categories.items():
        category_indices = [concept_vocab.get_concept_idx(name) for name in concept_names 
                          if concept_vocab.get_concept_idx(name) is not None]
        
        if category_indices:
            category_usage[category] = {
                "average_frequency": np.mean(usage_frequency[category_indices]),
                "average_activation": np.mean(average_activation[category_indices]),
                "concept_count": len(category_indices)
            }
    
    return {
        "total_samples": batch_size,
        "concept_dim": concept_dim,
        "usage_frequency": usage_frequency.tolist(),
        "average_activation": average_activation.tolist(),
        "max_activation": max_activation.tolist(),
        "activation_variance": activation_variance.tolist(),
        "most_used_concept": {
            "name": concept_vocab.get_concept_name(most_used_idx),
            "index": int(most_used_idx),
            "frequency": float(usage_frequency[most_used_idx])
        },
        "least_used_concept": {
            "name": concept_vocab.get_concept_name(least_used_idx),
            "index": int(least_used_idx),
            "frequency": float(usage_frequency[least_used_idx])
        },
        "category_usage": category_usage,
        "overall_diversity": float(np.mean(activation_variance))
    }


# Utility functions for concept processing
def normalize_concept_activations(concept_activations: torch.Tensor,
                                method: str = "sigmoid") -> torch.Tensor:
    """
    Normalize concept activations to [0, 1] range.
    
    Args:
        concept_activations: Input tensor
        method: Normalization method ("sigmoid", "minmax", "softmax")
        
    Returns:
        Normalized tensor
    """
    if method == "sigmoid":
        return torch.sigmoid(concept_activations)
    elif method == "minmax":
        min_vals = torch.min(concept_activations, dim=1, keepdim=True)[0]
        max_vals = torch.max(concept_activations, dim=1, keepdim=True)[0]
        return (concept_activations - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == "softmax":
        return torch.softmax(concept_activations, dim=1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def concept_similarity_matrix(concept_activations: torch.Tensor) -> torch.Tensor:
    """
    Compute concept similarity matrix using cosine similarity.
    
    Args:
        concept_activations: Tensor [batch_size, concept_dim]
        
    Returns:
        Similarity matrix [concept_dim, concept_dim]
    """
    # Transpose to get [concept_dim, batch_size]
    concepts_t = concept_activations.t()
    
    # Compute cosine similarity
    normalized_concepts = torch.nn.functional.normalize(concepts_t, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_concepts, normalized_concepts.t())
    
    return similarity_matrix


# Factory function for easy concept utilities creation
def create_concept_utils(concept_dim: int, domain: str = "general") -> Dict[str, Any]:
    """
    Create a complete concept utilities package for a domain.
    
    Args:
        concept_dim: Number of concepts
        domain: Domain type
        
    Returns:
        Dictionary with concept utilities
    """
    vocab = create_domain_concept_vocabulary(concept_dim, domain)
    
    return {
        "vocabulary": vocab,
        "domain": domain,
        "concept_dim": concept_dim,
        "extract_top_k": lambda activations, k=3: extract_top_k_concepts(activations, vocab, k),
        "generate_explanation": lambda top_concepts, action="action": generate_concept_explanation(top_concepts, action),
        "process_activations": process_concept_activations,
        "compute_diversity": compute_concept_diversity,
        "compute_alignment": compute_concept_alignment
    }