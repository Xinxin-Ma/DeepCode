# ACB-Agent: Robust Explanations for Human-Neural Multi-Agent Systems
# Main package initialization

__version__ = "1.0.0"
__author__ = "ACB-Agent Implementation"
__description__ = "Explainable AI framework using concept bottlenecks for multi-agent RL"

# Import main components for easy access
try:
    from .models.concept_bottleneck import ConceptBottleneckLayer, ConceptEncoder
    from .models.policy_network import ACBPolicyNetwork, ActorCriticACB
    from .models.acb_agent import ACBAgent, MultiAgentACBSystem
    from .training.acb_trainer import ACBTrainer, create_default_trainer
    from .utils.concept_utils import ConceptVocabulary, create_domain_concept_vocabulary
    from .utils.explanation_generator import ExplanationGenerator, create_explanation_generator
    from .utils.metrics import ComprehensiveMetricsEvaluator, create_default_metrics_evaluator
    
    __all__ = [
        'ConceptBottleneckLayer', 'ConceptEncoder',
        'ACBPolicyNetwork', 'ActorCriticACB', 
        'ACBAgent', 'MultiAgentACBSystem',
        'ACBTrainer', 'create_default_trainer',
        'ConceptVocabulary', 'create_domain_concept_vocabulary',
        'ExplanationGenerator', 'create_explanation_generator',
        'ComprehensiveMetricsEvaluator', 'create_default_metrics_evaluator'
    ]
except ImportError as e:
    print(f"Warning: Some imports failed during package initialization: {e}")
    __all__ = []