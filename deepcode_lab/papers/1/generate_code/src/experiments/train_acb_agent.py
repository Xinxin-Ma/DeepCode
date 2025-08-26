#!/usr/bin/env python3
"""
Main training script for ACB-Agent (Approximate Concept Bottleneck Agent)
Implements the complete training pipeline from the paper:
"Robust Explanations for Human-Neural Multi-Agent Systems via Approximate Concept Bottlenecks"

This script provides:
1. Complete training pipeline with configurable environments
2. Multi-agent coordination training
3. Concept alignment monitoring
4. Explanation quality evaluation
5. Model checkpointing and logging
6. Hyperparameter configuration
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.acb_agent import ACBAgent, MultiAgentACBSystem
from training.acb_trainer import ACBTrainer, create_default_trainer
from utils.concept_utils import create_domain_concept_vocabulary, ConceptVocabulary
from utils.explanation_generator import create_explanation_generator, ExplanationStyle
from utils.metrics import create_default_metrics_evaluator, ComprehensiveMetricsEvaluator

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Environment settings
    domain: str = "multi_agent"  # "multi_agent", "human_ai", "navigation"
    num_agents: int = 2
    state_dim: int = 64
    action_dim: int = 4
    concept_dim: int = 32
    hidden_dim: int = 128
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    lambda_explain: float = 0.1
    lambda_diverse: float = 0.05
    gamma: float = 0.99
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    
    # Concept settings
    concept_alignment_threshold: float = 0.6
    explanation_k: int = 3
    explanation_style: str = "detailed"  # "simple", "detailed", "causal"
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 50
    checkpoint_interval: int = 100
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

class SimpleMultiAgentEnvironment:
    """
    Simple multi-agent environment for testing ACB-Agent
    Implements a cooperative navigation task where agents must coordinate
    """
    
    def __init__(self, num_agents: int = 2, state_dim: int = 64, action_dim: int = 4):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = 200
        self.current_step = 0
        
        # Environment state
        self.agent_positions = np.random.rand(num_agents, 2) * 10
        self.target_positions = np.random.rand(num_agents, 2) * 10
        self.done = False
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial states"""
        self.current_step = 0
        self.done = False
        self.agent_positions = np.random.rand(self.num_agents, 2) * 10
        self.target_positions = np.random.rand(self.num_agents, 2) * 10
        return self._get_states()
    
    def _get_states(self) -> np.ndarray:
        """Get current states for all agents"""
        states = []
        for i in range(self.num_agents):
            # State includes: own position, target position, other agents' positions
            state = np.zeros(self.state_dim)
            state[:2] = self.agent_positions[i]  # Own position
            state[2:4] = self.target_positions[i]  # Target position
            
            # Other agents' positions
            other_positions = np.concatenate([
                self.agent_positions[j] for j in range(self.num_agents) if j != i
            ])
            state[4:4+len(other_positions)] = other_positions
            
            # Add some noise for realism
            state += np.random.normal(0, 0.1, self.state_dim)
            states.append(state)
            
        return np.array(states)
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """Execute actions and return next states, rewards, done, info"""
        self.current_step += 1
        
        # Update agent positions based on actions
        action_map = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}  # up, down, right, left
        
        for i, action in enumerate(actions):
            if action in action_map:
                self.agent_positions[i] += np.array(action_map[action]) * 0.5
                # Keep agents in bounds
                self.agent_positions[i] = np.clip(self.agent_positions[i], 0, 10)
        
        # Calculate rewards
        rewards = []
        for i in range(self.num_agents):
            # Distance to target (negative reward)
            dist_to_target = np.linalg.norm(self.agent_positions[i] - self.target_positions[i])
            reward = -dist_to_target
            
            # Bonus for being close to target
            if dist_to_target < 1.0:
                reward += 10.0
            
            # Penalty for collision with other agents
            for j in range(self.num_agents):
                if i != j:
                    dist_to_other = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                    if dist_to_other < 0.5:
                        reward -= 5.0
            
            rewards.append(reward)
        
        # Check if done
        self.done = (self.current_step >= self.max_steps or 
                    all(np.linalg.norm(self.agent_positions[i] - self.target_positions[i]) < 1.0 
                        for i in range(self.num_agents)))
        
        info = {
            'agent_positions': self.agent_positions.copy(),
            'target_positions': self.target_positions.copy(),
            'distances_to_target': [np.linalg.norm(self.agent_positions[i] - self.target_positions[i]) 
                                  for i in range(self.num_agents)]
        }
        
        return self._get_states(), rewards, self.done, info

class ACBTrainingPipeline:
    """Complete training pipeline for ACB-Agent"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_random_seeds()
        
        # Initialize components
        self.concept_vocab = self.create_concept_vocabulary()
        self.agent_system = self.create_agent_system()
        self.trainer = self.create_trainer()
        self.environment = self.create_environment()
        self.metrics_evaluator = self.create_metrics_evaluator()
        self.explanation_generator = self.create_explanation_generator()
        
        # Training state
        self.episode_count = 0
        self.best_performance = -float('inf')
        self.training_history = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_file = os.path.join(self.config.log_dir, f"training_{int(time.time())}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training configuration: {asdict(self.config)}")
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def setup_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
    
    def create_concept_vocabulary(self) -> ConceptVocabulary:
        """Create domain-specific concept vocabulary"""
        return create_domain_concept_vocabulary(
            concept_dim=self.config.concept_dim,
            domain=self.config.domain
        )
    
    def create_agent_system(self):
        """Create ACB agent system (single or multi-agent)"""
        if self.config.num_agents == 1:
            return ACBAgent(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                concept_dim=self.config.concept_dim,
                hidden_dim=self.config.hidden_dim,
                concept_names=self.concept_vocab.concept_names,
                domain=self.config.domain,
                device=self.config.device
            )
        else:
            return MultiAgentACBSystem(
                num_agents=self.config.num_agents,
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                concept_dim=self.config.concept_dim,
                hidden_dim=self.config.hidden_dim,
                shared_concepts=True,
                concept_names=self.concept_vocab.concept_names,
                domain=self.config.domain,
                device=self.config.device
            )
    
    def create_trainer(self):
        """Create ACB trainer"""
        if isinstance(self.agent_system, MultiAgentACBSystem):
            # For multi-agent, use the first agent for trainer
            agent = self.agent_system.agents[0]
        else:
            agent = self.agent_system
            
        return ACBTrainer(
            agent=agent,
            learning_rate=self.config.learning_rate,
            lambda_explain=self.config.lambda_explain,
            lambda_diverse=self.config.lambda_diverse,
            gamma=self.config.gamma,
            device=self.config.device
        )
    
    def create_environment(self):
        """Create training environment"""
        return SimpleMultiAgentEnvironment(
            num_agents=self.config.num_agents,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim
        )
    
    def create_metrics_evaluator(self) -> ComprehensiveMetricsEvaluator:
        """Create metrics evaluator"""
        return create_default_metrics_evaluator(domain=self.config.domain)
    
    def create_explanation_generator(self):
        """Create explanation generator"""
        style = ExplanationStyle.DETAILED
        if self.config.explanation_style == "simple":
            style = ExplanationStyle.SIMPLE
        elif self.config.explanation_style == "causal":
            style = ExplanationStyle.CAUSAL
            
        return create_explanation_generator(
            concept_vocab=self.concept_vocab,
            style=style,
            k_concepts=self.config.explanation_k
        )
    
    def train_single_episode(self) -> Dict[str, Any]:
        """Train a single episode"""
        # Reset environment
        states = self.environment.reset()
        
        # Collect trajectory
        trajectory_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'concept_activations': [],
            'explanations': [],
            'done': False
        }
        
        total_reward = 0
        step_count = 0
        
        while not trajectory_data['done'] and step_count < self.config.max_steps_per_episode:
            # Convert states to tensor
            if isinstance(self.agent_system, MultiAgentACBSystem):
                state_tensors = [torch.FloatTensor(state).unsqueeze(0).to(self.config.device) 
                               for state in states]
                
                # Get actions from all agents
                agent_outputs = self.agent_system.act(
                    states=state_tensors,
                    deterministic=False,
                    return_explanations=True
                )
                
                actions = [output['actions'].item() for output in agent_outputs]
                concept_activations = [output['concept_activations'] for output in agent_outputs]
                explanations = [output.get('explanation', '') for output in agent_outputs]
                
            else:
                state_tensor = torch.FloatTensor(states[0]).unsqueeze(0).to(self.config.device)
                output = self.agent_system.act(
                    state=state_tensor,
                    deterministic=False,
                    return_explanation=True,
                    explanation_k=self.config.explanation_k
                )
                
                actions = [output['actions'].item()]
                concept_activations = [output['concept_activations']]
                explanations = [output.get('explanation', '')]
            
            # Execute actions in environment
            next_states, rewards, done, info = self.environment.step(actions)
            
            # Store trajectory data
            trajectory_data['states'].append(states)
            trajectory_data['actions'].append(actions)
            trajectory_data['rewards'].append(rewards)
            trajectory_data['concept_activations'].append(concept_activations)
            trajectory_data['explanations'].append(explanations)
            
            states = next_states
            total_reward += sum(rewards)
            step_count += 1
            trajectory_data['done'] = done
        
        # Train on collected trajectory
        if isinstance(self.agent_system, MultiAgentACBSystem):
            # For multi-agent, train each agent
            training_metrics = []
            for i, agent in enumerate(self.agent_system.agents):
                agent_trajectory = {
                    'states': [s[i] for s in trajectory_data['states']],
                    'actions': [a[i] for a in trajectory_data['actions']],
                    'rewards': [r[i] for r in trajectory_data['rewards']],
                    'concept_activations': [c[i] for c in trajectory_data['concept_activations']],
                    'done': trajectory_data['done']
                }
                
                # Create temporary trainer for this agent
                temp_trainer = ACBTrainer(
                    agent=agent,
                    learning_rate=self.config.learning_rate,
                    lambda_explain=self.config.lambda_explain,
                    lambda_diverse=self.config.lambda_diverse,
                    gamma=self.config.gamma,
                    device=self.config.device
                )
                
                metrics = temp_trainer.train_step(agent_trajectory)
                training_metrics.append(metrics)
            
            # Average metrics across agents
            avg_metrics = {}
            for key in training_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in training_metrics])
            
        else:
            avg_metrics = self.trainer.train_step(trajectory_data)
        
        # Episode summary
        episode_summary = {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'steps': step_count,
            'training_metrics': avg_metrics,
            'trajectory_data': trajectory_data
        }
        
        return episode_summary
    
    def evaluate_system(self, num_eval_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the current system performance"""
        eval_data = {
            'episode_data': [],
            'concept_data': {'model_concepts': [], 'human_labels': []},
            'explanation_data': []
        }
        
        total_rewards = []
        
        for _ in range(num_eval_episodes):
            states = self.environment.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            episode_concepts = []
            episode_explanations = []
            
            while not done and step_count < self.config.max_steps_per_episode:
                if isinstance(self.agent_system, MultiAgentACBSystem):
                    state_tensors = [torch.FloatTensor(state).unsqueeze(0).to(self.config.device) 
                                   for state in states]
                    agent_outputs = self.agent_system.act(
                        states=state_tensors,
                        deterministic=True,
                        return_explanations=True
                    )
                    actions = [output['actions'].item() for output in agent_outputs]
                    concept_activations = [output['concept_activations'] for output in agent_outputs]
                    explanations = [output.get('explanation', '') for output in agent_outputs]
                else:
                    state_tensor = torch.FloatTensor(states[0]).unsqueeze(0).to(self.config.device)
                    output = self.agent_system.act(
                        state=state_tensor,
                        deterministic=True,
                        return_explanation=True
                    )
                    actions = [output['actions'].item()]
                    concept_activations = [output['concept_activations']]
                    explanations = [output.get('explanation', '')]
                
                next_states, rewards, done, info = self.environment.step(actions)
                
                episode_reward += sum(rewards)
                step_count += 1
                states = next_states
                
                # Collect concept and explanation data
                episode_concepts.extend(concept_activations)
                episode_explanations.extend([{'explanation': exp, 'concepts': concepts} 
                                           for exp, concepts in zip(explanations, concept_activations)])
            
            total_rewards.append(episode_reward)
            eval_data['episode_data'].append({
                'reward': episode_reward,
                'steps': step_count,
                'success': episode_reward > 0  # Simple success criterion
            })
            
            # Add concept data (simulate human labels for evaluation)
            if episode_concepts:
                eval_data['concept_data']['model_concepts'].extend(episode_concepts)
                # Simulate human labels (in real scenario, these would come from human annotators)
                human_labels = [torch.rand_like(concepts) for concepts in episode_concepts]
                eval_data['concept_data']['human_labels'].extend(human_labels)
            
            eval_data['explanation_data'].extend(episode_explanations)
        
        # Convert to tensors for metrics computation
        if eval_data['concept_data']['model_concepts']:
            eval_data['concept_data']['model_concepts'] = torch.stack(
                eval_data['concept_data']['model_concepts']
            )
            eval_data['concept_data']['human_labels'] = torch.stack(
                eval_data['concept_data']['human_labels']
            )
        
        # Compute comprehensive metrics
        evaluation_results = self.metrics_evaluator.evaluate_system(
            episode_data=eval_data['episode_data'],
            concept_data=eval_data['concept_data'],
            explanation_data=eval_data['explanation_data']
        )
        
        return evaluation_results
    
    def save_checkpoint(self, episode: int, metrics: Dict[str, Any]):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'config': asdict(self.config),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        if isinstance(self.agent_system, MultiAgentACBSystem):
            checkpoint['agent_states'] = [agent.state_dict() for agent in self.agent_system.agents]
        else:
            checkpoint['agent_state'] = self.agent_system.state_dict()
        
        checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_episode_{episode}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        if isinstance(self.agent_system, MultiAgentACBSystem):
            for i, agent in enumerate(self.agent_system.agents):
                agent.load_state_dict(checkpoint['agent_states'][i])
        else:
            self.agent_system.load_state_dict(checkpoint['agent_state'])
        
        self.episode_count = checkpoint['episode']
        self.training_history = checkpoint.get('training_history', [])
        self.logger.info(f"Loaded checkpoint from episode {self.episode_count}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting ACB-Agent training...")
        self.logger.info(f"Training for {self.config.max_episodes} episodes")
        self.logger.info(f"Device: {self.config.device}")
        
        start_time = time.time()
        
        for episode in range(self.episode_count, self.config.max_episodes):
            self.episode_count = episode
            
            # Train single episode
            episode_summary = self.train_single_episode()
            self.training_history.append(episode_summary)
            
            # Logging
            if episode % self.config.log_interval == 0:
                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={episode_summary['total_reward']:.2f}, "
                    f"Steps={episode_summary['steps']}, "
                    f"Loss={episode_summary['training_metrics'].get('total_loss', 0):.4f}"
                )
            
            # Evaluation
            if episode % self.config.eval_interval == 0:
                self.logger.info("Running evaluation...")
                eval_results = self.evaluate_system()
                
                self.logger.info(f"Evaluation Results:")
                self.logger.info(f"  Performance: {eval_results.get('task_performance', {})}")
                self.logger.info(f"  Concept Alignment: {eval_results.get('concept_alignment', {})}")
                self.logger.info(f"  Explanation Quality: {eval_results.get('explanation_quality', {})}")
                
                # Save best model
                current_performance = eval_results.get('task_performance', {}).get('mean_reward', -float('inf'))
                if current_performance > self.best_performance:
                    self.best_performance = current_performance
                    self.save_checkpoint(episode, eval_results)
                    self.logger.info(f"New best performance: {self.best_performance:.2f}")
            
            # Regular checkpointing
            if episode % self.config.checkpoint_interval == 0:
                self.save_checkpoint(episode, episode_summary['training_metrics'])
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final evaluation
        self.logger.info("Running final evaluation...")
        final_eval = self.evaluate_system(num_eval_episodes=50)
        self.logger.info(f"Final Results: {final_eval}")
        
        return final_eval

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train ACB-Agent")
    
    # Environment settings
    parser.add_argument("--domain", type=str, default="multi_agent",
                       choices=["multi_agent", "human_ai", "navigation"],
                       help="Training domain")
    parser.add_argument("--num_agents", type=int, default=2,
                       help="Number of agents")
    parser.add_argument("--state_dim", type=int, default=64,
                       help="State dimension")
    parser.add_argument("--action_dim", type=int, default=4,
                       help="Action dimension")
    parser.add_argument("--concept_dim", type=int, default=32,
                       help="Concept dimension")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--lambda_explain", type=float, default=0.1,
                       help="Explanation loss weight")
    parser.add_argument("--lambda_diverse", type=float, default=0.05,
                       help="Diversity loss weight")
    parser.add_argument("--max_episodes", type=int, default=1000,
                       help="Maximum training episodes")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Logging interval")
    parser.add_argument("--eval_interval", type=int, default=50,
                       help="Evaluation interval")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Checkpoint save directory")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Log directory")
    
    # Other settings
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--config_file", type=str, default=None,
                       help="JSON config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    return parser.parse_args()

def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Load configuration
    if args.config_file:
        config_dict = load_config_from_file(args.config_file)
        config = TrainingConfig(**config_dict)
    else:
        # Create config from command line arguments
        config_dict = vars(args)
        if config_dict['device'] == 'auto':
            config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Remove non-config arguments
        config_dict.pop('config_file', None)
        config_dict.pop('resume', None)
        
        config = TrainingConfig(**config_dict)
    
    # Create training pipeline
    pipeline = ACBTrainingPipeline(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        pipeline.load_checkpoint(args.resume)
    
    # Start training
    try:
        final_results = pipeline.train()
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final Results: {final_results}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        pipeline.save_checkpoint(pipeline.episode_count, {})
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())