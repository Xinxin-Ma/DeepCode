"""
Multi-Agent Environment Wrapper for ACB-Agent Systems

This module provides environment wrappers and utilities for multi-agent reinforcement
learning with explainable ACB agents. Supports various multi-agent coordination tasks
including cooperative navigation, resource allocation, and human-AI collaboration scenarios.

Key Features:
- Multi-agent environment wrapper with standardized interface
- Support for heterogeneous and homogeneous agent teams
- Human-AI interaction simulation capabilities
- Coordination task implementations (navigation, resource allocation)
- State normalization and action space management
- Episode management and trajectory collection
"""

import numpy as np
import torch
import gym
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentEnvWrapper(gym.Env):
    """
    Multi-agent environment wrapper that provides standardized interface
    for ACB agents in various coordination tasks.
    
    Supports:
    - Multiple agents with individual or shared observation/action spaces
    - Centralized or decentralized execution
    - Human-AI collaboration scenarios
    - Episode management and metrics collection
    """
    
    def __init__(self, 
                 base_env: Optional[gym.Env] = None,
                 num_agents: int = 2,
                 state_dim: int = 10,
                 action_dim: int = 4,
                 max_episode_steps: int = 200,
                 reward_type: str = "cooperative",
                 observation_type: str = "individual",
                 normalize_observations: bool = True,
                 enable_human_interaction: bool = False):
        """
        Initialize multi-agent environment wrapper.
        
        Args:
            base_env: Base gym environment (if None, creates default coordination task)
            num_agents: Number of agents in the environment
            state_dim: Dimension of state observations
            action_dim: Dimension of action space
            max_episode_steps: Maximum steps per episode
            reward_type: "cooperative", "competitive", or "mixed"
            observation_type: "individual", "shared", or "partial"
            normalize_observations: Whether to normalize observations
            enable_human_interaction: Enable human-AI interaction features
        """
        super().__init__()
        
        self.base_env = base_env
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.observation_type = observation_type
        self.normalize_observations = normalize_observations
        self.enable_human_interaction = enable_human_interaction
        
        # Environment state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_length = 0
        self.done = False
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Initialize environment state
        self.reset()
        
        logger.info(f"Initialized MultiAgentEnvWrapper with {num_agents} agents")
    
    def _setup_spaces(self):
        """Setup observation and action spaces for multi-agent environment."""
        # Individual agent spaces
        if self.observation_type == "individual":
            obs_dim = self.state_dim
        elif self.observation_type == "shared":
            obs_dim = self.state_dim * self.num_agents
        else:  # partial
            obs_dim = self.state_dim + (self.state_dim // 2) * (self.num_agents - 1)
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Multi-agent spaces (lists of individual spaces)
        self.observation_spaces = [self.observation_space for _ in range(self.num_agents)]
        self.action_spaces = [self.action_space for _ in range(self.num_agents)]
    
    def reset(self) -> List[np.ndarray]:
        """
        Reset environment and return initial observations.
        
        Returns:
            List of initial observations for each agent
        """
        self.current_step = 0
        self.episode_rewards = [0.0] * self.num_agents
        self.episode_length = 0
        self.done = False
        
        # Reset base environment if available
        if self.base_env is not None:
            base_obs = self.base_env.reset()
            observations = self._process_base_observations(base_obs)
        else:
            # Generate default initial observations
            observations = self._generate_initial_observations()
        
        # Normalize observations if enabled
        if self.normalize_observations:
            observations = [self._normalize_observation(obs) for obs in observations]
        
        logger.debug(f"Environment reset, initial observations shape: {[obs.shape for obs in observations]}")
        return observations
    
    def step(self, actions: List[Union[int, np.ndarray]]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations: List of next observations
            rewards: List of rewards for each agent
            done: Whether episode is finished
            info: Additional information dictionary
        """
        self.current_step += 1
        
        # Process actions
        processed_actions = self._process_actions(actions)
        
        # Execute step in base environment or simulate
        if self.base_env is not None:
            next_obs, rewards, done, info = self.base_env.step(processed_actions)
            observations = self._process_base_observations(next_obs)
            rewards = self._process_base_rewards(rewards)
        else:
            observations, rewards, done, info = self._simulate_step(processed_actions)
        
        # Update episode tracking
        for i, reward in enumerate(rewards):
            self.episode_rewards[i] += reward
        self.episode_length = self.current_step
        
        # Check episode termination
        if self.current_step >= self.max_episode_steps:
            done = True
        
        self.done = done
        
        # Normalize observations if enabled
        if self.normalize_observations:
            observations = [self._normalize_observation(obs) for obs in observations]
        
        # Add episode info
        info.update({
            'episode_step': self.current_step,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_length': self.episode_length,
            'agents': self.num_agents
        })
        
        return observations, rewards, done, info
    
    def _generate_initial_observations(self) -> List[np.ndarray]:
        """Generate initial observations for default coordination task."""
        observations = []
        
        for i in range(self.num_agents):
            if self.observation_type == "individual":
                # Individual agent observation
                obs = np.random.normal(0, 0.1, self.state_dim).astype(np.float32)
                # Add agent position (first 2 dimensions)
                obs[0] = np.random.uniform(-1, 1)  # x position
                obs[1] = np.random.uniform(-1, 1)  # y position
                
            elif self.observation_type == "shared":
                # Shared global observation
                obs = np.random.normal(0, 0.1, self.state_dim * self.num_agents).astype(np.float32)
                # Add all agent positions
                for j in range(self.num_agents):
                    obs[j*self.state_dim] = np.random.uniform(-1, 1)     # x position
                    obs[j*self.state_dim + 1] = np.random.uniform(-1, 1) # y position
                    
            else:  # partial observation
                # Own state + partial info about others
                own_dim = self.state_dim
                other_dim = self.state_dim // 2
                total_dim = own_dim + other_dim * (self.num_agents - 1)
                
                obs = np.random.normal(0, 0.1, total_dim).astype(np.float32)
                obs[0] = np.random.uniform(-1, 1)  # own x position
                obs[1] = np.random.uniform(-1, 1)  # own y position
            
            observations.append(obs)
        
        return observations
    
    def _simulate_step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Simulate environment step for default coordination task.
        
        Implements a simple cooperative navigation task where agents
        need to reach target locations while avoiding collisions.
        """
        observations = []
        rewards = []
        info = {'task_type': 'cooperative_navigation'}
        
        # Simple navigation dynamics
        for i, action in enumerate(actions):
            # Generate next observation based on action
            if self.observation_type == "individual":
                obs = np.random.normal(0, 0.1, self.state_dim).astype(np.float32)
                
                # Simple movement dynamics (action affects position)
                if action == 0:  # move up
                    obs[1] += 0.1
                elif action == 1:  # move down
                    obs[1] -= 0.1
                elif action == 2:  # move left
                    obs[0] -= 0.1
                elif action == 3:  # move right
                    obs[0] += 0.1
                
                # Clip to bounds
                obs[0] = np.clip(obs[0], -1, 1)
                obs[1] = np.clip(obs[1], -1, 1)
                
            else:
                # For shared/partial observations, create more complex state
                if self.observation_type == "shared":
                    obs_dim = self.state_dim * self.num_agents
                else:
                    obs_dim = self.state_dim + (self.state_dim // 2) * (self.num_agents - 1)
                
                obs = np.random.normal(0, 0.1, obs_dim).astype(np.float32)
            
            observations.append(obs)
            
            # Compute reward based on task and reward type
            if self.reward_type == "cooperative":
                # Cooperative reward: shared success
                reward = self._compute_cooperative_reward(i, action, observations)
            elif self.reward_type == "competitive":
                # Competitive reward: individual success
                reward = self._compute_competitive_reward(i, action, observations)
            else:  # mixed
                # Mixed reward: combination of individual and shared
                coop_reward = self._compute_cooperative_reward(i, action, observations)
                comp_reward = self._compute_competitive_reward(i, action, observations)
                reward = 0.7 * coop_reward + 0.3 * comp_reward
            
            rewards.append(reward)
        
        # Episode termination condition
        done = False
        if self.current_step >= self.max_episode_steps:
            done = True
        
        return observations, rewards, done, info
    
    def _compute_cooperative_reward(self, agent_id: int, action: int, observations: List[np.ndarray]) -> float:
        """Compute cooperative reward for coordination task."""
        # Simple distance-based reward (closer to target = higher reward)
        obs = observations[agent_id]
        
        # Target is at origin (0, 0)
        if len(obs) >= 2:
            distance_to_target = np.sqrt(obs[0]**2 + obs[1]**2)
            reward = -distance_to_target  # Negative distance as reward
            
            # Bonus for coordination (all agents close to target)
            if len(observations) > 1:
                avg_distance = np.mean([np.sqrt(o[0]**2 + o[1]**2) for o in observations if len(o) >= 2])
                if avg_distance < 0.5:  # All agents close
                    reward += 1.0
        else:
            reward = np.random.normal(0, 0.1)  # Random reward for non-spatial tasks
        
        return float(reward)
    
    def _compute_competitive_reward(self, agent_id: int, action: int, observations: List[np.ndarray]) -> float:
        """Compute competitive reward for individual performance."""
        obs = observations[agent_id]
        
        # Individual performance reward
        if len(obs) >= 2:
            distance_to_target = np.sqrt(obs[0]**2 + obs[1]**2)
            reward = -distance_to_target
            
            # Penalty if other agents are closer
            if len(observations) > 1:
                other_distances = [np.sqrt(o[0]**2 + o[1]**2) for j, o in enumerate(observations) 
                                 if j != agent_id and len(o) >= 2]
                if other_distances and min(other_distances) < distance_to_target:
                    reward -= 0.5
        else:
            reward = np.random.normal(0, 0.1)
        
        return float(reward)
    
    def _process_actions(self, actions: List[Union[int, np.ndarray]]) -> List[int]:
        """Process and validate actions from agents."""
        processed_actions = []
        
        for i, action in enumerate(actions):
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = int(action.item())
                else:
                    action = int(action[0])
            elif isinstance(action, torch.Tensor):
                action = int(action.item())
            else:
                action = int(action)
            
            # Clip action to valid range
            action = np.clip(action, 0, self.action_dim - 1)
            processed_actions.append(action)
        
        return processed_actions
    
    def _process_base_observations(self, base_obs) -> List[np.ndarray]:
        """Process observations from base environment."""
        if isinstance(base_obs, list):
            return [np.array(obs, dtype=np.float32) for obs in base_obs]
        else:
            # Single observation - replicate for all agents
            obs = np.array(base_obs, dtype=np.float32)
            return [obs.copy() for _ in range(self.num_agents)]
    
    def _process_base_rewards(self, base_rewards) -> List[float]:
        """Process rewards from base environment."""
        if isinstance(base_rewards, list):
            return [float(r) for r in base_rewards]
        else:
            # Single reward - replicate for all agents
            return [float(base_rewards)] * self.num_agents
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [-1, 1] range."""
        # Simple normalization (can be enhanced with running statistics)
        obs_normalized = np.tanh(obs * 0.1)  # Soft normalization
        return obs_normalized.astype(np.float32)
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get information about current episode."""
        return {
            'episode_step': self.current_step,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_length': self.episode_length,
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'done': self.done,
            'num_agents': self.num_agents,
            'reward_type': self.reward_type,
            'observation_type': self.observation_type
        }
    
    def render(self, mode='human'):
        """Render environment (basic implementation)."""
        if mode == 'human':
            print(f"Step: {self.current_step}, Rewards: {self.episode_rewards}")
        return None
    
    def close(self):
        """Close environment and cleanup."""
        if self.base_env is not None:
            self.base_env.close()


class CooperativeNavigationEnv(MultiAgentEnvWrapper):
    """
    Specialized cooperative navigation environment for multi-agent coordination.
    
    Task: Multiple agents must navigate to target locations while avoiding
    collisions and coordinating their movements.
    """
    
    def __init__(self, 
                 num_agents: int = 3,
                 world_size: float = 2.0,
                 num_targets: int = 3,
                 collision_penalty: float = -1.0,
                 target_reward: float = 10.0,
                 **kwargs):
        """
        Initialize cooperative navigation environment.
        
        Args:
            num_agents: Number of agents
            world_size: Size of the world ([-world_size, world_size])
            num_targets: Number of target locations
            collision_penalty: Penalty for agent collisions
            target_reward: Reward for reaching targets
        """
        self.world_size = world_size
        self.num_targets = num_targets
        self.collision_penalty = collision_penalty
        self.target_reward = target_reward
        
        # Agent and target positions
        self.agent_positions = None
        self.target_positions = None
        self.targets_reached = None
        
        # Enhanced state dimension for navigation
        state_dim = 4 + 2 * num_targets  # [x, y, vx, vy] + target positions
        
        super().__init__(
            num_agents=num_agents,
            state_dim=state_dim,
            action_dim=5,  # [no-op, up, down, left, right]
            reward_type="cooperative",
            **kwargs
        )
    
    def reset(self) -> List[np.ndarray]:
        """Reset navigation environment."""
        # Initialize agent positions
        self.agent_positions = np.random.uniform(
            -self.world_size * 0.8, self.world_size * 0.8, 
            (self.num_agents, 2)
        )
        
        # Initialize target positions
        self.target_positions = np.random.uniform(
            -self.world_size * 0.9, self.world_size * 0.9,
            (self.num_targets, 2)
        )
        
        # Track which targets have been reached
        self.targets_reached = [False] * self.num_targets
        
        return super().reset()
    
    def _generate_initial_observations(self) -> List[np.ndarray]:
        """Generate initial observations for navigation task."""
        observations = []
        
        for i in range(self.num_agents):
            obs = np.zeros(self.state_dim, dtype=np.float32)
            
            # Agent position and velocity
            obs[0:2] = self.agent_positions[i]  # x, y position
            obs[2:4] = 0.0  # initial velocity
            
            # Target positions (relative to agent)
            for j, target_pos in enumerate(self.target_positions):
                obs[4 + j*2:4 + j*2 + 2] = target_pos - self.agent_positions[i]
            
            observations.append(obs)
        
        return observations
    
    def _simulate_step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """Simulate navigation step."""
        # Movement dynamics
        dt = 0.1
        max_speed = 0.2
        
        velocities = np.zeros((self.num_agents, 2))
        
        for i, action in enumerate(actions):
            if action == 1:  # up
                velocities[i] = [0, max_speed]
            elif action == 2:  # down
                velocities[i] = [0, -max_speed]
            elif action == 3:  # left
                velocities[i] = [-max_speed, 0]
            elif action == 4:  # right
                velocities[i] = [max_speed, 0]
            # action == 0 is no-op (zero velocity)
        
        # Update positions
        self.agent_positions += velocities * dt
        
        # Clip to world bounds
        self.agent_positions = np.clip(
            self.agent_positions, -self.world_size, self.world_size
        )
        
        # Generate observations
        observations = []
        for i in range(self.num_agents):
            obs = np.zeros(self.state_dim, dtype=np.float32)
            obs[0:2] = self.agent_positions[i]  # position
            obs[2:4] = velocities[i]  # velocity
            
            # Target positions (relative)
            for j, target_pos in enumerate(self.target_positions):
                obs[4 + j*2:4 + j*2 + 2] = target_pos - self.agent_positions[i]
            
            observations.append(obs)
        
        # Compute rewards
        rewards = []
        for i in range(self.num_agents):
            reward = 0.0
            
            # Target reaching reward
            for j, target_pos in enumerate(self.target_positions):
                if not self.targets_reached[j]:
                    distance = np.linalg.norm(self.agent_positions[i] - target_pos)
                    if distance < 0.1:  # Reached target
                        reward += self.target_reward
                        self.targets_reached[j] = True
                    else:
                        reward -= distance * 0.1  # Distance penalty
            
            # Collision penalty
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                    if distance < 0.15:  # Collision
                        reward += self.collision_penalty
            
            rewards.append(reward)
        
        # Episode termination
        done = all(self.targets_reached) or self.current_step >= self.max_episode_steps
        
        info = {
            'task_type': 'cooperative_navigation',
            'targets_reached': sum(self.targets_reached),
            'total_targets': self.num_targets,
            'agent_positions': self.agent_positions.copy(),
            'target_positions': self.target_positions.copy()
        }
        
        return observations, rewards, done, info


class ResourceAllocationEnv(MultiAgentEnvWrapper):
    """
    Resource allocation environment for multi-agent coordination.
    
    Task: Agents must allocate limited resources efficiently while
    maximizing collective utility.
    """
    
    def __init__(self,
                 num_agents: int = 4,
                 num_resources: int = 3,
                 total_budget: float = 100.0,
                 resource_values: Optional[List[float]] = None,
                 **kwargs):
        """
        Initialize resource allocation environment.
        
        Args:
            num_agents: Number of agents
            num_resources: Number of resource types
            total_budget: Total budget to allocate
            resource_values: Value of each resource type
        """
        self.num_resources = num_resources
        self.total_budget = total_budget
        self.resource_values = resource_values or [1.0] * num_resources
        
        # Current allocations and remaining budget
        self.allocations = None
        self.remaining_budget = None
        
        # State: [remaining_budget, current_allocations, resource_values]
        state_dim = 1 + num_resources + num_resources
        
        super().__init__(
            num_agents=num_agents,
            state_dim=state_dim,
            action_dim=num_resources + 1,  # allocate to resource i or pass
            reward_type="cooperative",
            **kwargs
        )
    
    def reset(self) -> List[np.ndarray]:
        """Reset resource allocation environment."""
        self.allocations = np.zeros(self.num_resources)
        self.remaining_budget = self.total_budget
        
        return super().reset()
    
    def _generate_initial_observations(self) -> List[np.ndarray]:
        """Generate initial observations for resource allocation."""
        observations = []
        
        for i in range(self.num_agents):
            obs = np.zeros(self.state_dim, dtype=np.float32)
            obs[0] = self.remaining_budget / self.total_budget  # normalized budget
            obs[1:1+self.num_resources] = self.allocations / self.total_budget  # normalized allocations
            obs[1+self.num_resources:] = self.resource_values  # resource values
            
            observations.append(obs)
        
        return observations
    
    def _simulate_step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """Simulate resource allocation step."""
        allocation_amount = self.total_budget / (self.num_agents * 10)  # Small allocation per step
        
        # Process allocations
        for i, action in enumerate(actions):
            if action < self.num_resources and self.remaining_budget >= allocation_amount:
                # Allocate to resource
                self.allocations[action] += allocation_amount
                self.remaining_budget -= allocation_amount
        
        # Generate observations
        observations = []
        for i in range(self.num_agents):
            obs = np.zeros(self.state_dim, dtype=np.float32)
            obs[0] = self.remaining_budget / self.total_budget
            obs[1:1+self.num_resources] = self.allocations / self.total_budget
            obs[1+self.num_resources:] = self.resource_values
            
            observations.append(obs)
        
        # Compute rewards (utility-based)
        total_utility = np.sum(self.allocations * self.resource_values)
        efficiency = total_utility / (self.total_budget - self.remaining_budget + 1e-6)
        
        rewards = [efficiency] * self.num_agents  # Shared reward
        
        # Episode termination
        done = self.remaining_budget < allocation_amount or self.current_step >= self.max_episode_steps
        
        info = {
            'task_type': 'resource_allocation',
            'total_utility': total_utility,
            'efficiency': efficiency,
            'remaining_budget': self.remaining_budget,
            'allocations': self.allocations.copy()
        }
        
        return observations, rewards, done, info


def create_multi_agent_env(env_type: str = "default", **kwargs) -> MultiAgentEnvWrapper:
    """
    Factory function to create multi-agent environments.
    
    Args:
        env_type: Type of environment ("default", "navigation", "resource_allocation")
        **kwargs: Additional arguments for environment
        
    Returns:
        MultiAgentEnvWrapper instance
    """
    if env_type == "navigation":
        return CooperativeNavigationEnv(**kwargs)
    elif env_type == "resource_allocation":
        return ResourceAllocationEnv(**kwargs)
    else:
        return MultiAgentEnvWrapper(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test basic multi-agent environment
    print("Testing MultiAgentEnvWrapper...")
    env = MultiAgentEnvWrapper(num_agents=2, state_dim=6, action_dim=4)
    
    obs = env.reset()
    print(f"Initial observations: {[o.shape for o in obs]}")
    
    # Test step
    actions = [0, 1]  # Random actions
    next_obs, rewards, done, info = env.step(actions)
    print(f"Step results - Rewards: {rewards}, Done: {done}")
    print(f"Episode info: {env.get_episode_info()}")
    
    # Test cooperative navigation
    print("\nTesting CooperativeNavigationEnv...")
    nav_env = CooperativeNavigationEnv(num_agents=3, num_targets=2)
    nav_obs = nav_env.reset()
    print(f"Navigation observations: {[o.shape for o in nav_obs]}")
    
    nav_actions = [1, 2, 3]  # Move up, down, left
    nav_next_obs, nav_rewards, nav_done, nav_info = nav_env.step(nav_actions)
    print(f"Navigation rewards: {nav_rewards}")
    print(f"Navigation info: {nav_info}")
    
    # Test resource allocation
    print("\nTesting ResourceAllocationEnv...")
    res_env = ResourceAllocationEnv(num_agents=2, num_resources=3)
    res_obs = res_env.reset()
    print(f"Resource allocation observations: {[o.shape for o in res_obs]}")
    
    res_actions = [0, 1]  # Allocate to different resources
    res_next_obs, res_rewards, res_done, res_info = res_env.step(res_actions)
    print(f"Resource allocation rewards: {res_rewards}")
    print(f"Resource allocation info: {res_info}")
    
    print("\nAll environment tests completed successfully!")