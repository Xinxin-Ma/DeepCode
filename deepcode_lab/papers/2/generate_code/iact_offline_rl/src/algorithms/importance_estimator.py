"""
Importance Estimator for IACT Algorithm

This module implements the Kullback-Leibler Importance Estimation Procedure (KLIEP)
for estimating density ratios between policy and behavior distributions in offline RL.

The importance weights w(s) = π(s)/β(s) are estimated using Gaussian RBF kernels
and convex optimization to solve the KLIEP objective.

Key Components:
- KLIEP-based density ratio estimation
- Gaussian RBF kernel computation
- Convex optimization for kernel weights
- Importance weight estimation for states

Mathematical Formulation:
- Importance weight: w(s) = π(s)/β(s)
- KLIEP objective: max_α Σᵢ log(Σⱼ αⱼK(sᵢ, sⱼ)) subject to Σᵢ αᵢ = 1, αᵢ ≥ 0
- Gaussian RBF kernel: K(s, s') = exp(-||s - s'||²/(2σ²))
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.optimize import minimize
import warnings

# Suppress scipy optimization warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class GaussianRBFKernel:
    """
    Gaussian Radial Basis Function (RBF) Kernel for KLIEP.
    
    Implements K(s, s') = exp(-||s - s'||²/(2σ²))
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize Gaussian RBF kernel.
        
        Args:
            sigma: Kernel bandwidth parameter
        """
        self.sigma = sigma
        self.sigma_squared = sigma ** 2
    
    def compute_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix K(X, Y).
        
        Args:
            X: Input tensor of shape (n_samples_x, feature_dim)
            Y: Input tensor of shape (n_samples_y, feature_dim)
            
        Returns:
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        # Compute pairwise squared distances
        # ||x - y||² = ||x||² + ||y||² - 2<x, y>
        x_norm_sq = torch.sum(X ** 2, dim=1, keepdim=True)  # (n_x, 1)
        y_norm_sq = torch.sum(Y ** 2, dim=1, keepdim=True)  # (n_y, 1)
        
        # Compute squared distances matrix
        distances_sq = x_norm_sq + y_norm_sq.T - 2 * torch.mm(X, Y.T)
        
        # Apply Gaussian RBF kernel
        kernel_matrix = torch.exp(-distances_sq / (2 * self.sigma_squared))
        
        return kernel_matrix
    
    def compute_kernel_vector(self, x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel vector K(x, Y) for single point x.
        
        Args:
            x: Single input point of shape (feature_dim,)
            Y: Reference points of shape (n_samples, feature_dim)
            
        Returns:
            Kernel vector of shape (n_samples,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, feature_dim)
        
        kernel_vector = self.compute_kernel_matrix(x, Y).squeeze(0)
        return kernel_vector


class KLIEPOptimizer:
    """
    KLIEP optimization solver for finding optimal kernel weights.
    
    Solves the convex optimization problem:
    max_α Σᵢ log(Σⱼ αⱼK(sᵢ, sⱼ)) subject to Σᵢ αᵢ = 1, αᵢ ≥ 0
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize KLIEP optimizer.
        
        Args:
            max_iter: Maximum number of optimization iterations
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, kernel_matrix: torch.Tensor) -> torch.Tensor:
        """
        Solve KLIEP optimization problem.
        
        Args:
            kernel_matrix: Kernel matrix K of shape (n_policy, n_kernels)
            
        Returns:
            Optimal kernel weights α of shape (n_kernels,)
        """
        n_policy, n_kernels = kernel_matrix.shape
        
        # Convert to numpy for scipy optimization
        K_np = kernel_matrix.detach().cpu().numpy()
        
        # Define objective function (negative log-likelihood to minimize)
        def objective(alpha):
            # Compute log-likelihood: Σᵢ log(Σⱼ αⱼK(sᵢ, sⱼ))
            kernel_weighted = np.dot(K_np, alpha)  # (n_policy,)
            
            # Add small epsilon to avoid log(0)
            kernel_weighted = np.maximum(kernel_weighted, 1e-8)
            
            # Return negative log-likelihood
            return -np.sum(np.log(kernel_weighted))
        
        # Define gradient
        def gradient(alpha):
            kernel_weighted = np.dot(K_np, alpha)  # (n_policy,)
            kernel_weighted = np.maximum(kernel_weighted, 1e-8)
            
            # Gradient: -Σᵢ K(sᵢ, ·) / (Σⱼ αⱼK(sᵢ, sⱼ))
            weights = 1.0 / kernel_weighted  # (n_policy,)
            grad = -np.dot(K_np.T, weights)  # (n_kernels,)
            
            return grad
        
        # Constraints: Σᵢ αᵢ = 1, αᵢ ≥ 0
        constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1.0}
        bounds = [(0.0, None) for _ in range(n_kernels)]
        
        # Initial guess: uniform distribution
        alpha_init = np.ones(n_kernels) / n_kernels
        
        # Solve optimization problem
        try:
            result = minimize(
                objective,
                alpha_init,
                method='SLSQP',
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )
            
            if result.success:
                alpha_opt = result.x
            else:
                logger.warning(f"KLIEP optimization failed: {result.message}")
                # Fallback to uniform weights
                alpha_opt = np.ones(n_kernels) / n_kernels
                
        except Exception as e:
            logger.error(f"KLIEP optimization error: {e}")
            # Fallback to uniform weights
            alpha_opt = np.ones(n_kernels) / n_kernels
        
        # Ensure non-negative and normalized
        alpha_opt = np.maximum(alpha_opt, 0.0)
        alpha_opt = alpha_opt / np.sum(alpha_opt)
        
        return torch.tensor(alpha_opt, dtype=torch.float32)


class ImportanceEstimator:
    """
    Main importance estimator using KLIEP for density ratio estimation.
    
    Estimates importance weights w(s) = π(s)/β(s) where:
    - π(s) is the policy state distribution
    - β(s) is the behavior state distribution
    """
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        n_kernels: int = 100,
        sigma: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        device: str = 'cpu'
    ):
        """
        Initialize importance estimator.
        
        Args:
            kernel_type: Type of kernel ('rbf' for Gaussian RBF)
            n_kernels: Number of kernel centers
            sigma: Kernel bandwidth parameter
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            device: Device for computation ('cpu' or 'cuda')
        """
        self.kernel_type = kernel_type
        self.n_kernels = n_kernels
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        
        # Initialize components
        if kernel_type == 'rbf':
            self.kernel = GaussianRBFKernel(sigma=sigma)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        self.optimizer = KLIEPOptimizer(max_iter=max_iter, tol=tol)
        
        # Fitted parameters
        self.is_fitted = False
        self.kernel_centers = None
        self.kernel_weights = None
        
        logger.info(f"Initialized ImportanceEstimator with {n_kernels} kernels, sigma={sigma}")
    
    def _select_kernel_centers(
        self, 
        behavior_states: torch.Tensor, 
        method: str = 'random'
    ) -> torch.Tensor:
        """
        Select kernel centers from behavior states.
        
        Args:
            behavior_states: Behavior state samples of shape (n_samples, state_dim)
            method: Selection method ('random', 'kmeans', 'uniform')
            
        Returns:
            Kernel centers of shape (n_kernels, state_dim)
        """
        n_samples = behavior_states.shape[0]
        
        if method == 'random':
            # Random sampling
            if n_samples <= self.n_kernels:
                # Use all samples if not enough
                centers = behavior_states.clone()
            else:
                # Random subset
                indices = torch.randperm(n_samples)[:self.n_kernels]
                centers = behavior_states[indices]
        
        elif method == 'uniform':
            # Uniform sampling
            if n_samples <= self.n_kernels:
                centers = behavior_states.clone()
            else:
                indices = torch.linspace(0, n_samples - 1, self.n_kernels, dtype=torch.long)
                centers = behavior_states[indices]
        
        else:
            raise ValueError(f"Unsupported kernel center selection method: {method}")
        
        return centers.to(self.device)
    
    def fit(
        self, 
        policy_states: torch.Tensor, 
        behavior_states: torch.Tensor,
        center_selection: str = 'random'
    ) -> 'ImportanceEstimator':
        """
        Fit importance estimator using policy and behavior state samples.
        
        Args:
            policy_states: Policy state distribution samples of shape (n_policy, state_dim)
            behavior_states: Behavior state distribution samples of shape (n_behavior, state_dim)
            center_selection: Method for selecting kernel centers
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ImportanceEstimator with {len(policy_states)} policy states, "
                   f"{len(behavior_states)} behavior states")
        
        # Move to device
        policy_states = policy_states.to(self.device)
        behavior_states = behavior_states.to(self.device)
        
        # Select kernel centers from behavior states
        self.kernel_centers = self._select_kernel_centers(behavior_states, center_selection)
        n_centers = self.kernel_centers.shape[0]
        
        logger.info(f"Selected {n_centers} kernel centers")
        
        # Compute kernel matrix K(policy_states, kernel_centers)
        kernel_matrix = self.kernel.compute_kernel_matrix(policy_states, self.kernel_centers)
        
        # Solve KLIEP optimization
        self.kernel_weights = self.optimizer.solve(kernel_matrix)
        self.kernel_weights = self.kernel_weights.to(self.device)
        
        self.is_fitted = True
        
        logger.info("ImportanceEstimator fitting completed successfully")
        
        return self
    
    def estimate_weights(self, states: torch.Tensor) -> torch.Tensor:
        """
        Estimate importance weights for given states.
        
        Args:
            states: States to estimate weights for, shape (n_samples, state_dim)
            
        Returns:
            Importance weights of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("ImportanceEstimator must be fitted before estimating weights")
        
        states = states.to(self.device)
        
        # Compute kernel matrix K(states, kernel_centers)
        kernel_matrix = self.kernel.compute_kernel_matrix(states, self.kernel_centers)
        
        # Compute importance weights: w(s) = Σⱼ αⱼK(s, sⱼ)
        importance_weights = torch.mv(kernel_matrix, self.kernel_weights)
        
        # Ensure positive weights
        importance_weights = torch.clamp(importance_weights, min=1e-8)
        
        return importance_weights
    
    def estimate_single_weight(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate importance weight for a single state.
        
        Args:
            state: Single state of shape (state_dim,)
            
        Returns:
            Importance weight scalar
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)
        
        weight = self.estimate_weights(state)
        return weight.squeeze()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the fitted estimator.
        
        Returns:
            Dictionary with estimator statistics
        """
        if not self.is_fitted:
            return {"fitted": False}
        
        stats = {
            "fitted": True,
            "n_kernels": len(self.kernel_centers),
            "kernel_type": self.kernel_type,
            "sigma": self.sigma,
            "weight_mean": float(self.kernel_weights.mean()),
            "weight_std": float(self.kernel_weights.std()),
            "weight_min": float(self.kernel_weights.min()),
            "weight_max": float(self.kernel_weights.max())
        }
        
        return stats
    
    def save_state(self) -> Dict:
        """
        Save estimator state for serialization.
        
        Returns:
            State dictionary
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save state of unfitted estimator")
        
        state = {
            "kernel_type": self.kernel_type,
            "n_kernels": self.n_kernels,
            "sigma": self.sigma,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "kernel_centers": self.kernel_centers.cpu(),
            "kernel_weights": self.kernel_weights.cpu(),
            "is_fitted": self.is_fitted
        }
        
        return state
    
    def load_state(self, state: Dict) -> 'ImportanceEstimator':
        """
        Load estimator state from serialization.
        
        Args:
            state: State dictionary
            
        Returns:
            Self for method chaining
        """
        self.kernel_type = state["kernel_type"]
        self.n_kernels = state["n_kernels"]
        self.sigma = state["sigma"]
        self.max_iter = state["max_iter"]
        self.tol = state["tol"]
        self.kernel_centers = state["kernel_centers"].to(self.device)
        self.kernel_weights = state["kernel_weights"].to(self.device)
        self.is_fitted = state["is_fitted"]
        
        # Reinitialize kernel
        if self.kernel_type == 'rbf':
            self.kernel = GaussianRBFKernel(sigma=self.sigma)
        
        return self


def create_importance_estimator(config: Optional[Dict] = None) -> ImportanceEstimator:
    """
    Factory function to create ImportanceEstimator with configuration.
    
    Args:
        config: Configuration dictionary with estimator parameters
        
    Returns:
        Configured ImportanceEstimator instance
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        'kernel_type': 'rbf',
        'n_kernels': 100,
        'sigma': 1.0,
        'max_iter': 1000,
        'tol': 1e-6,
        'device': 'cpu'
    }
    
    # Update with provided config
    default_config.update(config)
    
    return ImportanceEstimator(**default_config)


# Utility functions for importance weight processing

def normalize_importance_weights(
    weights: torch.Tensor, 
    method: str = 'mean',
    clip_range: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Normalize importance weights to prevent extreme values.
    
    Args:
        weights: Raw importance weights
        method: Normalization method ('mean', 'max', 'none')
        clip_range: Optional clipping range (min_val, max_val)
        
    Returns:
        Normalized importance weights
    """
    if method == 'mean':
        # Normalize by mean
        normalized_weights = weights / weights.mean()
    elif method == 'max':
        # Normalize by maximum
        normalized_weights = weights / weights.max()
    elif method == 'none':
        normalized_weights = weights
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Optional clipping
    if clip_range is not None:
        min_val, max_val = clip_range
        normalized_weights = torch.clamp(normalized_weights, min=min_val, max=max_val)
    
    return normalized_weights


def compute_effective_sample_size(weights: torch.Tensor) -> float:
    """
    Compute effective sample size from importance weights.
    
    ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²
    
    Args:
        weights: Importance weights
        
    Returns:
        Effective sample size
    """
    weights_sum = weights.sum()
    weights_sq_sum = (weights ** 2).sum()
    
    ess = (weights_sum ** 2) / weights_sq_sum
    return float(ess)


def validate_importance_weights(weights: torch.Tensor) -> Dict[str, float]:
    """
    Validate and analyze importance weights.
    
    Args:
        weights: Importance weights to validate
        
    Returns:
        Validation statistics
    """
    stats = {
        "n_samples": len(weights),
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "median": float(weights.median()),
        "effective_sample_size": compute_effective_sample_size(weights),
        "zero_weights": int((weights == 0).sum()),
        "negative_weights": int((weights < 0).sum())
    }
    
    # Add warnings
    warnings = []
    if stats["negative_weights"] > 0:
        warnings.append(f"Found {stats['negative_weights']} negative weights")
    if stats["zero_weights"] > 0:
        warnings.append(f"Found {stats['zero_weights']} zero weights")
    if stats["max"] / stats["mean"] > 100:
        warnings.append("Very large weight ratios detected")
    
    stats["warnings"] = warnings
    
    return stats


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    state_dim = 4
    n_behavior = 1000
    n_policy = 500
    
    # Behavior distribution: Normal(0, 1)
    behavior_states = torch.randn(n_behavior, state_dim)
    
    # Policy distribution: Normal(0.5, 0.8) - shifted and scaled
    policy_states = 0.5 + 0.8 * torch.randn(n_policy, state_dim)
    
    print("Testing ImportanceEstimator...")
    print(f"Behavior states: {behavior_states.shape}")
    print(f"Policy states: {policy_states.shape}")
    
    # Create and fit estimator
    estimator = create_importance_estimator({
        'n_kernels': 50,
        'sigma': 1.0,
        'max_iter': 500
    })
    
    # Fit estimator
    estimator.fit(policy_states, behavior_states)
    
    # Estimate weights for test states
    test_states = torch.randn(100, state_dim)
    importance_weights = estimator.estimate_weights(test_states)
    
    print(f"\nImportance weights statistics:")
    stats = validate_importance_weights(importance_weights)
    for key, value in stats.items():
        if key != "warnings":
            print(f"  {key}: {value}")
    
    if stats["warnings"]:
        print("  Warnings:")
        for warning in stats["warnings"]:
            print(f"    - {warning}")
    
    # Test estimator statistics
    estimator_stats = estimator.get_statistics()
    print(f"\nEstimator statistics:")
    for key, value in estimator_stats.items():
        print(f"  {key}: {value}")
    
    print("\nImportanceEstimator testing completed successfully!")