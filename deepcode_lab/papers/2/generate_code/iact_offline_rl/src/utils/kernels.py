"""
Kernel functions for importance estimation in IACT algorithm.

This module provides various kernel implementations used in the KLIEP-based
density ratio estimation, particularly Gaussian RBF kernels for computing
similarity between states in the importance estimation process.
"""

import torch
import numpy as np
import logging
from typing import Optional, Union, Tuple, Dict, Any
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class BaseKernel(ABC):
    """Abstract base class for kernel functions."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize base kernel.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        
    @abstractmethod
    def compute(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix between X and Y.
        
        Args:
            X: Input tensor of shape (n_samples_x, n_features)
            Y: Input tensor of shape (n_samples_y, n_features)
            
        Returns:
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        pass
    
    @abstractmethod
    def compute_diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal elements of kernel matrix K(X, X).
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Diagonal elements of shape (n_samples,)
        """
        pass


class GaussianRBFKernel(BaseKernel):
    """
    Gaussian Radial Basis Function (RBF) kernel.
    
    K(x, y) = exp(-||x - y||² / (2 * sigma²))
    
    This is the primary kernel used in KLIEP for importance estimation.
    """
    
    def __init__(self, sigma: float = 1.0, device: str = 'cpu'):
        """
        Initialize Gaussian RBF kernel.
        
        Args:
            sigma: Bandwidth parameter (standard deviation)
            device: Device to run computations on
        """
        super().__init__(device)
        self.sigma = sigma
        self.gamma = 1.0 / (2.0 * sigma ** 2)  # Precompute for efficiency
        
    def compute(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel matrix between X and Y.
        
        Args:
            X: Input tensor of shape (n_x, d)
            Y: Input tensor of shape (n_y, d)
            
        Returns:
            Kernel matrix of shape (n_x, n_y)
        """
        # Ensure tensors are on correct device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Compute squared Euclidean distances efficiently
        # ||x - y||² = ||x||² + ||y||² - 2<x, y>
        X_norm_sq = torch.sum(X ** 2, dim=1, keepdim=True)  # (n_x, 1)
        Y_norm_sq = torch.sum(Y ** 2, dim=1, keepdim=True)  # (n_y, 1)
        
        # Compute cross terms
        XY = torch.mm(X, Y.t())  # (n_x, n_y)
        
        # Compute squared distances
        distances_sq = X_norm_sq + Y_norm_sq.t() - 2 * XY  # (n_x, n_y)
        
        # Apply RBF kernel
        kernel_matrix = torch.exp(-self.gamma * distances_sq)
        
        return kernel_matrix
    
    def compute_diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal elements K(x_i, x_i) = 1 for RBF kernel.
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Diagonal elements (all ones) of shape (n_samples,)
        """
        return torch.ones(X.shape[0], device=self.device)
    
    def compute_self_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix K(X, X) efficiently.
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Symmetric kernel matrix of shape (n_samples, n_samples)
        """
        return self.compute(X, X)


class LinearKernel(BaseKernel):
    """
    Linear kernel: K(x, y) = <x, y>
    
    Simple dot product kernel, useful for linear relationships.
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize linear kernel."""
        super().__init__(device)
    
    def compute(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute linear kernel matrix.
        
        Args:
            X: Input tensor of shape (n_x, d)
            Y: Input tensor of shape (n_y, d)
            
        Returns:
            Kernel matrix of shape (n_x, n_y)
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.mm(X, Y.t())
    
    def compute_diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal elements ||x_i||².
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Diagonal elements of shape (n_samples,)
        """
        X = X.to(self.device)
        return torch.sum(X ** 2, dim=1)


class PolynomialKernel(BaseKernel):
    """
    Polynomial kernel: K(x, y) = (<x, y> + c)^d
    
    Captures polynomial relationships between features.
    """
    
    def __init__(self, degree: int = 2, coef0: float = 1.0, device: str = 'cpu'):
        """
        Initialize polynomial kernel.
        
        Args:
            degree: Polynomial degree
            coef0: Independent term
            device: Device to run computations on
        """
        super().__init__(device)
        self.degree = degree
        self.coef0 = coef0
    
    def compute(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial kernel matrix.
        
        Args:
            X: Input tensor of shape (n_x, d)
            Y: Input tensor of shape (n_y, d)
            
        Returns:
            Kernel matrix of shape (n_x, n_y)
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        linear_kernel = torch.mm(X, Y.t())
        return (linear_kernel + self.coef0) ** self.degree
    
    def compute_diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal elements.
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Diagonal elements of shape (n_samples,)
        """
        X = X.to(self.device)
        linear_diag = torch.sum(X ** 2, dim=1)
        return (linear_diag + self.coef0) ** self.degree


class KernelMatrix:
    """
    Utility class for efficient kernel matrix operations.
    
    Provides methods for computing, storing, and manipulating kernel matrices
    with memory-efficient operations for large datasets.
    """
    
    def __init__(self, kernel: BaseKernel, chunk_size: int = 1000):
        """
        Initialize kernel matrix handler.
        
        Args:
            kernel: Kernel function to use
            chunk_size: Size of chunks for memory-efficient computation
        """
        self.kernel = kernel
        self.chunk_size = chunk_size
        
    def compute_chunked(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix in chunks to save memory.
        
        Args:
            X: Input tensor of shape (n_x, d)
            Y: Input tensor of shape (n_y, d)
            
        Returns:
            Kernel matrix of shape (n_x, n_y)
        """
        n_x, n_y = X.shape[0], Y.shape[0]
        
        # If matrices are small, compute directly
        if n_x * n_y <= self.chunk_size ** 2:
            return self.kernel.compute(X, Y)
        
        # Compute in chunks
        kernel_matrix = torch.zeros(n_x, n_y, device=self.kernel.device)
        
        for i in range(0, n_x, self.chunk_size):
            end_i = min(i + self.chunk_size, n_x)
            X_chunk = X[i:end_i]
            
            for j in range(0, n_y, self.chunk_size):
                end_j = min(j + self.chunk_size, n_y)
                Y_chunk = Y[j:end_j]
                
                kernel_matrix[i:end_i, j:end_j] = self.kernel.compute(X_chunk, Y_chunk)
        
        return kernel_matrix
    
    def compute_row_norms(self, K: torch.Tensor) -> torch.Tensor:
        """
        Compute row norms of kernel matrix.
        
        Args:
            K: Kernel matrix of shape (n, m)
            
        Returns:
            Row norms of shape (n,)
        """
        return torch.norm(K, dim=1)
    
    def compute_column_norms(self, K: torch.Tensor) -> torch.Tensor:
        """
        Compute column norms of kernel matrix.
        
        Args:
            K: Kernel matrix of shape (n, m)
            
        Returns:
            Column norms of shape (m,)
        """
        return torch.norm(K, dim=0)


class AdaptiveKernel:
    """
    Adaptive kernel that automatically selects bandwidth for RBF kernel.
    
    Uses heuristics like median distance or cross-validation to select
    optimal kernel parameters.
    """
    
    def __init__(self, base_kernel_type: str = 'rbf', device: str = 'cpu'):
        """
        Initialize adaptive kernel.
        
        Args:
            base_kernel_type: Type of base kernel ('rbf', 'linear', 'polynomial')
            device: Device to run computations on
        """
        self.base_kernel_type = base_kernel_type
        self.device = device
        self.fitted_kernel = None
        
    def fit(self, X: torch.Tensor, method: str = 'median') -> BaseKernel:
        """
        Fit kernel parameters to data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            method: Method for parameter selection ('median', 'mean', 'quantile')
            
        Returns:
            Fitted kernel
        """
        X = X.to(self.device)
        
        if self.base_kernel_type == 'rbf':
            sigma = self._estimate_rbf_bandwidth(X, method)
            self.fitted_kernel = GaussianRBFKernel(sigma=sigma, device=self.device)
            logger.info(f"Fitted RBF kernel with sigma={sigma:.4f}")
            
        elif self.base_kernel_type == 'linear':
            self.fitted_kernel = LinearKernel(device=self.device)
            logger.info("Fitted linear kernel")
            
        elif self.base_kernel_type == 'polynomial':
            # Use default parameters for polynomial
            self.fitted_kernel = PolynomialKernel(device=self.device)
            logger.info("Fitted polynomial kernel with default parameters")
            
        else:
            raise ValueError(f"Unknown kernel type: {self.base_kernel_type}")
        
        return self.fitted_kernel
    
    def _estimate_rbf_bandwidth(self, X: torch.Tensor, method: str = 'median') -> float:
        """
        Estimate RBF kernel bandwidth using distance-based heuristics.
        
        Args:
            X: Data tensor of shape (n_samples, n_features)
            method: Estimation method
            
        Returns:
            Estimated bandwidth (sigma)
        """
        n_samples = X.shape[0]
        
        # For large datasets, use subset for efficiency
        if n_samples > 2000:
            indices = torch.randperm(n_samples)[:2000]
            X_subset = X[indices]
        else:
            X_subset = X
        
        # Compute pairwise distances
        distances = torch.cdist(X_subset, X_subset, p=2)
        
        # Remove diagonal (zero distances)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=self.device)
        distances_flat = distances[mask]
        
        if method == 'median':
            sigma = torch.median(distances_flat).item()
        elif method == 'mean':
            sigma = torch.mean(distances_flat).item()
        elif method == 'quantile':
            sigma = torch.quantile(distances_flat, 0.5).item()
        else:
            raise ValueError(f"Unknown bandwidth estimation method: {method}")
        
        # Ensure sigma is not too small
        sigma = max(sigma, 1e-6)
        
        return sigma


def create_kernel(kernel_type: str, **kwargs) -> BaseKernel:
    """
    Factory function to create kernel instances.
    
    Args:
        kernel_type: Type of kernel ('rbf', 'linear', 'polynomial')
        **kwargs: Kernel-specific parameters
        
    Returns:
        Kernel instance
    """
    device = kwargs.get('device', 'cpu')
    
    if kernel_type.lower() == 'rbf' or kernel_type.lower() == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return GaussianRBFKernel(sigma=sigma, device=device)
    
    elif kernel_type.lower() == 'linear':
        return LinearKernel(device=device)
    
    elif kernel_type.lower() == 'polynomial' or kernel_type.lower() == 'poly':
        degree = kwargs.get('degree', 2)
        coef0 = kwargs.get('coef0', 1.0)
        return PolynomialKernel(degree=degree, coef0=coef0, device=device)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def compute_kernel_alignment(K1: torch.Tensor, K2: torch.Tensor) -> float:
    """
    Compute kernel alignment between two kernel matrices.
    
    Alignment = <K1, K2>_F / (||K1||_F * ||K2||_F)
    
    Args:
        K1: First kernel matrix
        K2: Second kernel matrix
        
    Returns:
        Kernel alignment score (0 to 1)
    """
    # Flatten matrices and compute inner product
    K1_flat = K1.flatten()
    K2_flat = K2.flatten()
    
    inner_product = torch.dot(K1_flat, K2_flat)
    norm_K1 = torch.norm(K1_flat)
    norm_K2 = torch.norm(K2_flat)
    
    alignment = inner_product / (norm_K1 * norm_K2)
    return alignment.item()


def estimate_effective_dimension(K: torch.Tensor, threshold: float = 0.95) -> int:
    """
    Estimate effective dimension of kernel matrix using eigenvalue decomposition.
    
    Args:
        K: Kernel matrix
        threshold: Cumulative variance threshold
        
    Returns:
        Effective dimension (number of eigenvalues needed to explain threshold variance)
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(K).real
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    
    # Compute cumulative variance explained
    total_variance = torch.sum(eigenvalues)
    cumulative_variance = torch.cumsum(eigenvalues, dim=0) / total_variance
    
    # Find effective dimension
    effective_dim = torch.sum(cumulative_variance < threshold).item() + 1
    
    return min(effective_dim, len(eigenvalues))


def validate_kernel_matrix(K: torch.Tensor, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Validate properties of a kernel matrix.
    
    Args:
        K: Kernel matrix to validate
        tol: Numerical tolerance
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check if matrix is square
    results['is_square'] = K.shape[0] == K.shape[1]
    
    if results['is_square']:
        # Check symmetry
        symmetry_error = torch.max(torch.abs(K - K.t())).item()
        results['is_symmetric'] = symmetry_error < tol
        results['symmetry_error'] = symmetry_error
        
        # Check positive semi-definiteness
        try:
            eigenvalues = torch.linalg.eigvals(K).real
            min_eigenvalue = torch.min(eigenvalues).item()
            results['is_psd'] = min_eigenvalue >= -tol
            results['min_eigenvalue'] = min_eigenvalue
            results['condition_number'] = (torch.max(eigenvalues) / torch.max(eigenvalues[eigenvalues > tol])).item()
        except Exception as e:
            results['eigenvalue_error'] = str(e)
            results['is_psd'] = False
    
    # Check for NaN or infinite values
    results['has_nan'] = torch.isnan(K).any().item()
    results['has_inf'] = torch.isinf(K).any().item()
    
    # Compute basic statistics
    results['mean'] = torch.mean(K).item()
    results['std'] = torch.std(K).item()
    results['min'] = torch.min(K).item()
    results['max'] = torch.max(K).item()
    
    return results


# Example usage and testing functions
def test_kernels():
    """Test kernel implementations with sample data."""
    print("Testing kernel implementations...")
    
    # Generate sample data
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    Y = torch.randn(50, 5)
    
    # Test RBF kernel
    rbf_kernel = GaussianRBFKernel(sigma=1.0)
    K_rbf = rbf_kernel.compute(X, Y)
    print(f"RBF kernel matrix shape: {K_rbf.shape}")
    print(f"RBF kernel range: [{K_rbf.min():.4f}, {K_rbf.max():.4f}]")
    
    # Test adaptive kernel
    adaptive_kernel = AdaptiveKernel('rbf')
    fitted_kernel = adaptive_kernel.fit(X)
    K_adaptive = fitted_kernel.compute(X, Y)
    print(f"Adaptive kernel matrix shape: {K_adaptive.shape}")
    
    # Validate kernel matrix
    K_self = rbf_kernel.compute(X, X)
    validation = validate_kernel_matrix(K_self)
    print(f"Kernel validation: {validation}")
    
    print("Kernel tests completed successfully!")


if __name__ == "__main__":
    test_kernels()