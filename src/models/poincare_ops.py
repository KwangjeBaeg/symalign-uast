"""
Poincaré Ball Operations for Hyperbolic Embeddings

Implements core mathematical operations on the Poincaré ball model of
hyperbolic space. The Poincaré ball B^n_c = {x ∈ R^n : c||x||^2 < 1}
is a Riemannian manifold with constant negative curvature -c.

Key operations:
    - Möbius addition
    - Exponential map (tangent space → manifold)
    - Logarithmic map (manifold → tangent space)
    - Hyperbolic distance
    - Poincaré mean (Einstein midpoint)

References:
    - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations" (NeurIPS 2017)
    - Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_NORM = 1e-15
BALL_EPS = 1e-5


def artanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arctanh."""
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable tanh."""
    return x.clamp(-15, 15).tanh()


def project(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Project points back onto the Poincaré ball to maintain numerical stability."""
    max_norm = (1.0 - BALL_EPS) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def lambda_x(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Conformal factor λ_c(x) = 2 / (1 - c||x||²)."""
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True).clamp(0, 1 - BALL_EPS)
    return 2.0 / (1.0 - c * x_sqnorm).clamp_min(MIN_NORM)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball.

    x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
               (1 + 2c<x,y> + c²||x||²||y||²)
    """
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True).clamp(0, 1 - BALL_EPS)
    y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True).clamp(0, 1 - BALL_EPS)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * c * xy_inner + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
    denom = 1 + 2 * c * xy_inner + c ** 2 * x_sqnorm * y_sqnorm
    result = num / denom.clamp_min(MIN_NORM)

    return project(result, c)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin: maps from tangent space T_0(B) to the ball.

    exp_0(v) = tanh(√c ||v||) * v / (√c ||v||)
    """
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    result = tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return project(result, c)


def logmap0(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map at the origin: maps from ball to tangent space T_0(B).

    log_0(y) = artanh(√c ||y||) * y / (√c ||y||)
    """
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    return artanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)


def expmap(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at point x: maps from tangent space T_x(B) to the ball.

    exp_x(v) = x ⊕_c (tanh(√c λ_x ||v|| / 2) * v / (√c ||v||))
    """
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    lam = lambda_x(x, c)
    second_term = tanh(sqrt_c * lam * v_norm / 2) * v / (sqrt_c * v_norm)
    return mobius_add(x, second_term, c)


def hyp_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Hyperbolic distance in the Poincaré ball.

    d_c(x, y) = (2/√c) artanh(√c ||(-x) ⊕_c y||)
    """
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    dist = 2.0 / sqrt_c * artanh(sqrt_c * diff_norm)
    return dist.squeeze(-1)


def hyp_distance_batch(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Pairwise hyperbolic distance matrix.

    Args:
        x: [N, D] points on Poincaré ball
        y: [M, D] points on Poincaré ball

    Returns:
        distances: [N, M] distance matrix
    """
    N = x.size(0)
    M = y.size(0)
    # Expand for pairwise computation
    x_exp = x.unsqueeze(1).expand(N, M, -1)  # [N, M, D]
    y_exp = y.unsqueeze(0).expand(N, M, -1)  # [N, M, D]
    # Flatten, compute, reshape
    x_flat = x_exp.reshape(N * M, -1)
    y_flat = y_exp.reshape(N * M, -1)
    dists = hyp_distance(x_flat, y_flat, c)
    return dists.reshape(N, M)


def poincare_mean(x: torch.Tensor, c: float = 1.0, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the Einstein midpoint (Poincaré mean) of points on the ball.

    Args:
        x: [N, D] or [B, N, D] points on the ball
        c: curvature
        weights: [N] or [B, N] optional weights

    Returns:
        mean: [D] or [B, D] mean point on the ball
    """
    sqrt_c = c ** 0.5
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    lam = lambda_x(x, c)  # [B, N, 1]
    if weights is not None:
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        lam = lam * weights.unsqueeze(-1)

    weighted_sum = (lam * x).sum(dim=1)  # [B, D]
    lam_sum = lam.sum(dim=1)  # [B, 1]
    euclidean_mean = weighted_sum / lam_sum.clamp_min(MIN_NORM)

    # Project to ball via normalization
    mean = project(euclidean_mean, c)
    if squeeze:
        mean = mean.squeeze(0)
    return mean


class PoincareLinear(nn.Module):
    """
    Linear layer in hyperbolic space.
    Maps from tangent space, applies linear transform, maps back.
    """

    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.c = c

        # Xavier-like init but scaled down for stability
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x on Poincaré ball → log map → linear → exp map → result on ball
        """
        h = logmap0(x, self.c)
        h = self.linear(h)
        return expmap0(h, self.c)
