"""ChessFly-style rate dynamics on a frozen signed graph.

h <- (1-a) h + a relu(gamma (W h + u - mu) / sigma + beta)
W_ij = sign_ij * exp(theta_ij)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_apply(h, src, dst, w, n):
    """y[dst] += w * h[src]  (batched: h is [B, N])."""
    contrib = h.index_select(-1, src) * w
    y = h.new_zeros(h.shape[0], n) if h.dim() == 2 else h.new_zeros(n)
    if h.dim() == 2:
        y.scatter_add_(1, dst.unsqueeze(0).expand(h.shape[0], -1), contrib)
    else:
        y.scatter_add_(0, dst, contrib)
    return y


def settle(h, u, src, dst, log_gain, sign, n, steps=5, a=0.5, gamma=None, mu=None, sigma=None, beta=None):
    w = sign * torch.exp(log_gain)
    if gamma is None:
        gamma = torch.ones(n, device=h.device, dtype=h.dtype)
    if mu is None:
        mu = torch.zeros(n, device=h.device, dtype=h.dtype)
    if sigma is None:
        sigma = torch.ones(n, device=h.device, dtype=h.dtype)
    if beta is None:
        beta = torch.zeros(n, device=h.device, dtype=h.dtype)
    for _ in range(steps):
        wh = sparse_apply(h, src, dst, w, n)
        pre = gamma * (wh + u - mu) / sigma.clamp(min=1e-3) + beta
        h = (1 - a) * h + a * F.relu(pre)
    return h
