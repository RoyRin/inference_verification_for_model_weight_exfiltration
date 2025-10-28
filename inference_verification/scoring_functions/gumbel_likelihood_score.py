"""
Gumbel Likelihood Score (GLS) implementation.

This module implements the Gumbel-max verification scoring method, which computes
the log-probability that a claimed token would be beaten by the competitor token
under a Gaussian noise model on the logits.

The key idea: If we assume true logits have Gaussian noise around the observed logits,
we can compute P(competitor beats claimed token | observed logits, noise model).
"""

import torch
from typing import Optional

EPSILON = 1e-12


def exponential_to_gumbel(exponential_noise: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """Convert exponential noise E ~ Exp(1) to Gumbel noise G = -log(E).

    Args:
        exponential_noise: Tensor of exponential random variables
        epsilon: Small constant to prevent log(0)

    Returns:
        Gumbel noise tensor with same shape as input
    """
    return -torch.log(exponential_noise.clamp(min=epsilon))


def compute_gumbel_likelihood_score(
    logits_V: torch.Tensor,
    exponential_noise_V: torch.Tensor,
    temperature: float,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    gold_idx: torch.Tensor,
    noise_sigma: float,
    apply_top_k_top_p_fn,
    epsilon: float = EPSILON,
) -> float:
    """
    Compute the Gumbel Likelihood Score (GLS) for a single token.

    This computes the log-probability that the gold token gets beaten by the best token,
    given a Gaussian noise model on the logits.

    Assumes: logits_true = logits_observed + noise, where noise ~ N(0, sigma^2)

    Returns: log P(max_token beats gold_token | observed logits, noise model)

    This is the probability that, under the noise model, the true max token's logit
    is higher than the true gold token's logit.

    Args:
        logits_V: [V] observed logits before sampling
        exponential_noise_V: [V] Exp(1) draws used for Gumbel-Max sampling
        temperature: sampling temperature
        top_k: tensor([k]) top-k parameter
        top_p: tensor([p]) top-p (nucleus) parameter
        gold_idx: tensor with index of claimed token
        noise_sigma: std of Gaussian logit noise for the noise model
        apply_top_k_top_p_fn: function to apply top-k/top-p filtering
        epsilon: small constant for numerical stability

    Returns:
        float: log-probability that competitor beats gold token
    """
    # Apply temperature and top-k/top-p filtering to get valid token set
    temp_logits_V = logits_V.clone()
    if temperature > 0.0:
        temp_logits_V = temp_logits_V / temperature

    temp_logits_V = apply_top_k_top_p_fn(temp_logits_V[None, :], top_k, top_p).squeeze()
    neg_inf_mask = ~torch.isfinite(temp_logits_V)

    # Apply Gumbel-max sampling to find the competitor (best token)
    gumbel_noise = exponential_to_gumbel(exponential_noise_V.float(), epsilon)
    logits_V = logits_V + (gumbel_noise * temperature)
    logits_V[neg_inf_mask] = float("-inf")

    max_token = logits_V.argmax(dim=-1)

    # Compute difference in logits between competitor and claimed token
    logit_diff = logits_V[max_token] - logits_V[gold_idx]

    # Under Gaussian noise model: diff_true ~ N(logit_diff, 2*sigma^2)
    # (variance adds when subtracting two independent Gaussians)
    # P(gold gets beaten) = P(diff_true > 0) = P(Z > -logit_diff / sqrt(2*sigma^2))

    std_dev = noise_sigma * (2**0.5)  # sqrt(2) * sigma

    z_score = -logit_diff / (std_dev + epsilon)

    # Compute log CDF (log-probability)
    prob = torch.special.log_ndtr(z_score.to(torch.float64)).to(torch.float32)

    return float(prob.item())


def compute_gumbel_likelihood_score_batch(
    logits_V: torch.Tensor,
    exponential_noise_V: torch.Tensor,
    temperature: float,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
    gold_idx_list: list[int] | torch.Tensor,
    noise_sigma: float,
    apply_top_k_top_p_fn,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """
    Vectorized batch version of compute_gumbel_likelihood_score.

    Computes GLS scores for multiple tokens at once (batched over vocabulary dimension).

    Args:
        logits_V: [V] observed logits before sampling
        exponential_noise_V: [V] Exp(1) draws used for Gumbel-Max
        temperature: sampling temperature
        top_k: tensor([k]) or None
        top_p: tensor([p]) or None
        gold_idx_list: list or 1D tensor of claimed token indices
        noise_sigma: std of Gaussian logit noise
        apply_top_k_top_p_fn: function to apply top-k/top-p filtering
        epsilon: small constant for numerical stability

    Returns:
        1D torch.Tensor of log-probabilities:
        log P(competitor beats gold | observed logits, noise model)
    """
    # Build the top-k/top-p mask once
    temp_logits = logits_V.clone()
    if temperature > 0.0:
        temp_logits = temp_logits / temperature
    masked = apply_top_k_top_p_fn(temp_logits[None, :], top_k, top_p).squeeze()
    neg_inf_mask = ~torch.isfinite(masked)

    # Gumbel-Max step under shared seed
    gumbel_noise = exponential_to_gumbel(exponential_noise_V, epsilon)
    logits_perturbed = logits_V + (gumbel_noise * temperature)
    logits_perturbed[neg_inf_mask] = float("-inf")

    # Observed best token (competitor)
    max_token = int(logits_perturbed.argmax(dim=-1).item())
    competitor_logit = logits_perturbed[max_token]

    # Gather gold logits
    gold_idx = torch.as_tensor(gold_idx_list, device=logits_V.device, dtype=torch.long)
    gold_logits = logits_perturbed.index_select(0, gold_idx)

    # Gaussian noise model across gold vs competitor
    # diff_true ~ N(logit_diff, 2*sigma^2)
    logit_diff = competitor_logit - gold_logits
    std_dev = noise_sigma * (2**0.5)
    z = -logit_diff / (std_dev)

    # Same output convention as scalar function: log CDF
    log_probs = torch.special.log_ndtr(z.to(torch.float64)).to(torch.float32)

    return log_probs
