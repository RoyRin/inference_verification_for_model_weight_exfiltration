"""
Convolved Gaussian Score (CGS) implementation.

This module implements the Circuit Genuineness Score (CGS) method, which verifies
tokens by computing the probability that a Gaussian-perturbed uniform random value
would fall into each token's CDF interval.

The key idea: Sample u ~ Uniform[0,1] deterministically from seed + past tokens,
then compute P(token_i is sampled | u ~ N(u_obs, sigma^2)) for each token.
"""

import torch
import xxhash

EPSILON = 1e-12


def get_seed(seed: int, past_tokens_ids: list[int]) -> int:
    """
    Generate a deterministic seed from base seed + past token history.

    This ensures that the random value u is deterministic and verifiable
    given the conversation history.

    Args:
        seed: Base random seed
        past_tokens_ids: List of previously generated token IDs

    Returns:
        int: Deterministic seed derived from base seed and token history
    """
    hasher = xxhash.xxh64(seed=seed)
    for token_id in past_tokens_ids:
        hasher.update(token_id.to_bytes(4, "little"))
    return hasher.intdigest()


def draw_u(seed: int, generator: torch.Generator) -> torch.Tensor:
    """
    Draw a deterministic uniform random number using the given seed.

    Args:
        seed: Random seed
        generator: PyTorch random generator

    Returns:
        torch.Tensor: Single uniform random value in [0, 1]
    """
    generator.manual_seed(seed)
    u = torch.rand(1, generator=generator, device=generator.device)
    return u


def compute_convolved_gaussian_score(
    cdf_V: torch.Tensor,
    u: torch.Tensor,
    sigma: float,
    epsilon: float = EPSILON
) -> torch.Tensor:
    """
    Compute CGS (Convolved Gaussian Score) for every token in vocabulary.

    Given:
    - CDF values for each token from the categorical distribution
    - A uniform random value u that was used for sampling
    - Gaussian noise parameter sigma

    Returns log probability that a Gaussian-perturbed u would fall into
    each token's CDF interval.

    The idea: If we assume u has Gaussian noise ~ N(u_obs, sigma^2),
    we can compute P(token_i is sampled | u_perturbed) for each token.

    Args:
        cdf_V: [V] CDF values for vocabulary (cumulative probabilities)
        u: Scalar uniform random value that was used for sampling
        sigma: Standard deviation of Gaussian noise on u
        epsilon: Small constant for numerical stability

    Returns:
        torch.Tensor [V]: Log-probabilities for each token
    """
    V = cdf_V.shape[0]

    # Build left and right endpoints of CDF intervals for each token
    # Token i corresponds to interval [CDF[i-1], CDF[i])
    cdf_left_all = torch.zeros_like(cdf_V)
    cdf_left_all[1:] = cdf_V[:-1]
    cdf_right_all = cdf_V

    # Create Gaussian distribution centered at observed u
    dist = torch.distributions.Normal(
        loc=u,
        scale=torch.tensor(sigma, device=cdf_V.device, dtype=cdf_V.dtype),
    )

    # Compute probability mass for each token's CDF interval
    # P(token_i) = P(cdf_left[i] <= u_perturbed < cdf_right[i])
    #            = Φ((cdf_right - u) / sigma) - Φ((cdf_left - u) / sigma)
    prob_mass_V = dist.cdf(cdf_right_all) - dist.cdf(cdf_left_all)
    log_scores_V = torch.log(prob_mass_V + epsilon)

    return log_scores_V
