"""The training loss: proper scoring rules against the soft gold.

* choice and yes/no questions: log loss + Brier score
* score questions: log loss + ranked probability score, so an answer one level off costs less than one three levels off

Every rule is minimised in expectation by reporting the true distribution, so the head learns how sure to be, not
only which option wins.
"""

import torch

from benchmaxx.features import SCORE


def log_loss(probs: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """``(N,)`` cross-entropy of the prediction under the gold distribution."""
    return -(gold * probs.clamp_min(1e-8).log()).sum(-1)


def brier_score(probs: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """``(N,)`` squared distance between the prediction and the gold distribution."""
    return (probs - gold).pow(2).sum(-1)


def ranked_probability_score(probs: torch.Tensor, gold: torch.Tensor, num_levels: torch.Tensor) -> torch.Tensor:
    """``(N,)`` squared gaps between the predicted and gold CDFs, averaged over the ``K - 1`` thresholds.

    Padding needs no mask: both CDFs reach 1 at the last real level, so the gaps beyond it are zero."""
    return (probs.cumsum(-1) - gold.cumsum(-1)).pow(2).sum(-1) / (num_levels - 1)


def decision_loss(probs: torch.Tensor, gold: torch.Tensor, option_mask: torch.Tensor, qtype: torch.Tensor,
                  brier_weight: float = 1.0, rps_weight: float = 1.0) -> torch.Tensor:
    """The mean loss of a batch: log loss plus the Brier score, or plus the RPS on score questions."""
    num_levels = option_mask.sum(-1).clamp_min(2)
    is_score = qtype == SCORE
    extra = torch.where(is_score, rps_weight * ranked_probability_score(probs, gold, num_levels),
                        brier_weight * brier_score(probs, gold))
    return (log_loss(probs, gold) + extra).mean()
