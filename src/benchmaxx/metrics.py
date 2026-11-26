"""Evaluation metrics for distributions over options, against soft gold.

* **accuracy**: the predicted top option is the gold's top option (ties in the gold go to the first tied option).
* **KL(gold || prediction)**: rewards getting the whole distribution right.
* **ECE**: expected calibration error of the top probability against accuracy, 15 equal-width bins.
* **AURC**: area under the risk-coverage curve when answers are kept from the most to the least confident; lower means
  the most confident answers are more often right.
"""

from collections.abc import Sequence

import numpy as np

ECE_BINS = 15


def accuracy(probs: Sequence[np.ndarray], gold: Sequence[np.ndarray]) -> float:
    return float(np.mean([np.argmax(p) == np.argmax(g) for p, g in zip(probs, gold, strict=True)]))


def kl_divergence(probs: Sequence[np.ndarray], gold: Sequence[np.ndarray]) -> float:
    """Mean KL(gold || prediction) in nats, both clipped at 1e-12."""
    values = [float((g * (np.log(np.clip(g, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0)))).sum())
              for p, g in zip(probs, gold, strict=True)]
    return float(np.mean(values))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    return float(sum((which == b).mean() * abs(confidence[which == b].mean() - correct[which == b].mean())
                     for b in range(bins) if (which == b).any()))


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Mean error rate of the kept answers over all coverages, keeping the most confident first."""
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    return float((np.cumsum(errors) / np.arange(1, len(order) + 1)).mean())


def evaluate(probs: Sequence[np.ndarray], gold: Sequence[np.ndarray]) -> dict[str, float]:
    """Accuracy, KL, ECE and AURC of predictions against gold, one array per decision."""
    confidence = np.array([float(np.max(p)) for p in probs])
    correct = np.array([float(np.argmax(p) == np.argmax(g)) for p, g in zip(probs, gold, strict=True)])
    return {"n": len(probs), "accuracy": float(correct.mean()), "kl": kl_divergence(probs, gold),
            "ece": expected_calibration_error(confidence, correct), "aurc": aurc(confidence, correct)}
