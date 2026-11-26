"""Step 4 of the recipe: a domain gate that hands decisions unlike the training data back to the LLM.

Per decision the gate sets a weight ``w`` in [0, 1] and mixes the two answers::

    final answer = w * corrected answer + (1 - w) * LLM's answer,    w = sigmoid(weights . standardise(x) + bias)

``x`` holds six numbers: two distances of the answer vector (at two cached layers) from the training data's
**manifold**, the relative residual outside its top principal directions and the mean distance to its nearest training
decisions inside them, each relative to its typical value in training (both as logs); the LLM's normalised entropy and
top probability; the correction head's top probability; and how far the correction moves the answer (total variation).

The gate is a logistic model fitted on *honest* answers, from heads that never trained on the decision: in domain,
the cross-fitting halves on the half each did not train on; out of domain, a head answering decisions of a domain it
never saw. In-domain and out-of-domain examples weigh equally, and the fit minimises the log loss of the mixed answer.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

GATE_INPUTS = ("pca_residual", "knn_distance", "llm_entropy", "llm_top", "correction_top", "shift")


@dataclass(frozen=True)
class ManifoldConfig:
    """``layers``: indices into the cached layers whose answer vectors define the manifold."""

    layers: tuple[int, ...] = (1, 2)
    components: int = 64
    neighbours: int = 10
    reference_rows: int = 40_000


class Manifold:
    """Standardisation, principal directions and a nearest-neighbour reference set of training answer vectors.

    Args:
        vectors: ``(N, F)`` training answer vectors (the manifold layers, concatenated).
        config: sizes of the manifold.
        device: where to compute distances.
        seed: picks the reference sample when there are more than ``config.reference_rows`` vectors.
    """

    CHUNK = 2048

    def __init__(self, vectors: torch.Tensor, config: ManifoldConfig | None = None,
                 device: torch.device | str = "cpu", seed: int = 0):
        self.config = config = config or ManifoldConfig()
        generator = torch.Generator().manual_seed(seed)
        sample = vectors[torch.randperm(len(vectors), generator=generator)[: config.reference_rows]]
        sample = sample.float().to(device)
        self.mean, self.std = sample.mean(0), sample.std(0).clamp_min(1e-4)
        z = (sample - self.mean) / self.std
        _, eigenvectors = torch.linalg.eigh((z.T @ z / len(z)).cpu())
        self.components = eigenvectors[:, -config.components:].to(device)  # (F, P), largest last
        self.reference = z @ self.components  # (R, P)
        residual, distance = self._raw(z, exclude_self=True)
        self.residual_scale, self.distance_scale = residual.median(), distance.median()

    def scores(self, vectors: torch.Tensor) -> torch.Tensor:
        """``(N, 2)``: residual and neighbour distance, each relative to its typical value in the reference set."""
        z = (vectors.float().to(self.mean.device) - self.mean) / self.std
        residual, distance = self._raw(z)
        return torch.stack([residual / self.residual_scale, distance / self.distance_scale], 1).cpu()

    def _raw(self, z: torch.Tensor, exclude_self: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        residuals, distances = [], []
        for start in range(0, len(z), self.CHUNK):
            zc = z[start : start + self.CHUNK]
            inside = zc @ self.components
            residuals.append((zc - inside @ self.components.T).norm(dim=1) / zc.norm(dim=1).clamp_min(1e-6))
            d = torch.cdist(inside, self.reference)
            if exclude_self:
                d[torch.arange(len(zc)), torch.arange(start, start + len(zc))] = float("inf")
            distances.append(d.topk(self.config.neighbours, largest=False).values.mean(1))
        return torch.cat(residuals), torch.cat(distances)


class DomainGate(nn.Module):
    """The fitted gate with its manifold, as one module whose state is a set of buffers.

    Args:
        hidden_size: the LLM's hidden size.
        reference_rows: training decisions in the nearest-neighbour reference set.
        config: the manifold's layers and sizes.
    """

    def __init__(self, hidden_size: int, reference_rows: int, config: ManifoldConfig | None = None):
        super().__init__()
        self.config = config = config or ManifoldConfig()
        features = hidden_size * len(config.layers)
        shapes = {"mean": (features,), "std": (features,), "components": (features, config.components),
                  "reference": (reference_rows, config.components), "residual_scale": (),
                  "distance_scale": (), "weights": (len(GATE_INPUTS),), "bias": (),
                  "input_mean": (len(GATE_INPUTS),), "input_std": (len(GATE_INPUTS),)}
        for name, shape in shapes.items():
            self.register_buffer(name, torch.zeros(shape))

    @classmethod
    def from_fit(cls, manifold: Manifold, fit: "GateFit", hidden_size: int) -> "DomainGate":
        """A gate from a fitted manifold and logistic model."""
        gate = cls(hidden_size, len(manifold.reference), manifold.config)
        values = {"mean": manifold.mean, "std": manifold.std, "components": manifold.components,
                  "reference": manifold.reference, "residual_scale": manifold.residual_scale,
                  "distance_scale": manifold.distance_scale, "weights": fit.weights, "bias": fit.bias,
                  "input_mean": fit.input_mean, "input_std": fit.input_std}
        with torch.no_grad():
            for name, value in values.items():
                getattr(gate, name).copy_(value.float().cpu())
        return gate

    def forward(self, head_probs: torch.Tensor, llm_probs: torch.Tensor, answer_vectors: torch.Tensor,
                option_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(probs (N, K), weight (N,))``: the mixed answer and the gate weight on the corrected answer."""
        x = gate_inputs(self.manifold_scores(answer_vectors), head_probs, llm_probs, option_mask)
        weight = torch.sigmoid(((x - self.input_mean) / self.input_std) @ self.weights + self.bias)
        probs = weight[:, None] * head_probs + (1 - weight[:, None]) * llm_probs
        return probs.masked_fill(~option_mask, 0.0), weight

    @torch.no_grad()
    def manifold_scores(self, answer_vectors: torch.Tensor) -> torch.Tensor:
        """``(N, 2)`` residual and neighbour distance of the answer vectors, relative to training."""
        z = (answer_vectors[:, list(self.config.layers)].flatten(1).float() - self.mean) / self.std
        out = []
        for start in range(0, len(z), Manifold.CHUNK):
            zc = z[start : start + Manifold.CHUNK]
            inside = zc @ self.components
            residual = (zc - inside @ self.components.T).norm(dim=1) / zc.norm(dim=1).clamp_min(1e-6)
            distance = torch.cdist(inside, self.reference.float()).topk(self.config.neighbours, largest=False)
            out.append(torch.stack([residual / self.residual_scale,
                                    distance.values.mean(1) / self.distance_scale], 1))
        return torch.cat(out)


def gate_inputs(scores: torch.Tensor, head_probs: torch.Tensor, llm_probs: torch.Tensor,
                option_mask: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` the gate's inputs (:data:`GATE_INPUTS`) from manifold scores and the two answers."""
    k = option_mask.sum(-1).clamp_min(2).float()
    entropy = -(llm_probs.clamp_min(1e-9).log() * llm_probs).sum(-1) / k.log()
    shift = 0.5 * (head_probs - llm_probs).abs().sum(-1)
    return torch.cat([scores.clamp_min(1e-6).log(),
                      torch.stack([entropy, llm_probs.max(-1).values, head_probs.max(-1).values, shift], -1)], -1)


@dataclass
class GateExamples:
    """Honest answers for fitting the gate: manifold scores ``(n, 2)``, the head's, the LLM's and the gold
    distributions ``(n, K)`` (zero-padded), and the option mask."""

    scores: torch.Tensor
    head_probs: torch.Tensor
    llm_probs: torch.Tensor
    gold: torch.Tensor
    option_mask: torch.Tensor

    def inputs(self) -> torch.Tensor:
        return gate_inputs(self.scores, self.head_probs, self.llm_probs, self.option_mask)

    def padded_to(self, k: int) -> "GateExamples":
        pad = lambda t: nn.functional.pad(t, (0, k - t.shape[1]))  # noqa: E731
        return GateExamples(self.scores, pad(self.head_probs), pad(self.llm_probs), pad(self.gold),
                            pad(self.option_mask))


@dataclass
class GateFit:
    """The fitted logistic gate: weights and bias on standardised inputs, and the standardisation."""

    weights: torch.Tensor
    bias: torch.Tensor
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def weight(self, examples: GateExamples) -> torch.Tensor:
        return torch.sigmoid(((examples.inputs() - self.input_mean) / self.input_std) @ self.weights + self.bias)


def fit_gate(in_domain: list[GateExamples], out_of_domain: list[GateExamples], l2: float = 1e-3,
             iterations: int = 200) -> GateFit:
    """Fit the logistic gate by L-BFGS on the log loss of the mixed answer.

    The in-domain groups together weigh as much as the out-of-domain groups together; within each side every group
    weighs the same, and within a group every example."""
    groups = [(g, 1 / len(in_domain)) for g in in_domain] + [(g, 1 / len(out_of_domain)) for g in out_of_domain]
    k = max(g.gold.shape[1] for g, _ in groups)
    padded = [g.padded_to(k) for g, _ in groups]
    x = torch.cat([g.inputs() for g in padded])
    head = torch.cat([g.head_probs for g in padded])
    llm = torch.cat([g.llm_probs for g in padded])
    gold = torch.cat([g.gold for g in padded])
    sample_weight = torch.cat([torch.full((len(g.scores),), share / len(g.scores)) for g, share in groups])
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    xz = (x - mean) / std
    weights = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=iterations, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        w = torch.sigmoid(xz @ weights + bias)[:, None]
        mixed = w * head + (1 - w) * llm
        nll = -(gold * mixed.clamp_min(1e-9).log()).sum(1)
        loss = (nll * sample_weight).sum() / sample_weight.sum() + l2 * (weights ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return GateFit(weights.detach(), bias.detach(), mean, std)


def separation_auroc(out_scores: np.ndarray, in_scores: np.ndarray) -> float:
    """How well one distance tells out-of-domain from in-domain decisions (0.5 = not at all)."""
    s = np.concatenate([out_scores, in_scores])
    ranks = s.argsort().argsort() + 1
    n_out, n_in = len(out_scores), len(in_scores)
    return float((ranks[:n_out].sum() - n_out * (n_out + 1) / 2) / (n_out * n_in))
