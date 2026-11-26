"""Step 3 of the recipe: start from the LLM's answer and learn a gated correction to it.

Every option becomes a token: its option vector, projected to the head's width, plus an embedding of five numbers on
how the LLM rated it. The answer vector becomes one token per cached layer (*attention over depth*), so each option's
attention, in every head and block, chooses which layers to read. Options never attend to each other. Each option then
receives a correction ``delta_k``, and a small gate on statistics of the LLM's answer decides how much of it to apply::

    o_k   = Projector(option vector k) + Linear(rating_k)       rating: log p_fwd, log p_rev, p, rank, position
    c_l   = LayerNorm(W z(answer vector at layer l)) + e_layer_l + e_answer
    o_k  <- CrossBlock(o_k, [c_1 .. c_L])   x depth
    s_k   = a log p_fwd(k) + b log p_rev(k) + g(statistics) * delta_k      a = b = 1/2 at the start
    p     = softmax_k(s_k)

``W`` and the standardisation ``z`` are shared by option vectors and answer-vector tokens, so both land in the same
space. The gate starts almost closed (``g`` about 0.02), so before training the head *is* the LLM.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from benchmaxx.features import SCORE, FeatureBatch, FeatureSet

STATISTICS = 11  # inputs of the correction gate, see answer_statistics


@dataclass(frozen=True)
class CorrectionHeadConfig:
    """Shape of a correction head. ``hidden_size`` and ``num_layers`` describe the LLM's cached vectors."""

    hidden_size: int = 2560
    num_layers: int = 4
    width: int = 256
    depth: int = 2
    num_heads: int = 4
    dropout: float = 0.15
    gate_hidden: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeadOutput:
    """``probs (N, K)``: the corrected answer, zero on padding; ``gate (N,)``: how much correction was applied."""

    probs: torch.Tensor
    gate: torch.Tensor


class Projector(nn.Module):
    """Standardise each (layer, dimension) of a cached vector, then project it to the head's width.

    LLM hidden states carry a few dimensions with enormous values that would dominate a plain LayerNorm, so each is
    first z-scored with statistics of the training vectors (fixed buffers, set by :meth:`set_statistics`).
    """

    def __init__(self, num_layers: int, in_dim: int, out_dim: int):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(num_layers, in_dim))
        self.register_buffer("feat_std", torch.ones(num_layers, in_dim))
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))  # softmax: a learned blend of the layers
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / self.feat_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., L, D) -> (..., d)``: standardise, blend the layers, project."""
        blended = torch.einsum("...ld,l->...d", self.standardise(x), self.layer_weights.softmax(0))
        return self.norm(self.proj(blended))

    def per_layer(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., L, D) -> (..., L, d)``: every layer projected on its own, with the same weights."""
        return self.norm(self.proj(self.standardise(x)))

    @torch.no_grad()
    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-3))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention of the option tokens over the context tokens, then a feed-forward layer:
    ``(N, K, d), (N, S, d) -> (N, K, d)``."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, options: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(context)
        read, _ = self.attn(self.norm_q(options), kv, kv, need_weights=False)
        options = options + read
        return options + self.ff(self.norm_ff(options))


class CorrectionHead(nn.Module):
    """The correction head (module docstring). Reads a :class:`~benchmaxx.features.FeatureBatch`."""

    def __init__(self, config: CorrectionHeadConfig):
        super().__init__()
        self.config = config
        d = config.width
        self.projector = Projector(config.num_layers, config.hidden_size, d)
        self.layer_embedding = nn.Parameter(torch.zeros(config.num_layers, d))
        self.answer_embedding = nn.Parameter(torch.randn(d))  # marks the answer-vector tokens
        self.rating_embedding = nn.Linear(5, d)
        self.blocks = nn.ModuleList(CrossBlock(d, config.num_heads, config.dropout) for _ in range(config.depth))
        self.correction = nn.Linear(d, 1)
        self.gate = nn.Sequential(nn.Linear(STATISTICS, config.gate_hidden), nn.GELU(),
                                  nn.Linear(config.gate_hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)  # sigmoid(-4) ~ 0.02: start at the LLM's own answer
        self.order_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # a, b: the weights of the two readings

    def forward(self, batch: FeatureBatch) -> HeadOutput:
        mask = batch.option_mask
        orders = batch.order_probs  # (N, 2, K)
        p = orders.mean(1)
        log_orders = orders.clamp_min(1e-9).log()
        options = self.rating_embedding(option_ratings(p, log_orders, mask, batch.qtype))
        options = options + self.projector(batch.option_vectors)  # (N, K, d)
        context = self.projector.per_layer(batch.answer_vectors) + self.layer_embedding + self.answer_embedding
        for block in self.blocks:
            options = block(options, context)
        delta = self.correction(options).squeeze(-1)  # (N, K)
        gate = torch.sigmoid(self.gate(answer_statistics(p, orders, mask, batch.qtype))).squeeze(-1)  # (N,)
        a, b = self.order_weights
        scores = a * log_orders[:, 0] + b * log_orders[:, 1] + gate[:, None] * delta
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        return HeadOutput(probs=probs, gate=gate)

    def fit_input_statistics(self, features: FeatureSet, chunk: int = 4096) -> None:
        """Set the projector's standardisation from training vectors: answer vectors and real option vectors."""
        total = torch.zeros(features.answer.shape[1:], dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0
        for source in (features.answer, features.options):
            for start in range(0, len(source), chunk):
                x = source[start : start + chunk].double()
                total += x.sum(0)
                squares += (x * x).sum(0)
                count += len(x)
        mean = total / count
        std = ((squares - count * mean * mean) / max(count - 1, 1)).clamp_min(0).sqrt()
        self.projector.set_statistics(mean.float(), std.float())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def option_ratings(p: torch.Tensor, log_orders: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
    """``(N, K, 5)`` how the LLM rated each option: its log-probability in each reading, its mean probability, its
    rank (0 = the LLM's answer) scaled to [0, 1], and for score questions its position on the scale."""
    k = mask.shape[1]
    rank = p.masked_fill(~mask, -1.0).argsort(-1, descending=True).argsort(-1).float()
    k_real = mask.sum(-1, keepdim=True).clamp_min(2).float()
    position = torch.arange(k, device=mask.device).float()[None] / (k_real - 1)
    ordered = (qtype == SCORE).float()[:, None]
    return torch.stack([log_orders[:, 0], log_orders[:, 1], p, rank / (k_real - 1), ordered * position], -1)


def answer_statistics(p: torch.Tensor, orders: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
    """``(N, 11)`` how the LLM answered: normalised entropy, top probability, margin to the second option, how far the
    two readings differ (total variation) and whether they agree, the chosen option's probability in each reading,
    log K, whether the question is a score, and two question-type flags (choice, yes/no)."""
    k = mask.sum(-1).clamp_min(2).float()
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / k.log()
    top2 = p.masked_fill(~mask, -1.0).topk(2, -1)
    chosen = top2.indices[:, :1]
    gap = 0.5 * (orders[:, 0] - orders[:, 1]).abs().sum(-1)
    agree = (orders[:, 0].argmax(-1) == orders[:, 1].argmax(-1)).float()
    chosen_in_orders = orders.gather(2, chosen[:, None, :].expand(-1, 2, 1)).squeeze(-1)
    ordered = (qtype == SCORE).float()
    margin = top2.values[:, 0] - top2.values[:, 1]
    return torch.cat([torch.stack([entropy, top2.values[:, 0], margin, gap, agree], -1), chosen_in_orders,
                      k.log()[:, None], ordered[:, None], F.one_hot(qtype, 3).float()[:, :2]], -1)
