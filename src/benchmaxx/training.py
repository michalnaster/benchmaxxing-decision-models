"""Step 5 of the recipe: train the correction head by cross-fitting, then fit the domain gate.

1. **Cross-fitting.** The training decisions are split by case into two halves (all decisions of a case stay
   together, and each dataset is spread evenly over both). Head A trains on one half and is early-stopped on the
   other by KL divergence, and head B the other way round, so no extra held-out data is needed.
2. **Refit.** One head trains on all the data for the average of the two halves' best epochs: the head you ship.
3. **Domain gate** (one-domain route). In domain, each half's head answers the half it did not train on; out of
   domain, the shipped head answers decisions of other domains. The gate is fitted on these honest answers.

Optimisation: AdamW, linear warm-up then cosine decay of the learning rate, gradient norm clipped at 1. Datasets can be
repeated within an epoch (``dataset_repeats``) so that small ones are not drowned out in a large mix, and early
stopping can average the KL per dataset (``early_stop="kl_per_dataset"``) for the same reason. Gold from generated
data, typically too sure of itself, can be softened towards uniform (``soften_gold``).
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from benchmaxx.correction_head import CorrectionHead, CorrectionHeadConfig
from benchmaxx.device import default_device
from benchmaxx.domain_gate import DomainGate, GateExamples, Manifold, ManifoldConfig, fit_gate
from benchmaxx.features import FeatureSet
from benchmaxx.losses import decision_loss
from benchmaxx.metrics import kl_divergence
from benchmaxx.model import Benchmaxx


@dataclass
class TrainingConfig:
    """Hyperparameters of the correction head's training (defaults: those of the benchmaxx-4B models)."""

    max_epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    patience: int = 10
    folds: int = 2
    early_stop: str = "kl"  # "kl", or "kl_per_dataset": the KL averaged per dataset, each counting once
    brier_weight: float = 1.0
    rps_weight: float = 1.0
    dataset_repeats: dict[str, int] = field(default_factory=dict)  # times a dataset's decisions appear per epoch
    soften_gold: dict[str, float] = field(default_factory=dict)  # dataset -> eps: gold <- (1 - eps) gold + eps / K
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.early_stop not in ("kl", "kl_per_dataset"):
            raise ValueError(f"unknown early-stopping metric {self.early_stop!r}")


@dataclass
class CrossFitResult:
    """The shipped head, the cross-fitting halves and the rows each half was early-stopped on (never trained on)."""

    head: CorrectionHead
    members: list[CorrectionHead]
    held_out_rows: list[np.ndarray]
    best_epochs: list[int]


def train(features: FeatureSet, config: TrainingConfig | None = None,
          head_config: CorrectionHeadConfig | None = None,
          out_of_domain: dict[str, list[FeatureSet]] | None = None) -> Benchmaxx:
    """Train a benchmaxx model on ``features``; with ``out_of_domain`` (group name -> feature sets of domains the
    model will not train on) also fit its domain gate."""
    config = config or TrainingConfig()
    if not features.labelled:
        raise ValueError("training needs gold for every decision")
    features = soften_gold(features, config.soften_gold)
    head_config = head_config or CorrectionHeadConfig(hidden_size=features.answer.shape[-1],
                                                      num_layers=features.answer.shape[1])
    result = cross_fit(features, config, head_config)
    gate = fit_domain_gate(features, result, out_of_domain, device=_device(config)) if out_of_domain else None
    return Benchmaxx(result.head.cpu(), gate, features.llm or None)


def cross_fit(features: FeatureSet, config: TrainingConfig, head_config: CorrectionHeadConfig) -> CrossFitResult:
    """Cross-fitting halves, then the refit on all the data (module docstring, steps 1 and 2)."""
    folds = case_folds([r["case_id"] for r in features.records], features.datasets.tolist(), config.folds)
    held_out = [np.nonzero(folds == k)[0] for k in range(config.folds)]
    members, best_epochs = [], []
    for k, rows in enumerate(held_out):
        rest = np.setdiff1d(np.arange(len(features)), rows)
        head, best = train_head(features.subset(rest), features.subset(rows), config, head_config,
                                epochs=config.max_epochs, seed=config.seed + k, description=f"half {k + 1}")
        members.append(head)
        best_epochs.append(best)
    epochs = int(np.median(best_epochs))
    head, _ = train_head(features, None, config, head_config, epochs=epochs, seed=config.seed + 100,
                         description="refit on all data")
    return CrossFitResult(head, members, held_out, best_epochs)


def train_head(fit: FeatureSet, validation: FeatureSet | None, config: TrainingConfig,
               head_config: CorrectionHeadConfig, epochs: int, seed: int,
               description: str = "training") -> tuple[CorrectionHead, int]:
    """Train one head on ``fit``. With ``validation``, keep the weights of the epoch with the lowest validation KL and
    stop after ``config.patience`` epochs without improvement; without it, train for exactly ``epochs``.
    Returns the head and its best epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device(config)
    head = CorrectionHead(head_config)
    head.fit_input_statistics(fit)
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    pool = torch.as_tensor(np.repeat(np.arange(len(fit)), _repeats(fit, config.dataset_repeats)))
    steps_per_epoch = math.ceil(len(pool) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(config.warmup_epochs * steps_per_epoch, epochs * steps_per_epoch))
    best_score, best_epoch, best_state, stale = math.inf, epochs, None, 0
    bar = tqdm(range(1, epochs + 1), desc=description, unit="epoch")
    for epoch in bar:
        head.train()
        for idx in pool[torch.randperm(len(pool))].split(config.batch_size):
            batch = fit.batch(idx).to(device)
            loss = decision_loss(head(batch).probs, batch.gold, batch.option_mask, batch.qtype,
                                 config.brier_weight, config.rps_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        if validation is None:
            continue
        score = validation_kl(head, validation, per_dataset=config.early_stop == "kl_per_dataset")
        bar.set_postfix(kl=f"{score:.4f}", best=best_epoch if best_state else epoch)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head.eval(), best_epoch


def fit_domain_gate(features: FeatureSet, result: CrossFitResult, out_of_domain: dict[str, list[FeatureSet]],
                    manifold_config: ManifoldConfig | None = None, in_rows: int = 24_000,
                    out_rows: int = 8_000, device: torch.device | str = "cpu", seed: int = 0) -> DomainGate:
    """Fit the domain gate on honest answers (module docstring, step 3).

    In domain, up to ``in_rows`` decisions in all, from each half's held-out rows, answered by the head that never
    trained on them and scored against the manifold of that head's own training rows. Out of domain, up to
    ``out_rows`` decisions per group, answered by the shipped head and scored against the manifold of all the
    training data, which is also the gate's deployed reference."""
    manifold_config = manifold_config or ManifoldConfig()
    rng = np.random.default_rng(seed)
    vectors = features.answer_layers(manifold_config.layers)
    in_domain = []
    for k, (member, rows) in enumerate(zip(result.members, result.held_out_rows, strict=True)):
        rows = rng.choice(rows, min(in_rows // len(result.members), len(rows)), replace=False)
        own = np.concatenate([r for j, r in enumerate(result.held_out_rows) if j != k])
        manifold = Manifold(vectors[torch.as_tensor(own)], manifold_config, device)
        in_domain.append(gate_examples(member, features, rows, manifold, device))
    manifold = Manifold(vectors, manifold_config, device)
    out = []
    for sets in out_of_domain.values():
        parts = []
        for other in sets:
            rows = np.sort(rng.choice(len(other), min(out_rows // len(sets), len(other)), replace=False))
            parts.append(gate_examples(result.head, other, rows, manifold, device))
        out.append(_concat(parts))
    fit = fit_gate(in_domain, out)
    return DomainGate.from_fit(manifold, fit, features.answer.shape[-1])


@torch.no_grad()
def gate_examples(head: CorrectionHead, features: FeatureSet, rows: np.ndarray, manifold: Manifold,
                  device: torch.device | str, batch_size: int = 2048) -> GateExamples:
    """The head's and the LLM's answers on ``rows``, with their manifold scores, for fitting the gate."""
    head = head.to(device).eval()
    vectors = features.answer_layers(manifold.config.layers)
    parts = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = features.batch(chunk).to(device)
        parts.append(GateExamples(scores=manifold.scores(vectors[torch.as_tensor(chunk)]),
                                  head_probs=head(batch).probs.cpu(), llm_probs=batch.llm_probs.cpu(),
                                  gold=batch.gold.cpu(), option_mask=batch.option_mask.cpu()))
    return _concat(parts)


@torch.no_grad()
def validation_kl(head: CorrectionHead, features: FeatureSet, per_dataset: bool = False,
                  batch_size: int = 512) -> float:
    """Mean KL(gold || prediction) of ``head`` on ``features``, optionally averaged per dataset."""
    head.eval()
    device = next(head.parameters()).device
    probs, gold = [], []
    for start in range(0, len(features), batch_size):
        batch = features.batch(np.arange(start, min(start + batch_size, len(features)))).to(device)
        p = head(batch).probs.cpu().numpy()
        g = batch.gold.cpu().numpy()
        for i, k in enumerate(batch.option_mask.sum(-1).tolist()):
            probs.append(p[i, :k])
            gold.append(g[i, :k])
    if not per_dataset:
        return kl_divergence(probs, gold)
    names = features.datasets
    return float(np.mean([kl_divergence([probs[i] for i in np.nonzero(names == n)[0]],
                                        [gold[i] for i in np.nonzero(names == n)[0]]) for n in np.unique(names)]))


def case_folds(case_ids: list[str], datasets: list[str], folds: int, seed: int = 0) -> np.ndarray:
    """``(N,)`` the fold of every decision: cases are dealt round-robin within each dataset, in a seeded hash order,
    so a case never spans two folds and every dataset is spread evenly."""
    by_dataset: dict[str, set[str]] = {}
    for case_id, dataset in zip(case_ids, datasets, strict=True):
        by_dataset.setdefault(dataset, set()).add(case_id)
    fold_of: dict[str, int] = {}
    for ids in by_dataset.values():
        for rank, case_id in enumerate(sorted(ids, key=lambda c: _stable_unit(c, seed))):
            fold_of[case_id] = rank % folds
    return np.array([fold_of[c] for c in case_ids])


def soften_gold(features: FeatureSet, eps_by_dataset: dict[str, float]) -> FeatureSet:
    """A copy with the gold of the given datasets mixed towards uniform: ``(1 - eps) gold + eps / K``."""
    if not eps_by_dataset:
        return features
    gold = features.gold.clone()
    for i, record in enumerate(features.records):
        if (eps := eps_by_dataset.get(record.get("dataset", ""), 0.0)) > 0:
            a, b = int(features.offsets[i]), int(features.offsets[i + 1])
            gold[a:b] = (1 - eps) * gold[a:b] + eps / (b - a)
    return FeatureSet(features.answer, features.options, features.order_probs, features.offsets, features.qtype,
                      gold, features.records, features.llm)


def warmup_cosine(warmup_steps: int, total_steps: int):
    """Learning-rate factor: linear warm-up over ``warmup_steps``, then cosine decay to zero at ``total_steps``."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))))
    return factor


def _repeats(features: FeatureSet, dataset_repeats: dict[str, int]) -> np.ndarray:
    return np.array([max(1, dataset_repeats.get(r.get("dataset", ""), 1)) for r in features.records])


def _stable_unit(key: str, seed: int) -> float:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64


def _concat(parts: list[GateExamples]) -> GateExamples:
    k = max(p.gold.shape[1] for p in parts)
    parts = [p.padded_to(k) for p in parts]
    return GateExamples(*(torch.cat([getattr(p, name) for p in parts])
                          for name in ("scores", "head_probs", "llm_probs", "gold", "option_mask")))


def _device(config: TrainingConfig) -> torch.device:
    return torch.device(config.device) if config.device else default_device()
