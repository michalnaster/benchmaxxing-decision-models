"""What the frozen LLM leaves behind for each decision, stored once and read by every training step.

Per decision, at each cached layer:

* the **answer vector**: the hidden state at the last prompt token, ``(L, D)``;
* one **option vector** per option: the hidden states averaged over the option's tokens, ``(K, L, D)``;
* the LLM's **answer in each reading**: its probability for every option code, renormalised over the options,
  ``(K, 2)`` (column 0 the displayed order, column 1 the reversed one).

Vectors are the mean of the two readings, stored in float16. A :class:`FeatureSet` keeps a whole dataset in memory
with options stored ragged; :meth:`FeatureSet.batch` pads a selection of decisions into a :class:`FeatureBatch`, the
input of the correction head.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from benchmaxx.schema import QuestionType

SCORE = QuestionType.SCORE.code


@dataclass
class FeatureBatch:
    """Padded features of ``N`` decisions with at most ``K`` options; padding is masked out by ``option_mask``.

    Attributes:
        option_vectors: ``(N, K, L, D)`` the option vectors.
        answer_vectors: ``(N, L, D)`` the answer vectors.
        order_probs: ``(N, 2, K)`` the LLM's answer in each reading, zero on padding.
        option_mask: ``(N, K)`` true for real options.
        qtype: ``(N,)`` question type codes (:attr:`QuestionType.code`).
        gold: ``(N, K)`` gold distributions, or ``None`` when unlabelled.
    """

    option_vectors: torch.Tensor
    answer_vectors: torch.Tensor
    order_probs: torch.Tensor
    option_mask: torch.Tensor
    qtype: torch.Tensor
    gold: torch.Tensor | None = None

    @property
    def llm_probs(self) -> torch.Tensor:
        """``(N, K)`` the LLM's answer: the mean of its two readings."""
        return self.order_probs.mean(1)

    def to(self, device: torch.device | str) -> "FeatureBatch":
        """The batch on ``device``, vectors in float32."""
        return FeatureBatch(
            option_vectors=self.option_vectors.to(device, torch.float32),
            answer_vectors=self.answer_vectors.to(device, torch.float32),
            order_probs=self.order_probs.to(device), option_mask=self.option_mask.to(device),
            qtype=self.qtype.to(device), gold=None if self.gold is None else self.gold.to(device))


@dataclass
class FeatureSet:
    """Features of a dataset, with options stored ragged: the options of decision ``i`` are rows
    ``offsets[i]:offsets[i + 1]`` of ``options``, ``order_probs`` and ``gold``.

    Attributes:
        answer: ``(N, L, D)`` float16 answer vectors.
        options: ``(sum K, L, D)`` float16 option vectors.
        order_probs: ``(sum K, 2)`` the LLM's answer in each reading.
        offsets: ``(N + 1,)`` option offsets.
        qtype: ``(N,)`` question type codes.
        gold: ``(sum K,)`` gold probabilities, NaN for unlabelled decisions.
        records: per decision, ``case_id``, ``dataset`` and ``option_keys``.
        llm: the LLM and layers the features were read with.
    """

    answer: torch.Tensor
    options: torch.Tensor
    order_probs: torch.Tensor
    offsets: torch.Tensor
    qtype: torch.Tensor
    gold: torch.Tensor
    records: list[dict]
    llm: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_options(self) -> torch.Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def labelled(self) -> bool:
        return bool(len(self.gold)) and not torch.isnan(self.gold).any().item()

    @property
    def datasets(self) -> np.ndarray:
        return np.array([r.get("dataset", "") for r in self.records])

    def batch(self, rows: np.ndarray | list[int] | torch.Tensor) -> FeatureBatch:
        """The decisions ``rows`` padded to their largest option count."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        sizes = self.num_options[rows]
        n, k = len(rows), int(sizes.max())
        _, layers, dim = self.options.shape
        option_vectors = torch.zeros(n, k, layers, dim, dtype=self.options.dtype)
        order_probs = torch.zeros(n, 2, k)
        gold = torch.zeros(n, k)
        mask = torch.arange(k) < sizes[:, None]
        for i, row in enumerate(rows.tolist()):
            a, b = int(self.offsets[row]), int(self.offsets[row + 1])
            option_vectors[i, : b - a] = self.options[a:b]
            order_probs[i, :, : b - a] = self.order_probs[a:b].T
            gold[i, : b - a] = self.gold[a:b]
        labelled = not torch.isnan(gold).any()
        return FeatureBatch(option_vectors=option_vectors, answer_vectors=self.answer[rows], order_probs=order_probs,
                            option_mask=mask, qtype=self.qtype[rows], gold=gold if labelled else None)

    def subset(self, rows: np.ndarray | list[int]) -> "FeatureSet":
        """A new set holding the decisions ``rows``, in that order."""
        rows = [int(r) for r in rows]
        spans = [torch.arange(int(self.offsets[r]), int(self.offsets[r + 1])) for r in rows]
        flat = torch.cat(spans) if spans else torch.zeros(0, dtype=torch.long)
        sizes = self.num_options[rows] if rows else torch.zeros(0, dtype=torch.long)
        return FeatureSet(answer=self.answer[rows], options=self.options[flat], order_probs=self.order_probs[flat],
                          offsets=torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]),
                          qtype=self.qtype[rows], gold=self.gold[flat], records=[self.records[r] for r in rows],
                          llm=self.llm)

    @classmethod
    def concat(cls, sets: list["FeatureSet"]) -> "FeatureSet":
        """One set from several read with the same LLM and layers."""
        if any(s.llm != sets[0].llm for s in sets):
            raise ValueError("feature sets were read with different LLMs or layers")
        sizes = torch.cat([s.num_options for s in sets])
        return cls(answer=torch.cat([s.answer for s in sets]), options=torch.cat([s.options for s in sets]),
                   order_probs=torch.cat([s.order_probs for s in sets]),
                   offsets=torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]),
                   qtype=torch.cat([s.qtype for s in sets]), gold=torch.cat([s.gold for s in sets]),
                   records=[r for s in sets for r in s.records], llm=sets[0].llm)

    def answer_layers(self, layers: tuple[int, ...]) -> torch.Tensor:
        """``(N, len(layers) * D)`` float16: the answer vector at the given cached-layer indices, concatenated."""
        return self.answer[:, list(layers)].flatten(1)

    def save(self, directory: str | Path) -> None:
        """Write ``features.safetensors`` and ``records.json`` to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {"answer": self.answer, "options": self.options, "order_probs": self.order_probs,
                   "offsets": self.offsets, "qtype": self.qtype, "gold": self.gold}
        save_file({k: v.contiguous() for k, v in tensors.items()}, directory / "features.safetensors")
        (directory / "records.json").write_text(json.dumps({"llm": self.llm, "records": self.records}))

    @classmethod
    def load(cls, directory: str | Path) -> "FeatureSet":
        directory = Path(directory)
        tensors = load_file(directory / "features.safetensors")
        meta = json.loads((directory / "records.json").read_text())
        return cls(**tensors, records=meta["records"], llm=meta["llm"])
