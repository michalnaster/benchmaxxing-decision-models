"""A benchmaxx model: a correction head on a frozen LLM's features, optionally behind a domain gate.

The final answer's top probability is its **confidence**: how much of the gold's probability the model expects on its
answer. No separate confidence model or temperature is fitted afterwards.

A saved model is a directory::

    config.json                  the LLM, its cached layers and the head's shape
    correction_head.safetensors  the correction head
    domain_gate.safetensors      the domain gate (absent for a model without one)
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from benchmaxx.correction_head import CorrectionHead, CorrectionHeadConfig
from benchmaxx.domain_gate import DomainGate, ManifoldConfig
from benchmaxx.features import FeatureBatch, FeatureSet
from benchmaxx.schema import Decision

if TYPE_CHECKING:
    from benchmaxx.reader import LLMReader

DEFAULT_LLM = {"model": "Qwen/Qwen3.5-4B", "layers": [12, 20, 28, 32]}


@dataclass
class Prediction:
    """One decision's answer: the probability of every option, the chosen option and its probability."""

    probs: list[float]
    answer: str
    confidence: float
    gate: float | None = None  # the domain gate's weight on the correction, when there is a gate


class Benchmaxx(torch.nn.Module):
    """Correction head plus optional domain gate.

    Args:
        head: the trained correction head.
        gate: the fitted domain gate, or ``None`` to answer with the corrected answer everywhere.
        llm: the LLM and cached layers the head was trained on (``{"model": ..., "layers": [...]}``).
    """

    def __init__(self, head: CorrectionHead, gate: DomainGate | None = None, llm: dict | None = None):
        super().__init__()
        self.head = head
        self.gate = gate
        self.llm = llm or dict(DEFAULT_LLM)

    def forward(self, batch: FeatureBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(probs (N, K), gate weight (N,) or None)`` for a batch of features."""
        probs = self.head(batch).probs
        if self.gate is None:
            return probs, None
        return self.gate(probs, batch.llm_probs, batch.answer_vectors, batch.option_mask)

    @torch.no_grad()
    def predict_features(self, features: FeatureSet, batch_size: int = 256) -> list[Prediction]:
        """Predictions for every decision of a feature set."""
        self.eval()
        device = next(self.head.parameters()).device
        out: list[Prediction] = []
        for start in range(0, len(features), batch_size):
            rows = np.arange(start, min(start + batch_size, len(features)))
            batch = features.batch(rows).to(device)
            probs, weight = self(batch)
            for i, row in enumerate(rows):
                k = int(batch.option_mask[i].sum())
                p = probs[i, :k].cpu()
                keys = features.records[row]["option_keys"]
                out.append(Prediction(probs=p.tolist(), answer=keys[int(p.argmax())], confidence=float(p.max()),
                                      gate=None if weight is None else float(weight[i])))
        return out

    def predict(self, decisions: Sequence[Decision], reader: "LLMReader") -> list[Prediction]:
        """Read ``decisions`` with the frozen LLM, then predict."""
        if reader.model_name != self.llm["model"] or list(reader.layers) != list(self.llm["layers"]):
            raise ValueError(f"the model was trained on {self.llm}, the reader uses "
                             f"{reader.model_name} at layers {list(reader.layers)}")
        return self.predict_features(reader.read(decisions))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {"llm": self.llm, "correction_head": self.head.config.to_dict()}
        save_file(_contiguous(self.head.state_dict()), directory / "correction_head.safetensors")
        if self.gate is not None:
            config["domain_gate"] = {"layers": list(self.gate.config.layers),
                                     "components": self.gate.config.components,
                                     "neighbours": self.gate.config.neighbours,
                                     "reference_rows": len(self.gate.reference)}
            save_file(_contiguous(self.gate.state_dict()), directory / "domain_gate.safetensors")
        (directory / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: torch.device | str = "cpu") -> "Benchmaxx":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        head = CorrectionHead(CorrectionHeadConfig(**config["correction_head"]))
        head.load_state_dict(load_file(directory / "correction_head.safetensors"))
        gate = None
        if gate_config := config.get("domain_gate"):
            manifold = ManifoldConfig(layers=tuple(gate_config["layers"]), components=gate_config["components"],
                                      neighbours=gate_config["neighbours"])
            gate = DomainGate(head.config.hidden_size, gate_config["reference_rows"], manifold)
            gate.load_state_dict(load_file(directory / "domain_gate.safetensors"))
        return cls(head, gate, config["llm"]).to(device).eval()


def _contiguous(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.contiguous() for k, v in state.items()}
