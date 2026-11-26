"""Shared fixtures: small random feature sets and a character-level tokenizer, so tests need no model downloads."""

import pytest
import torch

from benchmaxx.features import FeatureSet
from benchmaxx.schema import QuestionType

HIDDEN, LAYERS = 16, 4


def random_features(n: int = 60, seed: int = 0, shift: float = 0.0, dataset: str = "main") -> FeatureSet:
    """``n`` decisions of 2 to 5 options, three per case; ``shift`` moves every vector (a different domain)."""
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(2, 6, (n,), generator=g)
    total = int(sizes.sum())
    order_probs = torch.rand(total, 2, generator=g)
    gold = torch.rand(total, generator=g)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    for i in range(n):  # normalise per decision
        a, b = int(offsets[i]), int(offsets[i + 1])
        order_probs[a:b] /= order_probs[a:b].sum(0)
        gold[a:b] /= gold[a:b].sum()
    qtype = torch.randint(0, 3, (n,), generator=g)
    qtype[sizes != 2] = torch.where(qtype[sizes != 2] == QuestionType.YES_NO.code, 0, qtype[sizes != 2])
    return FeatureSet(answer=(torch.randn(n, LAYERS, HIDDEN, generator=g) + shift).half(),
                      options=(torch.randn(total, LAYERS, HIDDEN, generator=g) + shift).half(),
                      order_probs=order_probs, offsets=offsets, qtype=qtype, gold=gold,
                      records=[{"case_id": f"{dataset}-{i // 3}", "dataset": dataset,
                                "option_keys": [f"o{k}" for k in range(int(sizes[i]))]} for i in range(n)],
                      llm={"model": "test", "layers": [1, 2, 3, 4]})


@pytest.fixture
def features() -> FeatureSet:
    return random_features()


class CharTokenizer:
    """Every character is one token; a minimal chat template."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages: list[dict], tokenize: bool, add_generation_prompt: bool,
                            enable_thinking: bool) -> str:
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer()
