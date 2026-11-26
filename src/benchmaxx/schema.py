"""Typed decisions: the input format of every benchmaxx model.

A *decision* is one question about one document (the *state*), answered by choosing one of its options. A *case* is a
document with all the questions asked about it; cases matter for training, where all decisions of a case stay on the
same side of every split.

Decisions are stored one per line in JSON Lines::

    {"case_id": "ticket-17",
     "state": "I was charged twice, and the app crashes when I open the invoice.",
     "question": {"text": "Which team should handle this?",
                  "type": "choice",
                  "options": [{"key": "billing", "text": "payments and invoices"},
                              {"key": "technical", "text": "product faults"},
                              {"key": "sales", "text": "new business"}]},
     "gold": [0.7, 0.25, 0.05],
     "dataset": "tickets"}

``gold`` is a probability per option (soft gold); ``"label": 0`` may be given instead for a single right answer, and
neither is needed for prediction. A ``yes_no`` question may omit its options (they default to *no*, *yes*); a
``score`` question lists its levels from lowest to highest.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class QuestionType(StrEnum):
    """The three question types. Their order defines the integer codes used in feature tensors."""

    CHOICE = "choice"
    YES_NO = "yes_no"
    SCORE = "score"

    @property
    def code(self) -> int:
        return list(QuestionType).index(self)


YES_NO_KEYS = ("no", "yes")


@dataclass(frozen=True, slots=True)
class Question:
    """A question with its options. ``option_texts[k]`` describes ``option_keys[k]``; an empty text means the key
    says it all."""

    text: str
    type: QuestionType
    option_keys: tuple[str, ...]
    option_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.option_keys) != len(self.option_texts):
            raise ValueError("every option needs a key and a text")
        if len(self.option_keys) < 2:
            raise ValueError("a question needs at least two options")
        if self.type is QuestionType.YES_NO and self.option_keys != YES_NO_KEYS:
            raise ValueError(f"a yes/no question has the options {YES_NO_KEYS}, in that order")

    @property
    def num_options(self) -> int:
        return len(self.option_keys)


@dataclass(frozen=True, slots=True)
class Decision:
    """One question about one state, with its gold distribution over the options when known."""

    case_id: str
    state: str
    question: Question
    gold: tuple[float, ...] | None = None
    dataset: str = ""

    def __post_init__(self) -> None:
        if self.gold is not None:
            if len(self.gold) != self.question.num_options:
                raise ValueError(f"case {self.case_id}: gold has {len(self.gold)} values for "
                                 f"{self.question.num_options} options")
            if abs(sum(self.gold) - 1.0) > 1e-3:
                raise ValueError(f"case {self.case_id}: gold must sum to 1")

    @classmethod
    def from_dict(cls, row: dict) -> "Decision":
        """Parse one JSON Lines row (format in the module docstring)."""
        q = row["question"]
        qtype = QuestionType(q.get("type", "choice"))
        options = q.get("options") or ([{"key": k} for k in YES_NO_KEYS] if qtype is QuestionType.YES_NO else [])
        question = Question(text=q["text"], type=qtype,
                            option_keys=tuple(str(o["key"]) for o in options),
                            option_texts=tuple(str(o.get("text", "")) for o in options))
        gold = row.get("gold")
        if gold is None and row.get("label") is not None:
            gold = [float(k == row["label"]) for k in range(question.num_options)]
        return cls(case_id=str(row["case_id"]), state=row["state"], question=question,
                   gold=tuple(float(g) for g in gold) if gold is not None else None,
                   dataset=str(row.get("dataset", "")))


def read_decisions(path: str | Path) -> list[Decision]:
    """Every decision of a JSON Lines file."""
    return list(iter_decisions(path))


def iter_decisions(path: str | Path) -> Iterator[Decision]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield Decision.from_dict(json.loads(line))
