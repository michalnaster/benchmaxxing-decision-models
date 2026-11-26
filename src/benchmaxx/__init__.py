"""benchmaxx: near-fine-tune accuracy for decision models from a frozen LLM and a small head.

The recipe, one module per step:

1. :mod:`benchmaxx.prompt` asks the frozen LLM for one option code, in two option orders.
2. :mod:`benchmaxx.reader` keeps its hidden states: an answer vector and one vector per option.
3. :mod:`benchmaxx.correction_head` learns a gated correction to the LLM's own answer.
4. :mod:`benchmaxx.domain_gate` hands decisions unlike the training data back to the LLM.
5. :mod:`benchmaxx.training` trains both by cross-fitting.

:class:`~benchmaxx.model.Benchmaxx` puts the trained parts together.
"""

from benchmaxx.correction_head import CorrectionHead, CorrectionHeadConfig
from benchmaxx.domain_gate import DomainGate, ManifoldConfig
from benchmaxx.features import FeatureBatch, FeatureSet
from benchmaxx.model import Benchmaxx, Prediction
from benchmaxx.schema import Decision, Question, QuestionType, read_decisions
from benchmaxx.training import TrainingConfig, train

__version__ = "0.1.0"

__all__ = [
    "Benchmaxx", "CorrectionHead", "CorrectionHeadConfig", "Decision", "DomainGate", "FeatureBatch", "FeatureSet",
    "ManifoldConfig", "Prediction", "Question", "QuestionType", "TrainingConfig", "read_decisions", "train",
]
