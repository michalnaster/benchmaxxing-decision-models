import numpy as np
import torch

from benchmaxx.correction_head import CorrectionHeadConfig
from benchmaxx.domain_gate import ManifoldConfig
from benchmaxx.model import Benchmaxx
from benchmaxx.training import TrainingConfig, case_folds, cross_fit, fit_domain_gate, soften_gold, warmup_cosine
from tests.conftest import HIDDEN, LAYERS, random_features

HEAD = CorrectionHeadConfig(hidden_size=HIDDEN, num_layers=LAYERS, width=32)
FAST = TrainingConfig(max_epochs=3, batch_size=16, warmup_epochs=1, patience=2, device="cpu")


def test_case_folds_keep_cases_together():
    cases = [f"c{i // 3}" for i in range(90)]
    folds = case_folds(cases, ["d"] * 90, 2)
    for case in set(cases):
        assert len({folds[i] for i, c in enumerate(cases) if c == case}) == 1
    assert abs((folds == 0).sum() - (folds == 1).sum()) <= 3


def test_soften_gold_mixes_towards_uniform(features):
    softened = soften_gold(features, {"main": 0.1})
    a, b = int(features.offsets[0]), int(features.offsets[1])
    assert torch.allclose(softened.gold[a:b], 0.9 * features.gold[a:b] + 0.1 / (b - a))


def test_warmup_then_cosine():
    f = warmup_cosine(10, 100)
    assert f(0) == 0.1 and f(9) == 1.0 and abs(f(100)) < 1e-9


def test_cross_fit_gate_and_reload(tmp_path):
    train = random_features(60)
    result = cross_fit(train, FAST, HEAD)
    assert len(result.members) == 2 and sorted(np.concatenate(result.held_out_rows)) == list(range(60))
    gate = fit_domain_gate(train, result, {"other": [random_features(30, seed=1, shift=3.0, dataset="other")]},
                           ManifoldConfig(layers=(1, 2), components=4, neighbours=3))
    model = Benchmaxx(result.head, gate, train.llm)
    model.save(tmp_path)
    reloaded = Benchmaxx.load(tmp_path)
    before = [p.probs for p in model.predict_features(train)]
    after = [p.probs for p in reloaded.predict_features(train)]
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(before, after, strict=True))
