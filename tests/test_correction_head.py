import numpy as np
import torch

from benchmaxx.correction_head import CorrectionHead, CorrectionHeadConfig
from tests.conftest import HIDDEN, LAYERS


def small_head() -> CorrectionHead:
    return CorrectionHead(CorrectionHeadConfig(hidden_size=HIDDEN, num_layers=LAYERS, width=32)).eval()


def test_outputs_distributions_over_real_options(features):
    batch = features.batch(np.arange(10))
    out = small_head()(batch)
    assert out.probs.shape == batch.option_mask.shape
    assert torch.allclose(out.probs.sum(-1), torch.ones(10), atol=1e-5)
    assert (out.probs[~batch.option_mask] == 0).all()


def test_starts_at_the_llms_answer(features):
    head = small_head()
    head.fit_input_statistics(features)
    batch = features.batch(np.arange(20))
    out = head(batch)
    llm = torch.softmax(0.5 * batch.order_probs.clamp_min(1e-9).log().sum(1).masked_fill(~batch.option_mask, -1e9), -1)
    assert torch.allclose(out.gate, torch.full_like(out.gate, torch.sigmoid(torch.tensor(-4.0)).item()))
    assert (out.probs - llm).abs().max() < 0.1  # only the closed gate's small correction separates them


def test_benchmaxx_4b_head_size():
    assert CorrectionHead(CorrectionHeadConfig()).num_parameters() == 2_240_584
