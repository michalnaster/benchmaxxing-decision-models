import torch

from benchmaxx.domain_gate import DomainGate, GateExamples, Manifold, ManifoldConfig, fit_gate

CONFIG = ManifoldConfig(layers=(0,), components=4, neighbours=3)


def test_manifold_scores_far_points_higher():
    g = torch.Generator().manual_seed(0)
    scale = torch.tensor([5.0, 4, 3, 2, 0.1, 0.1, 0.1, 0.1])
    train = torch.randn(500, 8, generator=g) * scale
    manifold = Manifold(train, CONFIG)
    near = manifold.scores(torch.randn(50, 8, generator=g) * scale).mean(0)
    far = manifold.scores(torch.randn(50, 8, generator=g) * scale + torch.tensor([0, 0, 0, 0, 2.0, 2, 2, 2])).mean(0)
    assert (far > near).all()


def examples(n: int, head_right: bool, scores: float) -> GateExamples:
    gold = torch.zeros(n, 2)
    gold[:, 0] = 1
    right, wrong = torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])
    head = (right if head_right else wrong).expand(n, 2)
    return GateExamples(scores=torch.full((n, 2), scores) + 0.01 * torch.randn(n, 2), head_probs=head.clone(),
                        llm_probs=torch.full((n, 2), 0.5), gold=gold, option_mask=torch.ones(n, 2, dtype=torch.bool))


def test_gate_trusts_the_head_in_domain_only():
    fit = fit_gate([examples(200, head_right=True, scores=1.0)], [examples(200, head_right=False, scores=3.0)])
    assert fit.weight(examples(10, True, 1.0)).mean() > 0.8
    assert fit.weight(examples(10, True, 3.0)).mean() < 0.2


def test_module_reproduces_the_fit():
    train = torch.randn(300, 8)
    manifold = Manifold(train, CONFIG)
    fit = fit_gate([examples(50, True, 1.0)], [examples(50, False, 3.0)])
    gate = DomainGate.from_fit(manifold, fit, hidden_size=8)
    vectors = torch.randn(5, 1, 8)
    assert torch.allclose(gate.manifold_scores(vectors), manifold.scores(vectors[:, 0]), atol=1e-5)
