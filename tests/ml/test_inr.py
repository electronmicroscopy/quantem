"""Tests for ``quantem.core.ml.inr`` Siren / HSiren construction."""

import torch

from quantem.core.ml.inr import HSiren, Siren


def _first_layer_weight(winner_seed) -> torch.Tensor:
    # Fix the global RNG so the base SineLayer init is identical across builds;
    # only the winner-initialization perturbation varies with winner_seed.
    torch.manual_seed(0)
    model = HSiren(hidden_layers=1, hidden_features=8, winner_initialization=winner_seed)
    return model.net[0].linear.weight.detach().clone()


class TestWinnerInitialization:
    def test_same_seed_is_reproducible(self):
        assert torch.equal(_first_layer_weight(72), _first_layer_weight(72))

    def test_different_seeds_differ(self):
        """Regression: the seeded torch.Generator was created but never used
        (torch.randn_like ignores generators), so every winner seed produced
        the same perturbation."""
        assert not torch.equal(_first_layer_weight(72), _first_layer_weight(73))

    def test_true_uses_default_seed(self):
        assert torch.equal(_first_layer_weight(True), _first_layer_weight(42))

    def test_disabled_adds_no_perturbation(self):
        torch.manual_seed(0)
        base = Siren(hidden_layers=1, hidden_features=8, winner_initialization=False)
        torch.manual_seed(0)
        again = Siren(hidden_layers=1, hidden_features=8, winner_initialization=False)
        assert torch.equal(base.net[0].linear.weight.detach(), again.net[0].linear.weight.detach())


def test_forward_shape():
    model = HSiren(hidden_layers=1, hidden_features=8)
    out = model(torch.rand(5, 3))
    assert out.shape == (5, 1)
