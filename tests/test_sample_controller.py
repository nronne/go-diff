"""Tests for SampleController."""

from __future__ import annotations

import numpy as np
import pytest

from go_diff.controllers.sample import SampleController


class TestSampleController:
    """Unit tests for SampleController."""

    def test_empty_list_continues(self):
        sc = SampleController(initial_N=8, max_N=32, target_ess=4)
        assert sc.continue_sampling([], temperature=None) is True

    def test_first_iteration_samples_until_initial_N(self):
        sc = SampleController(initial_N=8, max_N=32, target_ess=4)
        # Below initial_N
        assert sc.continue_sampling([-1.0] * 7, temperature=None) is True
        # At initial_N – should stop
        assert sc.continue_sampling([-1.0] * 8, temperature=None) is False

    def test_max_N_hard_stops_sampling(self):
        sc = SampleController(initial_N=8, max_N=32, target_ess=4)
        all_equal_energies = [-1.0] * 32
        # All equal → ESS = N = 32, but max_N check fires first
        assert sc.continue_sampling(all_equal_energies, temperature=0.5) is False

    def test_ess_target_stops_sampling(self):
        """When all energies are equal the ESS = N, so target is met immediately."""
        sc = SampleController(initial_N=1, max_N=64, target_ess=4)
        energies = [-1.0] * 16   # ESS = 16 > target_ess=4 → stop
        assert sc.continue_sampling(energies, temperature=0.5) is False

    def test_low_ess_continues_sampling(self):
        """Very spread energies → low ESS → continue sampling."""
        sc = SampleController(initial_N=1, max_N=64, target_ess=50)
        # One energy dominates heavily → low ESS
        energies = [-100.0] + [-1.0] * 15   # weight on first >> rest
        assert sc.continue_sampling(energies, temperature=0.1) is True

    def test_calculate_ess_uniform_weights(self):
        """Equal energies → ESS = N."""
        sc = SampleController()
        energies = [-1.0] * 10
        ess = sc.calculate_ess(energies, temperature=1.0)
        assert ess == pytest.approx(10.0, rel=1e-6)

    def test_calculate_ess_one_dominant(self):
        """One dominant energy → ESS ≈ 1."""
        sc = SampleController()
        # Very large negative energy dominates
        energies = [-1000.0] + [-1.0] * 99
        ess = sc.calculate_ess(energies, temperature=0.01)
        assert ess == pytest.approx(1.0, abs=0.1)

    def test_calculate_ess_is_positive(self):
        sc = SampleController()
        ess = sc.calculate_ess([-1.0, -2.0, -3.0], temperature=1.0)
        assert ess > 0

    def test_calculate_ess_between_1_and_N(self):
        sc = SampleController()
        energies = list(range(-10, 0))
        ess = sc.calculate_ess(energies, temperature=1.0)
        assert 1.0 <= ess <= len(energies)
