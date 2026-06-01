"""Tests for TemperatureSchedule."""

from __future__ import annotations

import numpy as np
import pytest

from go_diff.controllers.temperature import TemperatureSchedule


class TestTemperatureSchedule:
    """Unit tests for TemperatureSchedule."""

    def test_get_temperature_before_next_raises(self):
        ts = TemperatureSchedule()
        with pytest.raises(ValueError, match="not initialised"):
            ts.get_temperature()

    def test_first_call_initialises_from_std(self):
        ts = TemperatureSchedule(k=1.0)
        energies = np.array([-1.0, -2.0, -3.0, -4.0])
        t = ts.next(energies)
        assert t == pytest.approx(float(np.std(energies)), rel=1e-6)

    def test_k_scales_initial_temperature(self):
        ts = TemperatureSchedule(k=2.0)
        energies = np.array([-1.0, -2.0, -3.0])
        t = ts.next(energies)
        assert t == pytest.approx(2.0 * float(np.std(energies)), rel=1e-6)

    def test_fast_cooling_when_heat_capacity_low(self):
        """C ≤ 1 → temperature *= fast."""
        ts = TemperatureSchedule(k=1.0, fast=0.5, slow=0.95)
        # First call to initialise
        energies_init = np.array([-10.0, -10.01, -9.99, -10.02])  # tiny std → tiny T
        ts.next(energies_init)
        T0 = ts.get_temperature()

        # Second call: use energies with variance << T^2 → C < 1 → fast cooling
        # Make variance very small relative to T^2
        tiny_energies = np.array([-10.0, -10.0, -10.0, -10.0])
        t_new = ts.next(tiny_energies)
        assert t_new == pytest.approx(T0 * 0.5, rel=1e-6)

    def test_slow_cooling_when_heat_capacity_high(self):
        """C > 1 → temperature *= slow."""
        ts = TemperatureSchedule(k=1.0, fast=0.5, slow=0.95)
        # Initialise with a small temperature
        energies_init = np.array([-1.0, -1.001])  # std ≈ 0.0005 → T very small
        ts.next(energies_init)
        T0 = ts.get_temperature()

        # Second call: variance >> T^2 → C >> 1 → slow cooling
        wide_energies = np.linspace(-1.0, -100.0, 100)
        t_new = ts.next(wide_energies)
        assert t_new == pytest.approx(T0 * 0.95, rel=1e-6)

    def test_history_is_accumulated(self):
        ts = TemperatureSchedule()
        e = np.array([-1.0, -2.0, -3.0])
        ts.next(e)
        ts.next(e)
        ts.next(e)
        assert len(ts.history) == 3

    def test_get_temperature_after_next(self):
        ts = TemperatureSchedule()
        e = np.array([-1.0, -2.0, -3.0, -4.0])
        expected = ts.next(e)
        assert ts.get_temperature() == pytest.approx(expected, rel=1e-9)

    def test_list_input(self):
        """next() should accept plain Python lists."""
        ts = TemperatureSchedule()
        t = ts.next([-1.0, -2.0, -3.0])
        assert isinstance(t, float)

    def test_returns_float(self):
        ts = TemperatureSchedule()
        t = ts.next(np.array([-1.0, -2.0, -3.0]))
        assert isinstance(t, float)
