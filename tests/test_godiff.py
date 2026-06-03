"""Tests for GODiff pure-Python helper methods.

These tests do *not* require AGeDi, PyTorch Lightning, or any GPU. The
`GODiff` class is instantiated with mock calculator and diffusion objects so
only the pure-NumPy/ASE helper methods are exercised.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from go_diff.controllers.temperature import TemperatureSchedule
from go_diff.controllers.sample import SampleController
from go_diff.filter import Filter, MinEnergyFilter, MaxEnergyFilter, MinDistFilter
from go_diff.go_diff import GODiff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atoms(energy: float, n: int = 3) -> "Atoms":
    """Return an ASE Atoms object with a SinglePointCalculator attached."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator as SPC
    atoms = Atoms("H" * n, positions=np.random.rand(n, 3) * 5)
    forces = np.random.rand(n, 3)
    atoms.calc = SPC(atoms, energy=energy, forces=forces)
    return atoms


def _make_godiff(**kwargs) -> GODiff:
    """Create a GODiff with minimal mocked dependencies."""
    defaults = dict(
        calculator=MagicMock(),
        diffusion=MagicMock(),
        temperature_schedule=TemperatureSchedule(k=1.0, fast=0.5, slow=0.9),
        sample_controller=SampleController(initial_N=4, max_N=8, target_ess=2),
    )
    defaults.update(kwargs)
    gd = GODiff(**defaults)
    # Manually set temperature so compute_weights doesn't raise
    gd.temperature_schedule.temperature = 1.0
    return gd


# ---------------------------------------------------------------------------
# MinDistFilter
# ---------------------------------------------------------------------------

class TestMinDistFilter:
    def test_accepts_well_separated(self):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator as SPC
        atoms = Atoms("HHH", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]])
        atoms.calc = SPC(atoms, energy=-1.0, forces=np.zeros((3, 3)))
        assert MinDistFilter(1.0)(atoms) is True

    def test_rejects_close_atoms(self):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator as SPC
        atoms = Atoms("HH", positions=[[0, 0, 0], [0.1, 0, 0]])
        atoms.calc = SPC(atoms, energy=-1.0, forces=np.zeros((2, 3)))
        assert MinDistFilter(1.0)(atoms) is False

    def test_single_atom_always_passes(self):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator as SPC
        atoms = Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = SPC(atoms, energy=-1.0, forces=np.zeros((1, 3)))
        assert MinDistFilter(1.0)(atoms) is True

    def test_filters_mixed_structures(self):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator as SPC
        good = Atoms("HH", positions=[[0, 0, 0], [3, 0, 0]])
        good.calc = SPC(good, energy=-1.0, forces=np.zeros((2, 3)))
        bad = Atoms("HH", positions=[[0, 0, 0], [0.1, 0, 0]])
        bad.calc = SPC(bad, energy=-2.0, forces=np.zeros((2, 3)))
        gd = _make_godiff(after_potential_filters=[MinDistFilter(1.0)])
        result = gd._apply_after_potential_filters([good, bad])
        assert len(result) == 1
        assert result[0] is good


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def test_equal_energies_give_uniform_weights(self):
        gd = _make_godiff()
        data = [_make_atoms(-1.0) for _ in range(5)]
        w = gd.compute_weights(data)
        np.testing.assert_allclose(w, np.ones(5), rtol=1e-6)

    def test_weights_sum_to_N(self):
        gd = _make_godiff()
        data = [_make_atoms(e) for e in [-1.0, -2.0, -3.0, -4.0]]
        w = gd.compute_weights(data)
        assert float(np.sum(w)) == pytest.approx(len(data), rel=1e-6)

    def test_lower_energy_gets_higher_weight(self):
        gd = _make_godiff()
        data = [_make_atoms(-1.0), _make_atoms(-10.0)]
        w = gd.compute_weights(data)
        assert w[1] > w[0]  # -10 eV should have higher weight


# ---------------------------------------------------------------------------
# compute_ess
# ---------------------------------------------------------------------------

class TestComputeEss:
    def test_equal_energies_ess_equals_N(self):
        gd = _make_godiff()
        data = [_make_atoms(-1.0) for _ in range(8)]
        ess = gd.compute_ess(data)
        assert ess == pytest.approx(8.0, rel=1e-5)

    def test_ess_between_1_and_N(self):
        gd = _make_godiff()
        data = [_make_atoms(e) for e in [-1.0, -2.0, -3.0, -4.0]]
        ess = gd.compute_ess(data)
        assert 1.0 <= ess <= len(data)

    def test_ess_is_positive(self):
        gd = _make_godiff()
        data = [_make_atoms(e) for e in [-1.0, -50.0, -3.0]]
        assert gd.compute_ess(data) > 0


# ---------------------------------------------------------------------------
# get_properties
# ---------------------------------------------------------------------------

class TestGetProperties:
    def test_returns_list_of_dicts(self):
        gd = _make_godiff()
        data = [_make_atoms(-1.0), _make_atoms(-2.0)]
        props = gd.get_properties(data)
        assert len(props) == 2
        for p in props:
            assert "weight" in p
            assert "energy" in p
            assert "forces" in p

    def test_energies_match(self):
        gd = _make_godiff()
        data = [_make_atoms(-1.0), _make_atoms(-3.0)]
        props = gd.get_properties(data)
        energies = [p["energy"] for p in props]
        assert energies[0] == pytest.approx(-1.0, rel=1e-6)
        assert energies[1] == pytest.approx(-3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# update_buffer
# ---------------------------------------------------------------------------

class TestUpdateBuffer:
    def test_empty_all_data_gives_empty_buffer(self):
        gd = _make_godiff()
        gd.all_data = []
        gd.update_buffer()
        assert gd.buffer == []

    def test_positive_energy_structures_excluded_by_default(self):
        """Default buffer_filters=[MaxEnergyFilter(0.0)] excludes positive-energy structures."""
        gd = _make_godiff()
        gd.all_data = [_make_atoms(+1.0), _make_atoms(+2.0), _make_atoms(-1.0)]
        gd.update_buffer()
        assert len(gd.buffer) == 1
        assert gd.buffer[0].get_potential_energy() < 0.0

    def test_positive_energy_structures_included_when_filters_disabled(self):
        """Positive-energy structures are kept when buffer_filters=[]."""
        gd = _make_godiff(buffer_filters=[])
        gd.all_data = [_make_atoms(+1.0), _make_atoms(+2.0)]
        gd.update_buffer()
        assert len(gd.buffer) == 2

    def test_fewer_than_buffer_size_uses_all(self):
        gd = _make_godiff()
        gd.buffer_controller.set_buffer_size(16)
        data = [_make_atoms(-float(i)) for i in range(1, 6)]
        gd.all_data = data
        gd.update_buffer()
        assert len(gd.buffer) == 5

    def test_buffer_does_not_exceed_buffer_size(self):
        gd = _make_godiff()
        gd.buffer_controller.set_buffer_size(4)
        gd.all_data = [_make_atoms(-float(i)) for i in range(1, 20)]
        gd.update_buffer()
        assert len(gd.buffer) <= 4

    def test_buffer_contains_negative_energy_structures(self):
        gd = _make_godiff()
        gd.buffer_controller.set_buffer_size(4)
        gd.all_data = [_make_atoms(-float(i)) for i in range(1, 10)]
        gd.update_buffer()
        for atoms in gd.buffer:
            assert atoms.get_potential_energy() < 0.0


# ---------------------------------------------------------------------------
# Filter classes
# ---------------------------------------------------------------------------

class TestMinEnergyFilter:
    def test_filters_below_threshold(self):
        f = MinEnergyFilter(threshold=-5.0)
        data = [_make_atoms(-3.0), _make_atoms(-10.0), _make_atoms(-1.0)]
        result = [a for a in data if f(a)]
        assert len(result) == 2
        assert all(a.get_potential_energy() > -5.0 for a in result)

    def test_keeps_valid_structures(self):
        f = MinEnergyFilter(threshold=-500.0)
        data = [_make_atoms(-1.0), _make_atoms(-2.0)]
        assert len([a for a in data if f(a)]) == 2

    def test_boundary_excluded(self):
        f = MinEnergyFilter(threshold=-5.0)
        assert not f(_make_atoms(-5.0))
        assert f(_make_atoms(-4.99))


class TestMaxEnergyFilter:
    def test_filters_above_threshold(self):
        f = MaxEnergyFilter(threshold=0.0)
        data = [_make_atoms(-1.0), _make_atoms(1.0), _make_atoms(-2.0)]
        result = [a for a in data if f(a)]
        assert len(result) == 2
        assert all(a.get_potential_energy() < 0.0 for a in result)

    def test_boundary_excluded(self):
        f = MaxEnergyFilter(threshold=0.0)
        assert not f(_make_atoms(0.0))
        assert f(_make_atoms(-0.01))

    def test_default_threshold_zero(self):
        f = MaxEnergyFilter()
        assert not f(_make_atoms(0.1))
        assert f(_make_atoms(-0.1))


class TestFilterProtocol:
    def test_min_energy_filter_satisfies_protocol(self):
        assert isinstance(MinEnergyFilter(), Filter)

    def test_max_energy_filter_satisfies_protocol(self):
        assert isinstance(MaxEnergyFilter(), Filter)

    def test_lambda_satisfies_protocol(self):
        assert isinstance(lambda a: True, Filter)


class TestApplyValidStructureFilters:
    def test_single_filter(self):
        gd = _make_godiff(after_potential_filters=[MinEnergyFilter(-5.0)])
        data = [_make_atoms(-3.0), _make_atoms(-10.0), _make_atoms(-1.0)]
        result = gd._apply_after_potential_filters(data)
        assert len(result) == 2
        assert all(a.get_potential_energy() > -5.0 for a in result)

    def test_multiple_filters_stacked(self):
        # Keep only structures with -4.0 < e < -0.5
        gd = _make_godiff(after_potential_filters=[
            MinEnergyFilter(-4.0),
            MaxEnergyFilter(-0.5),
        ])
        data = [_make_atoms(-1.0), _make_atoms(-2.0), _make_atoms(-5.0), _make_atoms(-0.1)]
        result = gd._apply_after_potential_filters(data)
        assert len(result) == 2
        assert all(-4.0 < a.get_potential_energy() < -0.5 for a in result)

    def test_none_filters_keeps_all(self):
        gd = _make_godiff(after_potential_filters=None)
        data = [_make_atoms(-1.0), _make_atoms(-600.0)]
        result = gd._apply_after_potential_filters(data)
        assert len(result) == 2

    def test_empty_filters_keeps_all(self):
        gd = _make_godiff(after_potential_filters=[])
        data = [_make_atoms(-1.0), _make_atoms(-600.0)]
        result = gd._apply_after_potential_filters(data)
        assert len(result) == 2


class TestBufferFiltersInUpdateBuffer:
    def test_default_max_energy_filter(self):
        gd = _make_godiff()
        # default buffer_filters = [MaxEnergyFilter(0.0)]
        gd.all_data = [_make_atoms(-1.0), _make_atoms(1.0), _make_atoms(-2.0)]
        gd.update_buffer()
        assert all(a.get_potential_energy() < 0.0 for a in gd.buffer)
        assert len(gd.buffer) == 2

    def test_custom_buffer_filter(self):
        gd = _make_godiff(buffer_filters=[MinEnergyFilter(-500.0)])
        gd.all_data = [_make_atoms(-1.0), _make_atoms(-600.0)]
        gd.update_buffer()
        assert len(gd.buffer) == 1
        assert gd.buffer[0].get_potential_energy() > -500.0

    def test_empty_buffer_filters_uses_all_data(self):
        gd = _make_godiff(buffer_filters=[])
        gd.all_data = [_make_atoms(-1.0), _make_atoms(1.0)]
        gd.update_buffer()
        assert len(gd.buffer) == 2
