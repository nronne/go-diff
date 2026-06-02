from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from ase import Atoms

@runtime_checkable
class Filter(Protocol):
    """Protocol for structure filter callables.

    A filter is any callable that accepts an :class:`ase.Atoms` object and
    returns ``True`` if the structure should be **kept**.

    Any function or object implementing ``__call__(atoms: Atoms) -> bool``
    satisfies this protocol.

    Examples
    --------
    Use the built-in filter classes::

        from go_diff.controllers import MinEnergyFilter, MaxEnergyFilter
        valid_structure_filters = [MinEnergyFilter(-500.0)]
        buffer_filters = [MaxEnergyFilter(0.0)]

    Or write your own::

        def my_filter(atoms):
            return atoms.get_potential_energy() > -10.0
    """

    def __call__(self, atoms: Atoms) -> bool: ...


class MinEnergyFilter:
    """Keep only structures whose potential energy is strictly above a threshold.

    Parameters
    ----------
    threshold : float
        Structures with ``get_potential_energy() <= threshold`` are removed.
        Pass ``-500.0`` to replicate the former ``GODiff(min_E=-500.0)``
        behaviour and discard clearly unphysical structures.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def __call__(self, atoms: Atoms) -> bool:
        return atoms.get_potential_energy() > self.threshold


class MaxEnergyFilter:
    """Keep only structures whose potential energy is strictly below a threshold.

    Parameters
    ----------
    threshold : float
        Structures with ``get_potential_energy() >= threshold`` are removed.
        The default ``0.0`` reproduces the historical silent ``e < 0.0``
        buffer filter, keeping only negative-energy structures.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def __call__(self, atoms: Atoms) -> bool:
        return atoms.get_potential_energy() < self.threshold


class MinDistFilter:
    """Keep only structures where all pairwise interatomic distances are at
    least *min_dist*.

    Parameters
    ----------
    min_dist : float
        Minimum allowed distance (Å) between any pair of atoms.  Structures
        with any pair closer than this threshold are discarded.
        Default: 1.0.

    Examples
    --------
    Supply as a :attr:`~go_diff.GODiff.valid_structure_filters` entry to
    replace the former hard-coded ``check_min_dist`` call::

        from go_diff.filter import MinDistFilter
        valid_structure_filters = [MinDistFilter(1.0)]
    """

    def __init__(self, min_dist: float = 1.0) -> None:
        self.min_dist = min_dist

    def __call__(self, atoms: Atoms) -> bool:
        positions = atoms.get_positions()
        dists = np.linalg.norm(
            positions[:, np.newaxis] - positions, axis=-1
        )
        np.fill_diagonal(dists, np.inf)
        return float(np.min(dists)) >= self.min_dist


