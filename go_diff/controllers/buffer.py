from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from ase import Atoms

from go_diff.utils import effective_sample_size


@runtime_checkable
class BufferFilter(Protocol):
    """Protocol for buffer filter callables.

    A buffer filter is any callable that accepts an :class:`ase.Atoms` object
    and returns ``True`` if the structure should be **kept** in the buffer.

    Any function or object implementing ``__call__(atoms: Atoms) -> bool``
    satisfies this protocol.

    Examples
    --------
    Use the built-in :class:`MinEnergyFilter`::

        from go_diff.controllers import MinEnergyFilter
        filters = [MinEnergyFilter(threshold=0.0)]

    Or write your own::

        def my_filter(atoms):
            return atoms.get_potential_energy() > -10.0
    """

    def __call__(self, atoms: Atoms) -> bool: ...


class MinEnergyFilter:
    """Filter out structures whose potential energy is at or below a threshold.

    Parameters
    ----------
    threshold : float
        Structures with ``get_potential_energy() <= threshold`` are removed.
        The default ``0.0`` reproduces the historical ``e < 0.0`` silent
        filter; pass ``-500.0`` (the previous :attr:`GODiff.min_E` default) to
        keep only physically reasonable structures.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def __call__(self, atoms: Atoms) -> bool:
        return atoms.get_potential_energy() > self.threshold


class BufferController:
    """Controls the replay buffer size via ESS-based adaptive updates.

    Parameters
    ----------
    initial_buffer_size : int
        Starting buffer size.  Default: 16.
    min_buffer_size : int
        Lower bound on the buffer size.  Default: 16.
    max_buffer_size : int
        Upper bound on the buffer size.  Default: 512.
    adaptation_rate : float
        Exponential smoothing coefficient for buffer-size updates (between 0
        and 1).  Higher values adapt faster.  Default: 0.2.
    """

    def __init__(
        self,
        initial_buffer_size: int = 16,
        min_buffer_size: int = 16,
        max_buffer_size: int = 512,
        adaptation_rate: float = 0.2,
    ) -> None:
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.adaptation_rate = adaptation_rate

        self.current_buffer_size = initial_buffer_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ess(self, energies, temperature) -> float:
        """Compute the Effective Sample Size (ESS).

        Delegates to :func:`go_diff.utils.effective_sample_size`.
        """
        return effective_sample_size(energies, temperature)

    def update_buffer_size(
        self,
        energies: list[float],
        temperature: float | None = None,
    ) -> int:
        """Update the buffer size based on the current ESS.

        Parameters
        ----------
        energies : list of float
            Energies of structures collected so far in the current iteration.
        temperature : float or None
            Current annealing temperature.  ``None`` indicates the first
            iteration (no Boltzmann weighting yet).

        Returns
        -------
        int
            The new buffer size for the next sampling step.
        """
        if len(energies) == 0:
            return self.current_buffer_size

        ess = self.compute_ess(energies, temperature)
        target_B = int(ess)

        new_B = (1.0 - self.adaptation_rate) * self.current_buffer_size + self.adaptation_rate * target_B
        self.current_buffer_size = int(np.clip(new_B, self.min_buffer_size, self.max_buffer_size))

        return self.current_buffer_size

    def get_buffer_size(self) -> int:
        """Return the current buffer size."""
        return self.current_buffer_size

    def set_buffer_size(self, size: int) -> None:
        """Set the current buffer size."""
        self.current_buffer_size = int(size)

    def reset(self) -> None:
        """Reset the buffer size to the initial value."""
        self.current_buffer_size = self.min_buffer_size
