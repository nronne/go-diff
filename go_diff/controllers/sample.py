"""Sampling controller for GO-Diff: decides when enough structures have been
collected based on the Effective Sample Size (ESS)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class SampleController:
    """Controls how many structures to sample per GO-Diff iteration.

    Sampling continues until either the ESS reaches *target_ess* or the total
    number of structures reaches *max_N*.  On the very first iteration
    (temperature = ``None``) sampling continues until *initial_N* structures
    are collected.

    Parameters
    ----------
    initial_N : int
        Minimum (and default) number of structures to sample when no
        temperature is available (first iteration).  Default: 32.
    max_N : int
        Hard upper limit on the number of structures per iteration.
        Default: 64.
    target_ess : float
        Target Effective Sample Size.  Sampling stops once the ESS computed
        from the current structures exceeds this value.  Default: 16.
    """

    def __init__(
        self,
        initial_N: int = 32,
        max_N: int = 64,
        target_ess: float = 16,
    ) -> None:
        self.initial_N = initial_N
        self.max_N = max_N
        self.target_ess = target_ess

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_ess(self, energies: ArrayLike, temperature: float) -> float:
        """Compute the Effective Sample Size for Boltzmann weights.

        Parameters
        ----------
        energies : array-like of float
            Scalar potential energies (eV).
        temperature : float
            Current annealing temperature.

        Returns
        -------
        float
            The ESS.
        """
        energies = np.asarray(energies, dtype=float)
        # Shift for numerical stability (largest weight = exp(0) = 1)
        shifted = (energies - np.min(energies)) / temperature
        e = np.exp(-shifted)
        weights = e / np.sum(e)
        return float(1.0 / np.sum(weights ** 2))

    def continue_sampling(
        self,
        energies: list[float],
        temperature: float | None = None,
    ) -> bool:
        """Decide whether to sample more structures.

        Parameters
        ----------
        energies : list of float
            Energies of structures collected so far in the current iteration.
        temperature : float or None
            Current annealing temperature.  ``None`` indicates the first
            iteration (no Boltzmann weighting yet).

        Returns
        -------
        bool
            ``True`` if more structures should be sampled.
        """
        if len(energies) == 0:
            return True

        if len(energies) >= self.max_N:
            return False

        if temperature is None:
            keep_going = len(energies) < self.initial_N
            return keep_going

        current_ess = self.calculate_ess(energies, temperature)

        if current_ess < self.target_ess:
            print(
                f"Continue sampling: ESS {current_ess:.3f} does not meet "
                f"target of {self.target_ess}."
            )
            return True

        print(f"Stopping sampling: ESS {current_ess:.3f} meets target.")
        return False

