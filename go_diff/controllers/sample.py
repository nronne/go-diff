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

    def compute_weights(self, energies, temperature) -> np.ndarray:
        """Compute Boltzmann importance weights

        Weights are normalised so that their sum equals ``len(data)``.
        Uses the current temperature from :attr:`temperature_schedule`.

        Parameters
        ----------
        energies : array-like of float
                Scalar potential energies (eV).
        temperature : float

        Returns
        -------
        np.ndarray of float, shape ``(len(data),)``
            Boltzmann importance weights, summing to ``len(data)``.
        """
        energies = np.array(energies, dtype=float)
        Es_scaled = -energies / temperature
        Es_shifted = Es_scaled - np.max(Es_scaled)
        exp_Es = np.exp(Es_shifted)
        weights = exp_Es / np.sum(exp_Es) * len(energies)
        return weights

    def compute_ess(self, energies, temperature) -> float:
        """Compute the Effective Sample Size (ESS)

        Parameters
        ----------
        energies : array-like of float
                Scalar potential energies (eV).
        temperature : float

        Returns
        -------
        float
            The ESS (between 1 and len(data)).
        """
        w = self.compute_weights(energies, temperature)
        w_norm = w / np.sum(w)
        return float(1.0 / np.sum(w_norm ** 2))

    def calculate_ess(self, energies, temperature) -> float:
        return self.compute_ess(energies, temperature)
    
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

        current_ess = self.compute_ess(energies, temperature)

        if current_ess < self.target_ess:
            print(
                f"Continue sampling: ESS {current_ess:.3f} does not meet "
                f"target of {self.target_ess}."
            )
            return True

        print(f"Stopping sampling: ESS {current_ess:.3f} meets target.")
        return False
