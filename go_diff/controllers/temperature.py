"""Temperature scheduling for GO-Diff annealing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class TemperatureSchedule:
    """Adaptive temperature schedule for Boltzmann-weighted diffusion training.

    The temperature is initialised from the standard deviation of the first
    batch of energies and is then adapted each iteration based on the
    dimensionless heat capacity C(T) = Var(E) / T².

    * When C > 1 the energy landscape is still "rough" – apply a *slow*
      (conservative) cooling factor.
    * When C ≤ 1 the distribution is narrow – apply a *fast* (aggressive)
      cooling factor to anneal more quickly.

    Parameters
    ----------
    k : float
        Scale factor applied to the initial temperature estimate
        (``T₀ = k · std(E)``).  Default: 1.0.
    fast : float
        Multiplicative cooling factor used when C ≤ 1 (< 1 means cooling).
        Default: 0.5.
    slow : float
        Multiplicative cooling factor used when C > 1 (< 1 means cooling).
        Default: 0.95.
    """

    def __init__(self, k: float = 1.0, fast: float = 0.5, slow: float = 0.95) -> None:
        self.k = k
        self.fast = fast
        self.slow = slow
        self.temperature: float | None = None
        self.history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        """Return the current temperature.

        Raises
        ------
        ValueError
            If :py:meth:`next` has not been called yet.
        """
        if self.temperature is None:
            raise ValueError(
                "Temperature schedule not initialised – call next() first."
            )
        return float(self.temperature)

    def next(self, energies: ArrayLike) -> float:
        """Update the temperature given the current batch of energies.

        On the first call the temperature is initialised as
        ``T = k · std(energies)``.  On subsequent calls it is multiplied by
        *slow* or *fast* depending on the heat capacity.

        Parameters
        ----------
        energies : array-like of float
            Scalar potential energies (eV) of the sampled structures.

        Returns
        -------
        float
            The new temperature after the update.
        """
        energies = np.asarray(energies, dtype=float)
        if self.temperature is None:
            self.temperature = float(max(np.std(energies) * self.k, 1e-6))
        else:
            C = self._compute_heat_capacity(energies)
            print(
                f"Temperature: {self.temperature:.4f}, "
                f"variance: {float(np.var(energies)):.4f}, "
                f"std: {float(np.std(energies)):.4f}, "
                f"C: {float(C):.4f}"
            )
            if C > 1:
                self.temperature *= self.slow
            else:
                self.temperature *= self.fast

        self.history.append(float(self.temperature))
        return float(self.temperature)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_heat_capacity(self, energies: np.ndarray) -> float:
        """Dimensionless heat capacity C(T) = Var(E) / T².

        Parameters
        ----------
        energies : np.ndarray
            Array of scalar energies.

        Returns
        -------
        float
            The estimated heat capacity.
        """
        variance = float(np.var(energies))
        return variance / (self.temperature ** 2 + 1e-8)


TemperatureController = TemperatureSchedule


