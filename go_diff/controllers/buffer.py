from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from copy import copy


class BufferController:
    """Controls the buffer size

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
        initial_buffer_size: int = 16,
        min_buffer_size: int = 16,
        max_buffer_size: int = 512,
        adaption_rate: float = 0.2,
    ) -> None:
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.adaption_rate = adaption_rate

        self.current_buffer_size = initial_buffer_size
        


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
    
    def update_buffer_size(
        self,
        energies: list[float],
        temperature: float | None = None,
    ) -> bool:
        """update the buffer size

        Parameters
        ----------
        energies : list of float
            Energies of structures collected so far in the current iteration.
        temperature : float or None
            Current annealing temperature.  ``None`` indicates the first
            iteration (no Boltzmann weighting yet).

        Returns
        -------
        int:
            The new buffer size for the next sampling step.

        """
        if len(energies) == 0:
            return self.current_buffer_size

        ess = self.compute_ess(energies, temperature)
        target_B = int(ess)

        new_B = (1.0 - self.adaption_rate) * self.current_buffer_size + self.adaption_rate * target_B
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
