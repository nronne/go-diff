"""Shared utility functions for GO-Diff."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def boltzmann_weights(energies: ArrayLike, temperature: float) -> np.ndarray:
    """Compute Boltzmann importance weights for a set of energies.

    Weights are normalised so that their sum equals ``len(energies)``.

    Parameters
    ----------
    energies : array-like of float
        Scalar potential energies (eV).
    temperature : float
        Current annealing temperature.

    Returns
    -------
    np.ndarray of float, shape ``(len(energies),)``
        Boltzmann importance weights, summing to ``len(energies)``.
    """
    energies = np.asarray(energies, dtype=float)
    Es_scaled = -energies / temperature
    Es_shifted = Es_scaled - np.max(Es_scaled)
    exp_Es = np.exp(Es_shifted)
    return exp_Es / np.sum(exp_Es) * len(energies)


def effective_sample_size(energies: ArrayLike, temperature: float) -> float:
    """Compute the Effective Sample Size (ESS) for a set of energies.

    Parameters
    ----------
    energies : array-like of float
        Scalar potential energies (eV).
    temperature : float
        Current annealing temperature.

    Returns
    -------
    float
        The ESS (between 1 and ``len(energies)``).
    """
    w = boltzmann_weights(energies, temperature)
    w_norm = w / np.sum(w)
    return float(1.0 / np.sum(w_norm ** 2))
