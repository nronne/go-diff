Concepts and model behaviour
============================

The GO-Diff algorithm
---------------------

GO-Diff implements an iterative outer loop over four stages:

1. **Sample** – generate candidate structures from the current diffusion model.
2. **Evaluate** – compute potential energies and forces with an ASE calculator.
3. **Buffer** – select a Boltzmann-weighted subset (the *replay buffer*) for
   training.
4. **Train** – retrain the diffusion model on the buffer until the training
   controller signals convergence.

After each iteration the annealing temperature is updated so that the model
gradually focuses on lower-energy regions of configuration space.

Temperature annealing
---------------------

The temperature is initialised from the standard deviation of the first batch
of energies (``T₀ = k · std(E)``).  At each subsequent iteration the
dimensionless heat capacity is computed:

.. code-block:: text

   C(T) = Var(E) / T²

* When **C > 1** the energy distribution is still wide – the *slow* cooling
  factor is applied (``T ← slow · T``).
* When **C ≤ 1** the distribution has narrowed – the *fast* cooling factor
  is applied (``T ← fast · T``).

This adaptive scheme avoids premature annealing while still converging on
low-energy structures.

Boltzmann-weighted replay buffer
---------------------------------

Structures are weighted by their Boltzmann factor at the current temperature:

.. code-block:: text

   w_i ∝ exp(-E_i / T)

Weights are normalised so that their sum equals the number of structures.
The buffer is then filled by **stochastic prioritised sampling** using the
reservoir-key trick:

.. code-block:: text

   key_i = U_i^(1 / w_i),   U_i ~ Uniform(0, 1)

The structures with the largest keys are selected.  This is equivalent to
weighted sampling without replacement and naturally over-represents
low-energy structures while retaining diversity.

Effective Sample Size (ESS)
---------------------------

The ESS provides a single number summarising how evenly the Boltzmann weights
are distributed:

.. code-block:: text

   ESS = 1 / Σ w̃_i²,   w̃_i = w_i / Σ w_j

An ESS equal to *N* means all structures contribute equally; an ESS of 1
means a single structure dominates.  The :class:`~go_diff.controllers.SampleController`
continues sampling until the ESS exceeds a configurable target.

Adaptive buffer size
--------------------

The buffer size is adapted each iteration using a smoothed update towards the
current ESS. E.g. for `adaptation_rate=0.2`:

.. code-block:: text

   buffer_size ← 0.8 · buffer_size + 0.2 · ESS

This ensures the buffer captures the effective diversity of the current
ensemble without becoming excessively large.

Training stop criteria
-----------------------

Two stopping callbacks are available:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~go_diff.controllers.MomentumConsensusStop`
     - Computes the cosine similarity between the current gradient and the
       Adam first-moment (momentum).  Stops when agreement drops below
       ``drop_factor × peak_agreement`` for ``patience`` consecutive steps.
   * - :class:`~go_diff.controllers.AdaptiveRefinementStop`
     - Splits each mini-batch in two, runs independent backward passes, and
       measures the cosine similarity of the resulting gradients.  An EMA of
       the agreement is tracked; training stops when the EMA has fallen to
       less than half its peak for ``patience`` steps.

Both callbacks reset their internal state at the start of each new GO-Diff
training stage.

WeightedPositions noiser
------------------------

:class:`~go_diff.noisers.WeightedPositions` extends the AGeDi
``Positions`` noiser by rescaling the diffusion score-matching loss with
per-structure Boltzmann weights:

.. code-block:: text

   L = mean( w_i · ||score_i · σ_t² + noise_i||² )

This steers the model towards low-energy structures without discarding
high-energy data entirely.

TensorBoard logging
-------------------

:class:`~go_diff.GODiffLogger` writes metrics at two granularities:

* **Iteration level** – temperature, energy statistics (min / mean / std of
  new samples, all accumulated data, and the buffer), ESS, heat capacity, and
  wall-clock time broken down by stage.
* **Training-step level** – gradient-agreement metrics (current, EMA, peak,
  patience counter) written by the training controller callback.

Four analysis figures are also produced each iteration under the
``analysis/`` TensorBoard tag group:

1. Best energy vs. cumulative dataset size.
2. Buffer-energy violin plots as a function of temperature.
3. Histogram of energies sampled in the current iteration.
4. Energy landscape overview (min, mean ± std vs. iteration).
