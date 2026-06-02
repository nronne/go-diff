Overview
========

What GO-Diff is
---------------

GO-Diff is a framework for atomistic global
structure search using diffusion models.  It wraps the
`AGeDi <https://github.com/nronne/agedi>`_ diffusion backend with an outer
optimisation loop that drives the model towards low-energy configurations via
Boltzmann-weighted retraining and adaptive temperature annealing.

Core capabilities
-----------------

- Generate candidate atomistic structures with a diffusion model
- Evaluate energies and forces with any ASE-compatible calculator
- Adaptively select training data via Boltzmann-weighted replay buffer
- Anneal the sampling temperature based on the energy landscape heat capacity
- Monitor convergence with gradient-agreement stopping criteria
- Log run metrics and analysis figures to TensorBoard

High-level package layout
-------------------------

- ``go_diff.GODiff``

  - Main outer optimisation loop (sample → evaluate → buffer → train)
  - Manages the replay buffer, temperature schedule, and trainer lifecycle

- ``go_diff.controllers``

  - ``TemperatureSchedule``: adaptive annealing; fast/slow cooling based on
    dimensionless heat capacity C(T) = Var(E) / T²
  - ``SampleController``: stops sampling when the Effective Sample Size (ESS)
    reaches a target threshold
  - ``BufferController``: adaptively updates the replay buffer size using an
    ESS-based exponential smoothing update, bounded by configurable min/max
    values
  - ``MomentumConsensusStop``: stops training when gradient–momentum cosine
    similarity drops below a fraction of its peak value
  - ``AdaptiveRefinementStop``: alternative stop criterion based on split-batch
    gradient cosine similarity EMA
  - ``FlopsAndTimingCallback``: tracks wall-time and FLOPs per training step

- ``go_diff.noisers``

  - ``WeightedPositions``: Boltzmann-weighted position noiser that rescales the
    diffusion loss by per-structure importance weights

- ``go_diff.GODiffLogger``

  - TensorBoard logger for iteration-level and training-step-level metrics,
    including four automatic analysis figures per iteration

Typical workflow
----------------

1. Build or load an ASE-compatible calculator (e.g. MACE-MP).
2. Create a diffusion model with :func:`agedi.create_diffusion`.
3. Instantiate :class:`~go_diff.GODiff` with the calculator, diffusion model,
   and desired controllers.
4. Call :py:meth:`~go_diff.GODiff.run` to start the optimisation loop.
5. Inspect saved checkpoints and trajectory files in the output
   directory and TensorBoard logs.

