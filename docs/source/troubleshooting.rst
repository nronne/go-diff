Pitfalls and troubleshooting
============================

Most issues arise from mismatched configuration between the AGeDi diffusion
model and the GO-Diff wrapper, or from numerical instabilities during early
iterations.

Sampling pitfalls
-----------------

- **All sampled structures are filtered out**

  GO-Diff discards structures with energy below ``min_E`` (default: -500 eV)
  and structures with interatomic distances below 1.0 Å.  If every structure
  in an iteration is rejected the buffer will be empty and training cannot
  proceed.

  *Fix*: lower ``min_E`` if your system legitimately reaches very negative
  energies, or check that the confinement bounds are physically reasonable
  so structures do not overlap.

- **ESS target never reached**

  If ``SampleController`` keeps sampling until ``max_N`` is hit without
  reaching ``target_ess``, the Boltzmann weight distribution is very uneven.
  This typically happens when the temperature is too low for the energy spread
  of the current ensemble.

  *Fix*: increase ``target_ess`` or raise ``k`` in ``TemperatureSchedule``
  to start from a higher initial temperature.

- **No valid structures with negative energy**

  The buffer update requires structures with negative energy.  This can fail
  on the very first iteration if the untrained diffusion model generates
  unphysical geometries.

  *Fix*: verify that the template and confinement are set correctly.  Running
  the calculator on a reasonable reference structure beforehand helps confirm
  that the energy scale is as expected.

Temperature and annealing pitfalls
------------------------------------

- **Temperature drops to zero immediately**

  If ``std(E)`` in the first iteration is zero (all structures have identical
  energy), ``T₀ = 0``.  Division by temperature will then produce NaN or
  infinite weights.

  *Fix*: ensure enough structural diversity in the first batch by increasing
  ``initial_N`` in ``SampleController``.

- **Cooling too aggressive**

  Setting ``fast`` close to 0 can collapse the temperature to near-zero in a
  single step, causing the sampler to get stuck.

  *Fix*: values in the range 0.3–0.7 for ``fast`` and 0.8–0.95 for ``slow``
  work well for typical surface-adatom problems.

Training pitfalls
-----------------

- **MomentumConsensusStop triggers immediately**

  If ``min_steps`` is smaller than the Adam warm-up phase, the momentum
  estimate is unreliable and the callback may stop training after only a few
  steps.

  *Fix*: set ``min_steps`` to at least 50–100 steps.  A larger ``patience``
  value also helps.

- **AdaptiveRefinementStop requires batch size ≥ 4**

  The split-batch gradient agreement cannot be computed when the mini-batch
  contains fewer than 4 graphs.  The callback silently skips these batches.

  *Fix*: use a ``batch_size`` of at least 8, or switch to
  ``MomentumConsensusStop``.

- **AGeDi not installed or wrong branch**

  GO-Diff requires the ``boltzmann-diffusion`` branch of AGeDi.  Using the
  main branch will result in import errors or missing ``create_diffusion``
  arguments.

  *Fix*: follow the installation instructions and verify the correct branch is
  checked out.

TensorBoard / logging pitfalls
--------------------------------

- **No TensorBoard logs appear**

  The ``GODiffLogger`` is only attached if a ``TensorBoardLogger`` is present
  in the Lightning trainer.  ``GODiff.get_trainer()`` sets this up
  automatically; if you build the trainer manually, ensure a
  ``TensorBoardLogger`` is included.

- **Figures not visible in TensorBoard**

  The four analysis figures require ``matplotlib`` and ``seaborn``.  Both are
  listed as project dependencies, but if you are using a minimal virtual
  environment they may need to be installed separately.

GPU / CUDA pitfalls
--------------------

- **CUDA out of memory**

  Large ``batch_size`` combined with a deep PaiNN model can exhaust GPU
  memory.  Reduce ``batch_size`` or the number of message-passing layers in
  the AGeDi model.

- **device mismatch**

  The ``device`` argument to :class:`~go_diff.GODiff` must match the device
  used by the calculator.  Mixing CPU and CUDA tensors will raise runtime
  errors.
