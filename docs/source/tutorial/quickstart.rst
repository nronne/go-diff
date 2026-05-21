Quickstart: Pt adatom on a Pt slab
===================================

This end-to-end example reproduces the Pt heptamer search from the paper.
It places a single Pt adatom on a Pt(1,2,2) × 5-layer slab, using MACE-MP
as the energy/force calculator.

Prerequisites
-------------

Make sure you have completed the :doc:`../installation` steps and have both
MACE and the full AGeDi backend available:

.. code-block:: console

   pip install mace-torch
   pip install "agedi[full]"

Full script
-----------

.. code-block:: python

   import numpy as np
   from ase.build import surface
   from mace.calculators import mace_mp
   from agedi import create_diffusion

   from go_diff import GODiff
   from go_diff.controllers import (
       TemperatureSchedule,
       SampleController,
       MomentumConsensusStop,
   )
   from go_diff.noisers import WeightedPositions

   # 1. Calculator
   calc = mace_mp(
       model="medium",
       dispersion=False,
       default_dtype="float32",
       device="cuda",
   )

   # 2. Substrate template (atoms to keep fixed)
   template = surface("Pt", (1, 2, 2), 5, vacuum=8.0)
   template.positions[:, 2] -= template.positions[:, 2].min()
   confinement = [0.0, 4.0 + template.positions[:, 2].max()]

   # 3. Diffusion model (AGeDi)
   diffusion = create_diffusion(noisers=(WeightedPositions(),))

   # 4. GO-Diff optimiser
   godiff = GODiff(
       calculator=calc,
       diffusion=diffusion,
       temperature_schedule=TemperatureSchedule(fast=0.5, slow=0.9),
       sample_controller=SampleController(initial_N=16, target_ess=8),
       training_controller=MomentumConsensusStop(
           min_steps=100, patience=250, drop_factor=0.9
       ),
       sample_config={
           "template": template,
           "atomic_numbers": [78],   # one Pt adatom
       },
       dataset_config={
           "mask": "MaskFixed",
           "confinement": confinement,
       },
       initial_buffer_size=16,
       min_E=-200,
   )

   # 5. Run
   final_checkpoint = godiff.run(max_iterations=50)
   print(f"Model saved to {final_checkpoint}")

Step-by-step walkthrough
------------------------

**Calculator**

Any ASE-compatible calculator can be used.  Here we use ``mace_mp`` with the
medium pre-trained model.  Specifying ``device="cuda"`` moves the model to GPU
for faster inference.

**Template and confinement**

The template slab defines which atoms are held fixed (via ASE ``FixAtoms``
constraints).  The ``confinement`` list ``[z_min, z_max]`` bounds the
z-coordinate range within which new atoms are generated.  Setting
``z_max = 4.0 + slab_top`` gives a 4 Å window above the surface.

**Diffusion model**

:func:`agedi.create_diffusion` builds an AGeDi diffusion model.  Passing
:class:`~go_diff.noisers.WeightedPositions` instead of the standard
``Positions`` noiser enables Boltzmann-weighted score-matching loss.

**Controllers**

* ``TemperatureSchedule(fast=0.5, slow=0.9)`` – applies a ×0.5 cooling step
  when C ≤ 1 and ×0.9 when C > 1.
* ``SampleController(initial_N=16, target_ess=8)`` – collects at least 16
  structures on the first iteration; on subsequent iterations it keeps sampling
  until ESS ≥ 8.
* ``MomentumConsensusStop(min_steps=100, patience=250, drop_factor=0.9)``
  – waits at least 100 training steps then stops when gradient–momentum
  agreement has been below 90 % of its peak for 250 steps.

**GODiff.run**

Calling ``godiff.run(max_iterations=50)`` starts the outer loop.  Output is
written to a timestamped directory (``godiff_<timestamp>/``) containing:

* ``all_data.traj`` – all evaluated structures sorted by energy.
* ``iteration_data_T<temp>.traj`` – per-iteration structures.
* ``lightning_logs/`` – PyTorch Lightning checkpoints and TensorBoard logs.

Monitoring with TensorBoard
----------------------------

.. code-block:: console

   tensorboard --logdir godiff_<timestamp>/lightning_logs

Key scalars to watch:

* ``iteration/best_energy`` – global minimum energy found so far.
* ``iteration/ess`` – effective sample size per iteration.
* ``iteration/temperature`` – annealing temperature.
* ``train/gradient_agreement_current`` – gradient–momentum cosine similarity.

Reproducing paper scripts
--------------------------

The ``scripts/`` directory contains the scripts used to produce the paper
results:

* ``scripts/heptamer.py`` – Pt heptamer search (main result).
* ``scripts/heptamer_ffg.py`` – heptamer search with force-field guidance.
* ``scripts/heptamer_transfer.py`` – transfer from a pre-trained checkpoint.
* ``scripts/stepped_Pt.py`` – stepped Pt surface search.

Additional utilities for identifying the Pt heptamer structure are in
``utils/``.
