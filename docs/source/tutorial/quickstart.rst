Quickstart: Pt adatom on a Pt slab
===================================

This end-to-end example reproduces the Pt heptamer search from the paper.
It places a single Pt adatom on a Pt(1,2,2) × 5-layer slab, using MACE MLIP
as the energy/force calculator.

Prerequisites
-------------

Make sure you have completed the :doc:`../installation` steps and have 
the MACE MLIP available:

.. code-block:: console

   pip install mace-torch

Full script
-----------

.. code-block:: python
		
   import numpy as np

   from ase.build import fcc111, surface
   from mace.calculators import mace_mp

   from agedi import AtomsGraph, create_diffusion
   from agedi.diffusion import ForcefieldGuidanceConfig

   from go_diff import GODiff, MinEnergyFilter
   from go_diff.controllers import SampleController, BufferController, TemperatureSchedule, MomentumConsensusStop
   from go_diff.noisers import WeightedConfinedCellPositions

   ##### SYSTEM #####
   formula = "Pt"

   template = surface('Pt', (1,2,2), 5, vacuum=8.0)
   template.positions[:, 2] -= template.positions[:, 2].min()

   confinement_above_zmax = np.array([0.0, 4.0])
   confinement = confinement_above_zmax + template.positions[:, 2].max()


   ##### CALCULATOR #####
   from mace.calculators import mace_mp
   calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')


   #### DIFFUSION MODEL #####
   diffusion = create_diffusion(noisers=(WeightedConfinedCellPositions(),), force_field=True)


   #### GO-DIFF #####
   godiff = GODiff(
       calculator=calc,
       diffusion=diffusion,
       temperature_schedule=TemperatureSchedule(fast=0.5, slow=0.9),
       sample_controller=SampleController(initial_N=16, target_ess=8),
       buffer_controller=BufferController(initial_buffer_size=16, max_buffer_size=96, adaption_rate=0.2),
       training_controller=MomentumConsensusStop(min_steps=100, patience=250, drop_factor=0.9),
       sample_config={
	   "template": template,
	   "formula": formula,
	   "confinement": confinement,
	   "ff_guidance": ForcefieldGuidanceConfig(guidance=1.0,)
       },
       dataset_config={
	   "mask": "MaskFixed",
	   "confinement": confinement,
	   "regressor_data": "all_data", # use all data for training the regressor, not just the data in the buffer
       },
       trainer_config={
	   "name": name
       },
       min_E=min_E,
       valid_structure_filters=[MinEnergyFilter(-200)],    
   )

   #### RUN GO-Diff #####
   godiff.run(max_iterations=20)


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

Calling ``godiff.run(max_iterations=20)`` starts the outer loop.  Output is
written to a logs directory  containing:

* ``all_data.traj`` – all evaluated structures sorted by energy.
* ``iteration_data_T<temp>.traj`` – per-iteration structures.
* ``lightning_logs/`` – PyTorch Lightning checkpoints and TensorBoard logs.

Monitoring with TensorBoard
----------------------------

.. code-block:: console

   tensorboard --logdir ...

Key scalars to watch:

* ``iteration/best_energy`` – global minimum energy found so far.
* ``iteration/ess`` – effective sample size per iteration.
* ``iteration/temperature`` – annealing temperature.
* ``train/gradient_agreement_current`` – gradient–momentum cosine similarity.

Reproducing paper scripts
--------------------------

The ``scripts/`` directory contains the several useful scripts.
