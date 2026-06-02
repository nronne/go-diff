# GO-Diff

**GO-Diff: Data-free and amortized global structure optimization** a
generative diffusion framework for atomistic structure search.  It couples a diffusion model with an
energy/force calculator and iteratively refines the model through
Boltzmann-weighted training, adaptive temperature annealing, and an adaptive
replay buffer.

---

## Installation

Install from pypi
```bash
pip install go-diff
```

### (Optional) Install of MLIPs for running GO-Diff
E.g.
```bash
pip install mace-torch
```

---

## Quick Start

```python
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

```

See `scripts/*.py` for the full scripts.

---

## Key components

| Class / module | Description |
|---|---|
| `GODiff` | Main optimisation loop (sample → evaluate → buffer → train) |
| `TemperatureSchedule` | Adaptive annealing; `fast` / `slow` cooling factors based on heat capacity |
| `SampleController` | Stops sampling when the Effective Sample Size (ESS) reaches a target |
| `MomentumConsensusStop` | Stops training when gradient–momentum agreement drops |
| `AdaptiveRefinementStop` | Stop criterion based on split-batch gradient agreement EMA |
| `WeightedPositions` | Boltzmann-weighted position noiser for the diffusion model |
| `GODiffLogger` | TensorBoard logger (energies, ESS, timing, analysis figures) |

---

## Running the tests

```bash
<!-- CLONE package -->
git clone https://github.com/nronne/go-diff.git 
cd go-diff

<!-- INSTALL with pip including test extras-->
pip install ".[test]"

<!-- RUN pytest -->
pytest
```

---


## Citation

If you use GO-Diff please cite:
> Ronne, Nikolaj, Tejs Vegge and Arghya Bhowmik. "GO-Diff: Data-free
> and amortized global structure optimization"
> arXiv:2510.13448. Preprint, arXiv, October 15, 2025. 
> <https://arxiv.org/abs/2510.13448>

> Ronne, Nikolaj, and Bjork Hammer. "Atomistic Generative Diffusion for Materials
> Modeling." arXiv:2507.18314. Preprint, arXiv, July 24, 2025.
> <https://arxiv.org/abs/2507.18314>

If studying surface-supported systems please also cite:
> Ronne, Nikolaj, Alan Aspuru-Guzik, and Bjork Hammer. "Generative Diffusion
> Model for Surface Structure Discovery." Physical Review B 110, no. 23 (2024):
> 235427. <https://doi.org/10.1103/PhysRevB.110.235427>

Optionally, if using any AGOX functionality please cite:
> Christiansen, Mads-Peter V., Nikolaj Ronne, and Bjork Hammer. "Atomistic
> Global Optimization X: A Python Package for Optimization of Atomistic
> Structures." The Journal of Chemical Physics 157, no. 5 (2022): 054701.
> <https://doi.org/10.1063/5.0094165>
