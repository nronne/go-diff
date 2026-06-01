# GO-Diff

**GO-Diff** (Gradient-Optimised Diffusion) is a generative diffusion framework for
atomistic structure search.  It couples a learned diffusion model with an
energy/force calculator (e.g. MACE) and iteratively refines the model through
Boltzmann-weighted training, adaptive temperature annealing, and an adaptive
replay buffer.

---

## Installation

```bash
pip install go-diff
```

### (Optional) Reproduce paper results

```bash
pip install mace-torch
pip install "agox[full]"
```

---

## Quick Start

```python
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
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")

# 2. Substrate template (atoms to keep fixed)
template = surface("Pt", (1, 2, 2), 5, vacuum=8.0)
template.positions[:, 2] -= template.positions[:, 2].min()
confinement = [0.0, 4.0 + template.positions[:, 2].max()]

# 3. Diffusion model
diffusion = create_diffusion(noisers=(WeightedPositions(),))

# 4. GO-Diff optimiser
godiff = GODiff(
    calculator=calc,
    diffusion=diffusion,
    temperature_schedule=TemperatureSchedule(fast=0.5, slow=0.9),
    sample_controller=SampleController(initial_N=16, target_ess=8),
    training_controller=MomentumConsensusStop(min_steps=100, patience=250, drop_factor=0.9),
    sample_config={
        "template": template,
        "atomic_numbers": [78],        # one Pt adatom
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
```

See `scripts/example_script.py` for the full script and `scripts/` for all
paper reproduction scripts.

---

## Key components

| Class / module | Description |
|---|---|
| `GODiff` | Main optimisation loop (sample → evaluate → buffer → train) |
| `TemperatureSchedule` | Adaptive annealing; `fast` / `slow` cooling factors based on heat capacity |
| `SampleController` | Stops sampling when the Effective Sample Size (ESS) reaches a target |
| `MomentumConsensusStop` | Stops training when gradient–momentum agreement drops |
| `AdaptiveRefinementStop` | Alternative stop criterion based on split-batch gradient agreement EMA |
| `WeightedPositions` | Boltzmann-weighted position noiser for the diffusion model |
| `GODiffLogger` | TensorBoard logger (energies, ESS, timing, analysis figures) |

---

## Running the tests

```bash
pip install ".[test]"
pytest
```

---

## Scripts

Scripts to reproduce the results in the paper are in `scripts/`.  Utilities for
identifying the Pt-heptamer structure are in `utils/`.

---

## Citation

If you use GO-Diff please cite:

> Ronne, Nikolaj, and Bjork Hammer. "Atomistic Generative Diffusion for Materials
> Modeling." arXiv:2507.18314. Preprint, arXiv, July 24, 2025.
> <https://arxiv.org/abs/2507.18314>

If studying surface-supported systems also cite:

> Ronne, Nikolaj, Alan Aspuru-Guzik, and Bjork Hammer. "Generative Diffusion
> Model for Surface Structure Discovery." Physical Review B 110, no. 23 (2024):
> 235427. <https://doi.org/10.1103/PhysRevB.110.235427>

Optionally, if using any AGOX functionality:

> Christiansen, Mads-Peter V., Nikolaj Ronne, and Bjork Hammer. "Atomistic
> Global Optimization X: A Python Package for Optimization of Atomistic
> Structures." The Journal of Chemical Physics 157, no. 5 (2022): 054701.
> <https://doi.org/10.1063/5.0094165>
