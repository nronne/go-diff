import numpy as np
from argparse import ArgumentParser

from ase.build import fcc111, surface
from mace.calculators import mace_mp

from agedi import create_diffusion

from go_diff import GODiff
from go_diff.controllors import SampleController, TemperatureController, MomentumConsensusStop
from go_diff.noisers import WeightedPositions

parser = ArgumentParser(description="Train a diffusion model for GO tasks.")
parser.add_argument('-i', '--index', type=int, default=0, help="Index for the training run.")
args = parser.parse_args()
index = args.index



##### HYPERPARAMETERS #####
name = __file__.split('/')[-1].split('.')[0]  # use the filename as the name of the experiment

initial_buffer_size = 16
min_E = -200                    # to avoid false minimas in the MACE potential

n_atoms = 1                     # number of atoms in the optimization
atomic_numbers = [78] * n_atoms # atomic numbers of the atoms in the optimization (e.g., [78] for Pt)
confinement_above_zmax = np.array([0.0, 4.0])  # confinement above the maximum z position of the template 


##### CALCULATOR #####
from mace.calculators import mace_mp
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')

##### TEMPLATE #####
template = surface('Pt', (1,2,2), 5, vacuum=8.0)
template.positions[:, 2] -= atoms.positions[:, 2].min()

confinement = list(confinement_above_zmax + template.positions[:, 2].max())

#### DIFFUSION MODEL #####
diffusion = create_diffusion(noisers=(WeightedPositions(),))

#### GO-DIFF #####


godiff = GODiff(
    calculator=calc,
    diffusion=diffusion,
    temperature_schedule=TemperatureSchedule(fast=0.5, slow=0.9),
    sample_controller=SampleController(initial_N=16, target_ess=8),
    training_controller=MomentumConsensusStop(min_steps=100, patience=250, drop_factor=0.9),
    sample_config={
        "template": template,
        "atomic_numbers": atomic_numbers,
    },
    dataset_config={
        "mask": "MaskFixed",
        "confinement": confinement,
    },
    initial_buffer_size=initial_buffer_size,
    min_E=min_E,
)

# Train the model
godiff.run(run_index=index, max_iterations=50)
