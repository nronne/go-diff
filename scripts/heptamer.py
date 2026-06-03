import numpy as np
from argparse import ArgumentParser

from ase.build import fcc111, surface
from mace.calculators import mace_mp

from agedi import AtomsGraph, create_diffusion
from agedi.diffusion import ForcefieldGuidanceConfig

from go_diff import GODiff, MinEnergyFilter
from go_diff.controllers import SampleController, TemperatureSchedule, MomentumConsensusStop
from go_diff.noisers import WeightedConfinedCellPositions

parser = ArgumentParser(description="Train a diffusion model for GO tasks.")
parser.add_argument('-i', '--index', type=int, default=0, help="Index for the training run.")
args = parser.parse_args()
index = args.index


##### HYPERPARAMETERS #####
name = __file__.split('/')[-1].split('.')[0]  # use the filename as the name of the experiment

initial_buffer_size = 16
min_E = -500                    # to avoid false minimas in the MACE potential
formula = "Pt7"
confinement_above_zmax = np.array([1.0, 6.0])  # confinement above the maximum z position of the template 


##### CALCULATOR #####
from mace.calculators import mace_mp
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')

##### TEMPLATE #####
template = fcc111('Pt', size=(6, 6, 2), vacuum=8.0)
template.positions[:, 2] -= template.positions[:, 2].min()


confinement = confinement_above_zmax + template.positions[:, 2].max()
template = AtomsGraph.from_atoms(template, confinement=confinement)


#### DIFFUSION MODEL #####
diffusion = create_diffusion(noisers=(WeightedConfinedCellPositions(),))

#### GO-DIFF #####


godiff = GODiff(
    calculator=calc,
    diffusion=diffusion,
    temperature_schedule=TemperatureSchedule(fast=0.5, slow=0.9),
    sample_controller=SampleController(initial_N=16, target_ess=8),
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
    after_potential_filter=[MinEnergyFilter(min_E)],
)

# Train the model
godiff.run(max_iterations=50)
