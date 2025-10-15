from argparse import ArgumentParser
from mace.calculators import mace_mp
from go_diff import GODiff
from lightning.pytorch import seed_everything
from ase.build import fcc111
import numpy as np

parser = ArgumentParser(description="Train a diffusion model for GO tasks.")
parser.add_argument('-i', '--index', type=int, default=0, help="Index for the training run.")
args = parser.parse_args()
index = args.index
seed_everything(index)


##### HYPERPARAMETERS #####
name = __file__.split('/')[-1].split('.')[0]  # use the filename as the name of the experiment

n_steps_per_loop = 8000         # number of steps per training loop
temperatures = np.exp(np.linspace(np.log(5.0),np.log(0.02), 20))  # log spaced temperatures
samples_per_stage = 128         # number of evaluated samples per loop
buffer_size = 64
min_E = -500                    # to avoid false minimas in the MACE potential
ckpt_path = None

cutoff = 6.0                    # cutoff for the neighbor list
feature_size = 64               # size of the feature vector for each atom
n_blocks = 4                    # number of interaction blocks in the PaiNN representation

batch_size = 16                 # batch size for training
lr = 1e-4                       # learning rate
lr_factor = 0.95                # learning rate factor for the scheduler
lr_patience = 100               # patience for the learning rate scheduler

sampling_steps = 500           # number of sampling steps for the posterior sampler
n_atoms = 7                     # number of atoms in the optimization
atomic_numbers = [78] * n_atoms # atomic numbers of the atoms in the optimization (e.g., [78] for Pt)
confinement_above_zmax = [1.0, 6.0]  # confinement above the maximum z position of the template 

force_field_guidance = 0.2


##### CALCULATOR #####
from mace.calculators import mace_mp
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')

##### TEMPLATE #####
def get_template():
    atoms = fcc111('Pt', size=(6, 6, 2), vacuum=8.0)
    atoms.positions[:, 2] -= atoms.positions[:, 2].min()
    return atoms

template = get_template()
    
godiff = GODiff(
    calculator=calc,
    template=template,
    atomic_numbers=atomic_numbers,
    name=name,
    n_steps_per_loop=n_steps_per_loop,
    temperatures=temperatures,
    samples_per_stage=samples_per_stage,
    buffer_size=buffer_size,
    min_E=min_E,
    ckpt_path=ckpt_path,
    cutoff=cutoff,
    feature_size=feature_size,
    n_blocks=n_blocks,
    batch_size=batch_size,
    lr=lr,
    lr_factor=lr_factor,
    lr_patience=lr_patience,
    sampling_steps=sampling_steps,
    confinement_above_zmax=confinement_above_zmax,
    force_field_guidance=force_field_guidance,
)

# Train the model
godiff.run(run_index=index)



