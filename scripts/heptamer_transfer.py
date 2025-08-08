import torch
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.build import fcc111, surface
from ase.io import write, Trajectory
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import schnetpack as spk
from agedi.models.schnetpack import PositionsScore, SchNetPackTranslator

from agedi.diffusion import Diffusion
from agedi.data import Dataset, AtomsGraph
from agedi.models import ScoreModel
from agedi.diffusion.noisers.weighted_pos import WeightedPositionsNoiser
from agedi.diffusion.distributions import TruncatedNormal, UniformCellConfined
from agedi.cli.train import get_conditioning



##### HYPERPARAMETERS #####
name = __file__.split('/')[-1].split('.')[0]  # use the filename as the name of the experiment

n_epochs_per_loop = 4000        # number of epochs per training loop
temperatures = np.exp(np.linspace(np.log(3.0),np.log(0.02), 20))  # log spaced temperatures
samples_per_stage = 64         # number of evaluated samples per loop
buffer_size = 64
min_E = -500                    # to avoid false minimas in the MACE potential
ckpt_path = "stepped_Pt_T0.02.ckpt"

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

##### CALCULATOR #####
from mace.calculators import mace_mp
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')

##### TEMPLATE #####
def get_template():
    atoms = fcc111('Pt', size=(6, 6, 2), vacuum=8.0)
    atoms.positions[:, 2] -= atoms.positions[:, 2].min()
    return atoms


##### POSTERIOR SAMPLER #####
def sample(diffusion,  N):
    diffusion.eval()
    diffusion.to(torch.device("cuda"))
    t = get_template()
    z_max = t.positions[:, 2].max()
    confinement = [z_max + confinement_above_zmax[0], z_max + confinement_above_zmax[1]]

    template = AtomsGraph.from_atoms(t, initialize_mask=False)
    template.confinement = torch.tensor(confinement, dtype=torch.float32).reshape(1,2)

    with torch.no_grad():
        graph_list = diffusion.sample(
            N=N,
            batch_size=201,
            template=template,
            steps=sampling_steps,
            eps=0.005,
            save_path=False,
            progress_bar=False,
            n_atoms=n_atoms,
            positions=np.zeros((n_atoms, 3)),
            cell=t.get_cell(),
            atomic_numbers=atomic_numbers,
            confinement=confinement,
        )

    atoms_list = [g.to_atoms() for g in graph_list]
    return atoms_list

def create_data_posterior(diffusion, N=100):
    data = sample(diffusion, N)
    template = get_template()
    
    for atoms in data:
        atoms.set_constraint(FixAtoms(mask=[atom.index for atom in atoms if atom.index < len(template)]))
        
    return data

###### MODEL SETUP #####
def get_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    input_modules = [
        spk.atomistic.PairwiseDistances(),
    ]

    translator = SchNetPackTranslator(input_modules=input_modules)

    representation = spk.representation.PaiNN(
        n_atom_basis=feature_size,
        n_interactions=n_blocks,
        radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    conditionings = get_conditioning('none')

    cond_features = sum([c.output_dim for c in conditionings])
    heads = [
        PositionsScore(input_dim_scalar=feature_size+cond_features),
    ]
    noisers = [WeightedPositionsNoiser(distribution=TruncatedNormal(), prior=UniformCellConfined()), ]

    score_model = ScoreModel(
        translator=translator,
        representation=representation,
        conditionings=conditionings,
        heads=[h.to(device) for h in heads],
    )

    diffusion = Diffusion(
        score_model=score_model,
        noisers=noisers,
        optim_config={"lr": lr},
        scheduler_config={"factor": lr_factor, "patience": lr_patience},
    )
    return diffusion

##### TRAINER KWARGS #####
def get_trainer(index):
    # Training
    logger = TensorBoardLogger(save_dir='logs', name=name, version=index)

    # save hparams:
    logger.log_hyperparams({
        'n_epochs_per_loop': n_epochs_per_loop,
        'temperatures': temperatures.tolist(),
        'samples_per_stage': samples_per_stage,
        'min_E': min_E,
        'ckpt_path': ckpt_path,
        'cutoff': cutoff,
        'feature_size': feature_size,
        'n_blocks': n_blocks,
        'batch_size': batch_size,
        'lr': lr,
        'lr_factor': lr_factor,
        'lr_patience': lr_patience,
        'sampling_steps': sampling_steps,
        'n_atoms': n_atoms,
        'atomic_numbers': atomic_numbers,
        'confinement_above_zmax': confinement_above_zmax
    })

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="val_loss",
            filename="best_model",
            save_top_k=1,
            mode="min",
        ),
        ModelCheckpoint(
            filename="last_model",
            monitor=None,
            save_top_k=1,
            every_n_epochs=1,
        ),
    ]
    
    trainer_kwargs = dict(
        accelerator="auto",
        devices=1,
        max_epochs=n_epochs_per_loop,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=10.0,
        enable_progress_bar=False,
        log_every_n_steps=10,
        inference_mode=False,
        max_time={"hours": 64},
    )

    trainer = Trainer(**trainer_kwargs)
    return trainer

###### UTILITY FUNCTIONS #####
def save_traj(data, e, path=None, writer=None):
    assert path is not None or writer is not None, "Either path or writer must be provided"
    
    traj = [d.copy() for d in data]
    for d, energy in zip(traj, e):
        d.calc = SPC(d, energy=energy)

    argsort = np.argsort(e)
    traj = [traj[i] for i in argsort]

    if writer is not None:
        for d in traj:
            writer.write(d)
    if path is not None:
        write(path, traj)

def check_min_dist(data, min_dist=1.0):
    """Check if the minimum distance between atoms is larger than min_dist."""
    new_data = []
    for atoms in data:
        positions = atoms.get_positions()
        dists = np.linalg.norm(positions[:, np.newaxis] - positions, axis=-1)
        np.fill_diagonal(dists, np.inf)  # Ignore self-distances
        if np.min(dists) >= min_dist:
            new_data.append(atoms)
    return new_data

def weights(Es, T):
    Es = np.array(Es)
    Es_scaled = -Es / T
    Es_shifted = Es_scaled - np.max(Es_scaled)
    exp_Es = np.exp(Es_shifted)
    weights = exp_Es / np.sum(exp_Es) * len(Es)

    # weights = np.exp(-Es/T) / np.sum(np.exp(-Es/T)) * len(data)
    print('Weights:', weights)

    properties = []
    for w in weights:
        properties.append({'weight': w})

    return properties

def eval(data, T):
    Es = []
    for d in data:
        d.calc = calc
        Es.append(d.get_potential_energy())

    Es = np.array(Es)
    print('Energies:', Es)

    return Es

###### BUFFER #####
def get_buffer(data, energies, T, N):
    data = [atoms for atoms, energy in zip(data, energies) if energy < 0.0]
    energies = [energy for energy in energies if energy < 0.0]

    if len(data) < N:
        print(f"Not enough data for temperature {T}, using all available data.")
        return data, energies

    w = weights(energies, T)
    w = np.array([prop['weight'] for prop in w])
    k = np.random.uniform(size=len(data))**(1/w)
    sorter = np.argsort(k)[::-1]  # sort in descending order
    return [data[i] for i in sorter[:N]], np.array([energies[i] for i in sorter[:N]])
    

##### STAGES #####
def sample_stage(T, diffusion, all_data, all_energies, energy_cut, data_writer, logdir):
    new_data = create_data_posterior(diffusion, samples_per_stage)
    new_data = check_min_dist(new_data, min_dist=1.0)
    e = eval(new_data, T)
    save_traj(new_data, e, path=str(logdir/f"new_data_T{T:.2f}.traj"), writer=data_writer)
    
    # filter out energies below min_E
    new_data = [atoms for atoms, energy in zip(new_data, e) if energy > min_E]
    e = [energy for energy in e if energy > min_E]

    save_traj(new_data, e, writer=data_writer)

    all_data.extend(new_data)
    all_energies.extend(e)

    buffer, buffer_energies = get_buffer(all_data, all_energies, T, N=buffer_size)
    w = weights(buffer_energies, T)

    save_traj(buffer, buffer_energies, path=str(logdir/f"buffer_T{T:.2f}.traj"))
    return buffer, w, all_data, all_energies, energy_cut

def train_stage(T, diffusion, trainer, buffer, w, confinement):
    print(f"Training Stage: {T}")
    dataset = Dataset(
        cutoff=cutoff,
        batch_size=batch_size,
        n_train=0.9,
        n_val=0.1,
    )
    dataset.add_atoms_data(buffer, mask_method="MaskFixed", properties=w, confinement=confinement)
    dataset.setup()

    trainer.fit(diffusion, dataset)
    print("Training Stage Finished. Saving model...")
    trainer.save_checkpoint(str(Path(trainer.log_dir)/f"diffusion_model_T{T:.2f}.ckpt"))

    #increase the max number of epochs for Trainer
    trainer.fit_loop.max_epochs += n_epochs_per_loop

    return diffusion, trainer
    
##### TRAINING FUNCTION #####
def train(index):
    diffusion = get_model()

    if ckpt_path is not None:
        print(f"Loading pre-trained model from {ckpt_path}")
        diffusion.load_state_dict(torch.load(ckpt_path)['state_dict'])
    
    trainer = get_trainer(index)

    # Ensure Logging directory exists
    logdir = Path(trainer.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logging directory: {logdir}")

    data_writer = Trajectory(str(logdir/"data.traj"), mode='w')

    t = get_template()
    z_max = t.positions[:, 2].max()
    confinement = [z_max + confinement_above_zmax[0], z_max + confinement_above_zmax[1]]
    
    all_data, all_energies = [], []
    energy_cut = 0.0
    for T in temperatures:
        buffer, w, all_data, all_energies, energy_cut = sample_stage(
            T,
            diffusion,
            all_data,
            all_energies,
            energy_cut,
            data_writer,
            logdir
        )
        
        diffusion, trainer = train_stage(T, diffusion, trainer, buffer, w, confinement)

    print("Training completed.")

    
if __name__ == "__main__":
    from argparse import ArgumentParser
    from lightning.pytorch import seed_everything
    parser = ArgumentParser(description="Train a diffusion model for GO tasks.")
    parser.add_argument('-i', '--index', type=int, default=0, help="Index for the training run.")
    args = parser.parse_args()
    index = args.index
    seed_everything(index)

    train(index)

