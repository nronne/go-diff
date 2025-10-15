import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.io import write, Trajectory

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import schnetpack as spk
from agedi.models.schnetpack import PositionsScore, SchNetPackTranslator
from agedi.models.schnetpack.regressor_heads import Forces

from agedi.diffusion.diffusion import Diffusion
from agedi.data import Dataset, AtomsGraph
from agedi.models import ScoreModel
from agedi.models.regressor import RegressorModel
from agedi.diffusion.noisers.weighted_pos import WeightedPositionsNoiser
from agedi.diffusion.distributions import TruncatedNormal, UniformCellConfined
from agedi.cli.train import get_conditioning


class GODiff:
    """
    Gradient-Optimized Diffusion model for atomic systems.
    Implements an iterative training procedure with temperature annealing.
    """
    
    def __init__(self, 
                 calculator,
                 template,
                 atomic_numbers,                 
                 name="godiff",
                 n_epochs_per_loop=2000,
                 n_steps_per_loop=8000,
                 temperatures=None,
                 samples_per_stage=128,
                 buffer_size=64,
                 min_E=-500,
                 ckpt_path=None,
                 cutoff=6.0,
                 feature_size=64,
                 n_blocks=4,
                 batch_size=16,
                 lr=1e-4,
                 lr_factor=0.95,
                 lr_patience=100,
                 sampling_steps=500,
                 confinement_above_zmax=[1.0, 6.0],
                 force_field_guidance=0.2,
                 max_extra_steps=0,
                 force_threshold=0.05,
                 use_weighting=False,
                 device="cuda"):
        """
        Initialize the GODiff model.
        
        Parameters:
        -----------
        calculator : object
            ASE-compatible calculator for energy and force evaluations
        template_atoms : ASE Atoms or None
            Template structure for sampling (e.g., surface). If None, creates a default Pt(111) surface
        name : str
            Name of the experiment for logging
        n_epochs_per_loop : int
            Number of epochs per training loop
        n_steps_per_loop : int
            Number of steps per training loop
        temperatures : list or None
            List of temperatures for annealing; if None, uses default log-spaced values
        samples_per_stage : int
            Number of evaluated samples per loop
        buffer_size : int
            Size of the replay buffer for each temperature
        min_E : float
            Minimum energy threshold to filter out false minima
        ckpt_path : str or None
            Path to checkpoint for resuming training
        cutoff : float
            Cutoff distance for the neighbor list
        feature_size : int
            Size of the feature vector for each atom
        n_blocks : int
            Number of interaction blocks in the representation
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        lr_factor : float
            Learning rate decay factor for scheduler
        lr_patience : int
            Patience for learning rate scheduler
        sampling_steps : int
            Number of sampling steps for the posterior sampler
        n_atoms : int
            Number of atoms in the optimization
        atomic_numbers : list or None
            Atomic numbers of the atoms; if None, defaults to [78] * n_atoms
        confinement_above_zmax : list
            Confinement range above the maximum z position of the template
        force_field_guidance : float
            Scale of force field guidance during sampling
        max_extra_steps : int
            Maximum number of extra relaxation steps
        force_threshold : float
            Force threshold for termination of relaxation
        use_weighting : bool
            Whether to use weighting in the regressor model loss
        device : str
            Computation device ('cuda' or 'cpu')
        """
        self.calculator = calculator
        self.template_atoms = template
        self.name = name
        self.n_epochs_per_loop = n_epochs_per_loop
        self.n_steps_per_loop = n_steps_per_loop
        
        if temperatures is None:
            self.temperatures = np.exp(np.linspace(np.log(5.0), np.log(0.02), 20))
        else:
            self.temperatures = temperatures
            
        self.samples_per_stage = samples_per_stage
        self.buffer_size = buffer_size
        self.min_E = min_E
        self.ckpt_path = ckpt_path
        
        self.cutoff = cutoff
        self.feature_size = feature_size
        self.n_blocks = n_blocks
        
        self.batch_size = batch_size
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        
        self.sampling_steps = sampling_steps
        self.atomic_numbers = atomic_numbers
        self.n_atoms = len(self.atomic_numbers)
            
        self.confinement_above_zmax = confinement_above_zmax
        self.force_field_guidance = force_field_guidance
        self.max_extra_steps = max_extra_steps
        self.force_threshold = force_threshold
        self.use_weighting = use_weighting
        self.device = device
        
        self.diffusion = None
        self.template = None
        self.confinement = None
        
        # Initialize model and template
        self._init_template(template)
        self._init_model()
        
    def _init_template(self, template_atoms):
        """Initialize the template structure and confinement."""
        # Create default template if none was provided
        # Calculate confinement based on template
        z_max = template_atoms.positions[:, 2].max()
        self.confinement = [
            z_max + self.confinement_above_zmax[0], 
            z_max + self.confinement_above_zmax[1]
        ]
        
        # Create AtomsGraph from template atoms
        self.template = AtomsGraph.from_atoms(template_atoms, initialize_mask=False)
        self.template.confinement = torch.tensor(self.confinement, dtype=torch.float32).reshape(1, 2)
    
    def _init_model(self):
        """Initialize the diffusion model with score and regressor components."""
        device = torch.device(self.device)

        # SchNetPack input modules
        input_modules = [
            spk.atomistic.PairwiseDistances(),
        ]

        # Create translator, representation and conditioning
        translator = SchNetPackTranslator(input_modules=input_modules)
        representation = spk.representation.PaiNN(
            n_atom_basis=self.feature_size,
            n_interactions=self.n_blocks,
            radial_basis=spk.nn.GaussianRBF(n_rbf=30, cutoff=self.cutoff),
            cutoff_fn=spk.nn.CosineCutoff(self.cutoff),
        )
        conditionings = get_conditioning('none')

        # Create score model
        cond_features = sum([c.output_dim for c in conditionings])
        heads = [
            PositionsScore(input_dim_scalar=self.feature_size + cond_features),
        ]
        noisers = [WeightedPositionsNoiser(distribution=TruncatedNormal(), prior=UniformCellConfined())]

        score_model = ScoreModel(
            translator=translator,
            representation=representation,
            conditionings=conditionings,
            heads=[h.to(device) for h in heads],
        )

        # Create regressor model for forces
        if self.force_field_guidance > 0:
            regressor = RegressorModel(
                translator=translator,
                representation=representation,
                heads=[Forces()],
                use_weighting=self.use_weighting,
            )
        else:
            regressor = None

        # Create diffusion model
        self.diffusion = Diffusion(
            score_model=score_model,
            regressor_model=regressor,
            noisers=noisers,
            optim_config={"lr": self.lr},
            scheduler_config={"factor": self.lr_factor, "patience": self.lr_patience},
        )

        # Load checkpoint if provided
        if self.ckpt_path is not None:
            print(f"Loading pre-trained model from {self.ckpt_path}")
            self.diffusion.load_state_dict(torch.load(self.ckpt_path)['state_dict'])
        
        self.diffusion = self.diffusion.to(device)
    
    def get_trainer(self, index):
        """Create a PyTorch Lightning trainer.
        
        Parameters:
        -----------
        index : int
            Run index for logging
            
        Returns:
        --------
        trainer : lightning.Trainer
            Configured trainer
        """
        # Create logger
        logger = TensorBoardLogger(save_dir='logs', name=self.name, version=index)

        # Log hyperparameters
        logger.log_hyperparams({
            'n_epochs_per_loop': self.n_epochs_per_loop,
            'n_steps_per_loop': self.n_steps_per_loop,
            'temperatures': self.temperatures.tolist(),
            'samples_per_stage': self.samples_per_stage,
            'min_E': self.min_E,
            'ckpt_path': self.ckpt_path,
            'cutoff': self.cutoff,
            'feature_size': self.feature_size,
            'n_blocks': self.n_blocks,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'lr_factor': self.lr_factor,
            'lr_patience': self.lr_patience,
            'sampling_steps': self.sampling_steps,
            'n_atoms': self.n_atoms,
            'atomic_numbers': self.atomic_numbers,
            'confinement_above_zmax': self.confinement_above_zmax,
            'force_field_guidance': self.force_field_guidance,
            'max_extra_steps': self.max_extra_steps,
            'force_threshold': self.force_threshold
        })

        # Set up callbacks
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
            max_epochs=0,  # Will be incremented during training
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=10.0,
            enable_progress_bar=False,
            log_every_n_steps=10,
            inference_mode=False,
            max_time={"hours": 64},
        )

        return Trainer(**trainer_kwargs)
    
    def sample(self, N=1, guidance=None, progress_bar=False):
        """Sample structures using the diffusion model.
        
        Parameters:
        -----------
        N : int
            Number of structures to sample
        guidance : float or None
            Force field guidance strength; if None, uses self.force_field_guidance
        progress_bar : bool
            Whether to display a progress bar
            
        Returns:
        --------
        list of ASE Atoms
            Sampled structures
        """
        if guidance is None:
            guidance = self.force_field_guidance
            
        self.diffusion.eval()
        
        with torch.no_grad():
            graph_list = self.diffusion.sample(
                N=N,
                template=self.template,
                steps=self.sampling_steps,
                eps=0.005,
                save_path=False,
                progress_bar=progress_bar,
                n_atoms=self.n_atoms,
                positions=np.zeros((self.n_atoms, 3)),
                cell=self.template_atoms.get_cell(),
                atomic_numbers=self.atomic_numbers,
                confinement=self.confinement,
                force_field_guidance=guidance,
                force_threshold=self.force_threshold,
                max_extra_steps=self.max_extra_steps,
            )

        atoms_list = [g.to_atoms() for g in graph_list]

        # Add constraints to template atoms
        template_len = len(self.template_atoms)
        for atoms in atoms_list:
            atoms.set_constraint(FixAtoms(mask=[atom.index for atom in atoms 
                                                if atom.index < template_len]))
        
        return atoms_list
    
    def evaluate(self, atoms_list):
        """Evaluate energies and forces for a list of structures.
        
        Parameters:
        -----------
        atoms_list : list of ASE Atoms
            Structures to evaluate
            
        Returns:
        --------
        energies : list of float
            Potential energies
        forces : list of numpy.ndarray
            Atomic forces
        """
        energies = []
        forces_list = []
        
        for atoms in atoms_list:
            atoms.calc = self.calculator
            energies.append(atoms.get_potential_energy())
            forces_list.append(atoms.get_forces(apply_constraint=False))
            
        return np.array(energies), forces_list
    
    def check_min_dist(self, atoms_list, min_dist=1.0):
        """Filter out structures with too small interatomic distances.
        
        Parameters:
        -----------
        atoms_list : list of ASE Atoms
            Structures to filter
        min_dist : float
            Minimum allowed distance between atoms
            
        Returns:
        --------
        list of ASE Atoms
            Filtered structures
        """
        filtered_list = []
        for atoms in atoms_list:
            positions = atoms.get_positions()
            dists = np.linalg.norm(positions[:, np.newaxis] - positions, axis=-1)
            np.fill_diagonal(dists, np.inf)  # Ignore self-distances
            if np.min(dists) >= min_dist:
                filtered_list.append(atoms)
        return filtered_list
    
    def compute_weights(self, energies, forces, temperature):
        """Compute Boltzmann weights for structures at a given temperature.
        
        Parameters:
        -----------
        energies : list or array
            Potential energies
        forces : list of arrays
            Atomic forces
        temperature : float
            Temperature for Boltzmann weighting
            
        Returns:
        --------
        list of dict
            Properties including weights and forces
        """
        energies = np.array(energies)
        # Scale and shift energies for numerical stability
        Es_scaled = -energies / temperature
        Es_shifted = Es_scaled - np.max(Es_scaled)
        exp_Es = np.exp(Es_shifted)
        weights = exp_Es / np.sum(exp_Es) * len(energies)

        properties = []
        for w, f in zip(weights, forces):
            properties.append({'weight': w, 'forces': f})

        return properties
    
    def save_trajectory(self, atoms_list, energies, forces, path=None, writer=None):
        """Save structures to a trajectory file.
        
        Parameters:
        -----------
        atoms_list : list of ASE Atoms
            Structures to save
        energies : list or array
            Potential energies
        forces : list of arrays
            Atomic forces
        path : str or None
            File path to save trajectory
        writer : ASE Trajectory or None
            Trajectory writer object
        """
        assert path is not None or writer is not None, "Either path or writer must be provided"

        # Create copies to avoid modifying original structures
        traj = [atoms.copy() for atoms in atoms_list]
        
        # Add energy and forces as calculators
        for atoms, energy, force in zip(traj, energies, forces):
            atoms.calc = SPC(atoms, energy=energy, forces=force)

        # Sort by energy
        argsort = np.argsort(energies)
        traj = [traj[i] for i in argsort]

        # Write trajectory
        if writer is not None:
            for atoms in traj:
                writer.write(atoms)
                
        if path is not None:
            write(path, traj)
    
    def get_buffer(self, data, energies, forces, temperature):
        """Create a buffer of structures selected by Boltzmann weighting.
        
        Parameters:
        -----------
        data : list of ASE Atoms
            Candidate structures
        energies : list or array
            Potential energies
        forces : list of arrays
            Atomic forces
        temperature : float
            Temperature for Boltzmann weighting
            
        Returns:
        --------
        tuple
            (buffer_structures, buffer_energies, buffer_forces, weighted_properties)
        """
        # Filter out structures with positive energy
        valid_idx = [i for i, e in enumerate(energies) if e < 0.0]
        valid_data = [data[i] for i in valid_idx]
        valid_energies = [energies[i] for i in valid_idx]
        valid_forces = [forces[i] for i in valid_idx]
        
        if not valid_data:
            print(f"No valid structures found with negative energy at T={temperature}")
            return [], [], [], []
        
        if len(valid_data) <= self.buffer_size:
            print(f"Not enough data for T={temperature}, using all {len(valid_data)} available structures.")
            weighted_props = self.compute_weights(valid_energies, valid_forces, temperature)
            return valid_data, valid_energies, valid_forces, weighted_props
        
        # Use stochastic prioritized sampling based on weights
        weighted_props = self.compute_weights(valid_energies, valid_forces, temperature)
        weights = np.array([prop['weight'] for prop in weighted_props])
        
        # Prioritized sampling with randomness
        k = np.random.uniform(size=len(valid_data))**(1/weights)
        sorter = np.argsort(k)[::-1]  # Sort in descending order
        
        # Select top buffer_size structures
        buffer_data = [valid_data[i] for i in sorter[:self.buffer_size]]
        buffer_energies = np.array([valid_energies[i] for i in sorter[:self.buffer_size]])
        buffer_forces = [valid_forces[i] for i in sorter[:self.buffer_size]]
        buffer_props = [weighted_props[i] for i in sorter[:self.buffer_size]]
        
        return buffer_data, buffer_energies, buffer_forces, buffer_props
    
    def sample_stage(self, temperature, guidance, all_data, all_energies, all_forces, 
                     energy_cut, data_writer, logdir):
        """Run a sampling stage at a specific temperature."""
        # Sample new structures
        new_data = self.sample(N=self.samples_per_stage, guidance=guidance)
        new_data = self.check_min_dist(new_data, min_dist=1.0)
        
        # Evaluate energies and forces
        new_energies, new_forces = self.evaluate(new_data)
        self.save_trajectory(
            new_data, 
            new_energies, 
            new_forces, 
            path=str(logdir/f"new_data_T{temperature:.2f}.traj"), 
            writer=data_writer
        )
        
        # Filter out structures with energy below min_E
        valid_idx = [i for i, e in enumerate(new_energies) if e > self.min_E]
        filtered_data = [new_data[i] for i in valid_idx]
        filtered_energies = [new_energies[i] for i in valid_idx]
        filtered_forces = [new_forces[i] for i in valid_idx]

        # Save filtered structures
        self.save_trajectory(filtered_data, filtered_energies, filtered_forces, writer=data_writer)

        # Update global data
        all_data.extend(filtered_data)
        all_energies.extend(filtered_energies)
        all_forces.extend(filtered_forces)

        # Create buffer for training
        buffer, buffer_energies, buffer_forces, weighted_props = self.get_buffer(
            all_data, all_energies, all_forces, temperature
        )
        
        # Save buffer
        self.save_trajectory(
            buffer, 
            buffer_energies, 
            buffer_forces, 
            path=str(logdir/f"buffer_T{temperature:.2f}.traj")
        )
        
        return buffer, weighted_props, all_data, all_energies, all_forces, energy_cut

    def train_diffusion_stage(self, temperature, trainer, buffer, weighted_props):
        """Train the diffusion model at a specific temperature."""
        print(f"\n=== Training Diffusion Model at T={temperature:.4f} ===")
        
        # Increment max_epochs based on buffer size and steps per loop
        steps_per_epoch = len(buffer) // self.batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        epochs_to_add = max(1, self.n_steps_per_loop // steps_per_epoch)
        trainer.fit_loop.max_epochs += epochs_to_add
        
        print(f"Training for {epochs_to_add} epochs ({steps_per_epoch} steps/epoch)")

        # Create dataset
        dataset = Dataset(
            cutoff=self.cutoff,
            batch_size=self.batch_size,
            n_train=0.9,
            n_val=0.1,
        )
        dataset.add_atoms_data(
            buffer, 
            mask_method="MaskFixed", 
            properties=weighted_props, 
            confinement=self.confinement
        )
        dataset.setup()

        # Train diffusion model
        self.diffusion.regressor_training = False
        trainer.fit(self.diffusion, dataset)
        
        # Save checkpoint
        ckpt_path = str(Path(trainer.log_dir) / f"diffusion_model_T{temperature:.2f}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"Saved diffusion model checkpoint to {ckpt_path}")

        return self.diffusion, trainer

    def train_regressor_stage(self, temperature, trainer, data, energies, forces):
        """Train the regressor model at a specific temperature."""
        print(f"\n=== Training Force Regressor at T={temperature:.4f} ===")
        
        # Increment max_epochs based on data size and steps per loop
        steps_per_epoch = len(data) // self.batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        epochs_to_add = max(1, self.n_steps_per_loop // steps_per_epoch)
        trainer.fit_loop.max_epochs += epochs_to_add
        
        print(f"Training for {epochs_to_add} epochs ({steps_per_epoch} steps/epoch)")

        # Create dataset with energies and forces
        dataset = Dataset(
            cutoff=self.cutoff,
            batch_size=self.batch_size,
            n_train=0.9,
            n_val=0.1,
        )
        
        properties = []
        for e, f in zip(energies, forces):
            properties.append({'energy': e, 'forces': f})
            
        dataset.add_atoms_data(data, mask_method="MaskFixed", properties=properties)
        dataset.setup()

        # Set regressor training mode
        self.diffusion.regressor_training = True
        trainer.fit(self.diffusion, dataset)
        
        # Save checkpoint
        ckpt_path = str(Path(trainer.log_dir) / f"regressor_model_T{temperature:.2f}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"Saved regressor model checkpoint to {ckpt_path}")

        # Reset regressor training mode
        self.diffusion.regressor_training = False

        return self.diffusion, trainer

    def run(self, run_index=0):
        """Run the complete training procedure with temperature annealing."""
        # Initialize trainer
        trainer = self.get_trainer(run_index)
        
        # Ensure logging directory exists
        logdir = Path(trainer.log_dir)
        logdir.mkdir(parents=True, exist_ok=True)
        print(f"Logging directory: {logdir}")

        # Initialize trajectory writer for all data
        data_writer = Trajectory(str(logdir/"all_data.traj"), mode='w')

        # Initialize storage for accumulated data
        all_data, all_energies, all_forces = [], [], []
        energy_cut = 0.0
        
        # Main training loop across temperatures
        for i, temperature in enumerate(self.temperatures):
            print(f"\n{'='*50}")
            print(f"STAGE {i+1}/{len(self.temperatures)}: Temperature = {temperature:.4f}")
            print(f"{'='*50}")
            
            # Set guidance level (0 for first stage, then use configured value)
            guidance = 0.0 if i == 0 else self.force_field_guidance
            
            # Track iteration number
            if hasattr(self.diffusion, 'iteration'):
                self.diffusion.iteration = i
            
            # Sample at current temperature
            buffer, weighted_props, all_data, all_energies, all_forces, energy_cut = self.sample_stage(
                temperature,
                guidance,
                all_data,
                all_energies,
                all_forces,
                energy_cut,
                data_writer,
                logdir
            )
            
            if not buffer:
                print(f"Warning: Empty buffer at T={temperature}, skipping training")
                continue
                
            # Train regressor and diffusion models
            if self.force_field_guidance > 0:            
                self.diffusion, trainer = self.train_regressor_stage(
                    temperature, trainer, all_data, all_energies, all_forces
                )

            self.diffusion, trainer = self.train_diffusion_stage(
                temperature, trainer, buffer, weighted_props
            )

        print("\nTraining completed.")
        
        # Save final model
        final_ckpt = str(logdir / "final_model.ckpt")
        trainer.save_checkpoint(final_ckpt)
        print(f"Final model saved to: {final_ckpt}")
        
        return final_ckpt
