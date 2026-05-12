import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.io import write, Trajectory

# AGeDi
from agedi import create_diffusion, create_dataset, create_trainer, train, sample

# GO-Diff
from go_diff.controllers import TemperatureController, SampleController, MomentumConsensusStop
from .logger import GODiffLogger

class GODiff:
    """
    Gradient-Optimized Diffusion model for atomic systems.
    Implements an iterative training procedure with temperature annealing.
    """
    
    def __init__(
            self,
            calculator,
            diffusion,
            temperature_schedule=TemperatureSchedule(),
            sample_controller=SampleController(),
            training_controller=MomentumConsensusStop(), #AdaptiveRefinementStop(),
            sample_config={},
            dataset_config={},
            initial_buffer_size=16,
            min_E=-500,
            device="cuda"
    ):
        """
        Initialize the GO-Diff model.
        
        Parameters:
        -----------
        """
        self.calculator = calculator
        self.diffusion = diffusion
        self.temperature_schedule = temperature_schedule
        self.sample_controller = sample_controller
        self.training_controller = training_controller

        self.sample_config = sample_config
        self.dataset_config = dataset_config
        
        self._godiff_logger = None  # Initialized in run() once the TB writer is available
        self._flops_timing_cb = None  # Initialized in get_trainer()

        self.buffer_size = initial_buffer_size
        
        self.buffer = []
        self.all_data = []
        
        
    def get_trainer(self, max_time_hours=64):
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
        
        self._flops_timing_cb = FlopsAndTimingCallback()
        
        callbacks = [
            self.training_controller,
            self._flops_timing_cb,
        ]
        
        self.trainer = create_trainer(max_time={"hours": max_time_hours}, extra_callbacks=callbacks)

    
    def sample(self):
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

        atoms_list = sample(diffusion, **self.sample_kwargs)
        
        # Add constraints to template atoms
        if "template" in self.sample_kwargs and self.sample_kwargs["template"] is not None:
            template_len = len(self.template_atoms)
            for atoms in atoms_list:
                atoms.set_constraint(FixAtoms(mask=[atom.index for atom in atoms 
                                                    if atom.index < template_len]))
        
        return atoms_list
    
    def evaluate(self, atoms_list, iteration=None):
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
        
        for atoms in atoms_list:
            atoms.calc = self.calculator
            e, f = atoms.get_potential_energy(), atoms.get_forces(apply_constraint=False)

            atoms.calc = SPC(atoms, energy=e, forces=f)  # Store results in a calculator for later retrieval
            if iteration is not None:
                atoms.calc.name = f"Iteration{iteration}"
            
            
        return atoms_list
    
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

    def compute_weights(self, data):
        """Compute Boltzmann weights for structures at a given temperature.
        
        Parameters:
        -----------
        energies : list or array
            Potential energies
        temperature : float
            Temperature for Boltzmann weighting
            
        Returns:
        --------
        list of dict
            Properties including weights and forces
        """
        energies = np.array([atoms.get_potential_energy() for atoms in data])
        # Scale and shift energies for numerical stability
        Es_scaled = -energies / self.temperature_schedule.get_temperature()
        Es_shifted = Es_scaled - np.max(Es_scaled)
        exp_Es = np.exp(Es_shifted)
        weights = exp_Es / np.sum(exp_Es) * len(energies)

        return weights

    def compute_ess(self, data):
        w = self.compute_weights(data)
        weights = w / np.sum(w)
        ess = 1.0 / np.sum(weights**2)

        return ess
    
    def get_proporties(self, atoms_list, temperature):
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
        energies = np.array([atoms.get_potential_energy() for atoms in atoms_list])
        forces = [atoms.get_forces(apply_constraint=False) for atoms in atoms_list]
        weights = self.compute_weights(energies)
        
        properties = []
        for w, e, f in zip(weights, energies, forces):
            properties.append({'weight': w, 'energy': e, 'forces': f})

        return properties

    def save_trajectory(self, atoms_list, path=None, writer=None):
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
        energies = [atoms.get_potential_energy() for atoms in traj]
        
        argsort = np.argsort(energies)
        traj = [traj[i] for i in argsort]

        # Write trajectory
        if writer is not None:
            for atoms in traj:
                writer.write(atoms)
                
        if path is not None:
            write(path, traj)

    def update_adaptive_buffer_size(self, data, min_B=16, max_B=512):
        ess = self.calculate_ess(data)
        target_B = int(ess)
        print(f"ESS: {ess:.2f}, Target Buffer Size: {target_B}")

        # 4. Smooth the update (Moving Average) to prevent jitter
        new_B = 0.8 * self.buffer_size + 0.2 * target_B

        self.buffer_size = int(np.clip(new_B, min_B, max_B))

    def update_buffer(self):
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
        energies = [atoms.get_potential_energy() for atoms in self.all_data]
        valid_idx = [i for i, e in enumerate(energies) if e < 0.0]
        valid_data = [self.all_data[i] for i in valid_idx]

        
        if not valid_data:
            print(f"No valid structures found with negative energy at T={temperature}")
            return 
        
        if len(valid_data) <= self.buffer_size:
            print(f"Not enough data for T={temperature}, using all {len(valid_data)} available structures.")
            self.buffer = valid_data
        
        # Use stochastic prioritized sampling based on weights
        weights = self.compute_weights([atoms.get_potential_energy() for atoms in valid_data])
        
        # Prioritized sampling with randomness
        k = np.random.uniform(size=len(valid_data))**(1/weights)
        sorter = np.argsort(k)[::-1]  # Sort in descending order
        
        # Select top buffer_size structures
        buffer_data = [valid_data[i] for i in sorter[:self.buffer_size]]

        self.buffer = buffer_data

    def _min_energy_filter(self, data):
        energies = [atoms.get_potential_energy() for atoms in data]
        valid_idx = [i for i, e in enumerate(energies) if e > self.min_E]
        filtered_data = [data[i] for i in valid_idx]

        return filtered_data
    
    def sample_stage(self, iteration, data_writer, logdir):
        """Run a sampling stage at a specific temperature.

        Parameters
        ----------
        iteration : int
            Current outer-loop iteration index, used as the TensorBoard step
            for iteration-level metric logging.
        guidance : float
            Force-field guidance strength passed to the sampler.
        all_data : list
            Accumulated structures from all previous iterations.
        all_energies : list
            Accumulated energies from all previous iterations.
        all_forces : list
            Accumulated forces from all previous iterations.
        energy_cut : float
            Current energy cut-off value (passed through unchanged).
        data_writer : ase.io.Trajectory
            Trajectory writer for persisting all sampled structures.
        logdir : pathlib.Path
            Directory used for saving trajectory files.
        """

        temperature = self.temperature_schedule.temperature

        sampling_wall_s = 0.0
        evaluation_wall_s = 0.0

        iteration_data = []

        while self.sample_controller.continue_sampling(data, temperature): # Sample new structures
            t0 = time.perf_counter()
            data = self.sample()
            data = self.check_min_dist(data, min_dist=1.0)
            sampling_wall_s += time.perf_counter() - t0

            # Evaluate energies and forces
            t0 = time.perf_counter()
            data = self.evaluate(data)
            evaluation_wall_s += time.perf_counter() - t0

            data = self._min_energy_filter(data)
            
            iteration_data.extend(data)
            

        name = f"{temperature:.3f}" if temperature is not None else "initial"
        self.save_trajectory(
            iteration_data, 
            path=str(logdir/f"iteration_data_T{name}.traj"), 
            writer=data_writer
        )
        
        # Save filtered structures
        self.save_trajectory(iteration_data, writer=data_writer)

        # Compute ESS and heat capacity before updating the temperature schedule
        if temperature is not None and len(iteration_data) > 0:
            ess = self.compute_ess(iteration_data)

        # Update temperature
        temperature = self.temperature_schedule.next(iteration_data)

        # Update global data
        self.all_data.extend(iteration_data)
        
        self.update_adaptive_buffer_size(all_energies, temperature)
        print(f"Adaptive buffer size for T={temperature:.2f}: {self.buffer_size}")
        
        # Create buffer for training
        self.update_buffer()
        
        # Save buffer
        self.save_trajectory(
            self.buffer, 
            buffer_energies, 
            buffer_forces, 
            path=str(logdir/f"buffer_T{temperature:.2f}.traj")
        )

        # Build stage_info dict for iteration-level logging in run()
        # (training_wall_s and iteration_wall_s are added there, after training completes)
        stage_info = dict(
            temperature=temperature,
            new_energies=new_energies,
            new_forces=new_constrained_forces,
            all_energies=all_energies,
            all_forces=all_forces,
            buffer_energies=buffer_energies,
            buffer_forces=buffer_forces,
            ess=ess,
            sampling_wall_s=sampling_wall_s,
            evaluation_wall_s=evaluation_wall_s,
        )

        return stage_info

    def train_diffusion_stage(self):
        """Train the diffusion model at a specific temperature."""
        print(f"\n=== Training Diffusion Model at T={self.temperature_schedule.get_temperature():.4f} ===")
        
        # Increment max_epochs based on buffer size and steps per loop
        steps_per_epoch = len(self.buffer) // self.batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        epochs_to_add = max(1, self.max_steps_per_loop // steps_per_epoch)

        current_epoch = self.trainer.current_epoch
        self.trainer.fit_loop.max_epochs = current_epoch + epochs_to_add

        buffer_props = self.get_proporties(self.buffer, self.temperature_schedule.get_temperature())

        dataset = create_dataset(self.buffer, properties=buffer_props, **self.dataset_config)

        train(self.diffusion, dataset, self.trainer)
        
        # Save checkpoint
        ckpt_path = str(Path(trainer.log_dir) / f"diffusion_model_T{temperature:.2f}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"Saved diffusion model checkpoint to {ckpt_path}")

    def run(self, max_time_hours=64, max_iterations=100, run_index=0):
        """Run the complete training procedure with temperature annealing."""
        # Initialize trainer
        self.get_trainer(run_index)

        # Ensure logging directory exists
        logdir = Path(trainer.log_dir)
        logdir.mkdir(parents=True, exist_ok=True)

        # Create the GO-Diff TensorBoard logger and wire it to the training controller
        self._godiff_logger = GODiffLogger(trainer.logger.experiment)
        self.training_controller.set_logger(self._godiff_logger)
        self._flops_timing_cb.set_logger(self._godiff_logger)

        # Initialize trajectory writer for all data
        data_writer = Trajectory(str(logdir/"all_data.traj"), mode='w')

        # Main training loop across temperatures
        i = 0
        total_wall_s = 0.0
        while i < max_iterations:
            print(f"\n{'='*50}")
            print(f"STAGE {i}")
            print(f"{'='*50}")
            
            # Set guidance level (0 for first stage, then use configured value)
            if i == 0:
                if "ForceFieldGuidanceConfig" in self.sample_config:
                    FFG_config = self.sample_config.pop("ForceFieldGuidanceConfig")
                else:
                    FFG_config = None
                    
                    
            t_iter_start = time.perf_counter()
            # Sample at current temperature
            stage_info = self.sample_stage(i, data_writer, logdir)

            temperature = self.temperature_schedule.get_temperature()
            print(
                f"Current temperature: {temperature:.4f}, Buffer size: {len(self.buffer)}, \
                Total data size: {len(self.all_data)}"
            )
            
            if len(self.buffer) == 0:
                print(f"Warning: Empty buffer at T={temperature}, skipping training")
                if self._godiff_logger is not None:
                    elapsed = time.perf_counter() - t_iter_start
                    self._godiff_logger.log_iteration(
                        i, **stage_info, iteration_wall_s=elapsed
                    )
                i += 1
                continue
                
            # Train regressor and diffusion models
            t_train_start = time.perf_counter()
            self._flops_timing_cb.reset()
            
            self.train_diffusion_stage()
            training_wall_s = time.perf_counter() - t_train_start
            iteration_wall_s = time.perf_counter() - t_iter_start

            # Log all iteration-level metrics (including timing) to TensorBoard
            if self._godiff_logger is not None:
                self._godiff_logger.log_iteration(
                    i,
                    **stage_info,
                    training_wall_s=training_wall_s,
                    iteration_wall_s=iteration_wall_s,
                )
                

            if i==0 and FFG_config is not None:
                self.sample_config["ForceFieldGuidanceConfig"] = FFG_config
                
            i += 1

        print("\nTraining completed.")
        
        # Save final model
        final_ckpt = str(logdir / "final_model.ckpt")
        trainer.save_checkpoint(final_ckpt)
        print(f"Final model saved to: {final_ckpt}")
        
        return final_ckpt
