"""Core GO-Diff optimisation loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, runtime_checkable
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike

from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.io import write, Trajectory

# AGeDi
from agedi import create_dataset, create_trainer, train, sample

# GO-Diff
from .filter import (
    Filter,
    MinEnergyFilter,
    MaxEnergyFilter,
)
from go_diff.controllers import (
    TemperatureSchedule,
    SampleController,
    BufferController,
    MomentumConsensusStop,
    FlopsAndTimingCallback,
)
from go_diff.utils import boltzmann_weights, effective_sample_size
from .logger import GODiffLogger


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

@runtime_checkable
class Calculator(Protocol):
    """Minimal interface expected of an ASE-compatible energy/force calculator."""

    def get_potential_energy(self) -> float: ...
    def get_forces(self) -> np.ndarray: ...


@runtime_checkable
class DiffusionModel(Protocol):
    """Minimal interface expected of an AGeDi diffusion model."""

    def to(self, device: str) -> "DiffusionModel": ...


@runtime_checkable
class TrainingController(Protocol):
    """Minimal interface expected of a Lightning training-stop callback."""

    def set_logger(self, logger: object) -> None: ...


class GODiff:
    """Gradient-Optimised Diffusion model for atomic structure search.

    Implements an iterative training procedure with adaptive temperature
    annealing.  At each iteration:

    1. **Sample** candidate structures from the current diffusion model.
    2. **Evaluate** their energies and forces with the provided calculator.
    3. **Build a replay buffer** using Boltzmann-weighted prioritised sampling.
    4. **Retrain** the diffusion model on the buffer until the training
       controller signals convergence.
    5. **Update the temperature** schedule for the next iteration.

    Parameters
    ----------
    calculator : Calculator
        ASE-compatible calculator used to evaluate potential energies and
        forces (must implement :class:`Calculator`).
    diffusion : DiffusionModel
        Diffusion model created with :func:`agedi.create_diffusion` (must
        implement :class:`DiffusionModel`).
    temperature_schedule : TemperatureSchedule
        Controls the annealing temperature.  Defaults to
        ``TemperatureSchedule()``.
    sample_controller : SampleController
        Decides when enough structures have been collected per iteration.
        Defaults to ``SampleController()``.
    buffer_controller : BufferController
        Controls the replay buffer size.  Defaults to ``BufferController()``.
    training_controller : TrainingController
        Decides when to stop training each iteration (must implement
        :class:`TrainingController`).  Defaults to
        ``MomentumConsensusStop()``.
    sample_config : dict
        Keyword arguments forwarded to :func:`agedi.sample`.  Common keys:
        ``template``, ``atomic_numbers``, ``ForceFieldGuidanceConfig``.
    dataset_config : dict
        Keyword arguments forwarded to :func:`agedi.create_dataset`.  Common
        keys: ``mask``, ``confinement``, ``regressor_data``.
    batch_size : int
        Mini-batch size used to estimate the number of training epochs per
        iteration.  Default: 32.
    max_steps_per_loop : int
        Maximum number of training steps per GO-Diff iteration.  Default: 500.
    valid_structure_filters : list of Filter or None, optional
        Sequence of callables ``(atoms: Atoms) -> bool`` applied right after
        energy/force evaluation to discard physically unreasonable structures.
        Discarded structures are never added to ``all_data`` or the buffer.
        Defaults to ``None`` (no filtering).  Pass
        ``[MinEnergyFilter(-500.0)]`` to replicate the former
        ``GODiff(min_E=-500.0)`` behaviour.  Use :class:`~go_diff.filter.MinDistFilter`
        to enforce a minimum interatomic distance, e.g.
        ``[MinDistFilter(1.0)]``.
    buffer_filters : list of Filter, optional
        Sequence of callables ``(atoms: Atoms) -> bool`` applied when
        rebuilding the buffer from ``all_data``.  A structure from
        ``all_data`` is included in the buffer candidate pool only when
        **all** filters return ``True``.  Defaults to
        ``[MaxEnergyFilter(0.0)]``, which restricts the buffer to structures
        with negative energy — reproducing the historical silent ``e < 0.0``
        filter.  Use an empty list (``[]``) to disable buffer-level filtering.
    device : str
        Device passed to the trainer / sampler.  Default: ``"cuda"``.
    """

    def __init__(
        self,
        calculator: Calculator,
        diffusion: DiffusionModel,
        temperature_schedule: TemperatureSchedule | None = None,
        sample_controller: SampleController | None = None,
        buffer_controller: BufferController | None = None,
        training_controller: TrainingController | None = None,
        sample_config: dict | None = None,
        dataset_config: dict | None = None,
        trainer_config: dict | None = None,
        batch_size: int = 32,
        sample_batch_size: int = 16,
        max_steps_per_loop: int = 500,
        valid_structure_filters: list[Filter] | None = None,
        buffer_filters: list[Filter] | None = None,
        device: str = "cuda",
    ) -> None:
        self.calculator = calculator
        self.diffusion = diffusion
        self.temperature_schedule = temperature_schedule or TemperatureSchedule()
        self.sample_controller = sample_controller or SampleController()
        self.training_controller = training_controller or MomentumConsensusStop()
        self.buffer_controller = buffer_controller or BufferController()

        self.sample_config: dict = sample_config or {}
        self.dataset_config: dict = dataset_config or {}
        self.trainer_config: dict = trainer_config or {}

        self.batch_size: int = batch_size
        self.sample_batch_size: int = sample_batch_size
        self.max_steps_per_loop: int = max_steps_per_loop
        self.valid_structure_filters: list[Filter] | None = valid_structure_filters
        self.buffer_filters: list[Filter] = (
            buffer_filters if buffer_filters is not None else [MaxEnergyFilter(0.0)]
        )
        self.device: str = device

        self._godiff_logger: GODiffLogger | None = None
        self._flops_timing_cb: FlopsAndTimingCallback | None = None
        self.trainer: object | None = None

        self.buffer: list[Atoms] = []
        self.all_data: list[Atoms] = []

    # ------------------------------------------------------------------
    # Trainer setup
    # ------------------------------------------------------------------

    def _get_trainer(self, max_time_hours: float = 64) -> None:
        """Create and store a PyTorch Lightning trainer.

        Parameters
        ----------
        max_time_hours : float
            Maximum wall-clock training time in hours passed to
            :func:`agedi.create_trainer`.  Default: 64.
        """
        self._flops_timing_cb = FlopsAndTimingCallback()

        callbacks = [
            self.training_controller,
            self._flops_timing_cb,
        ]

        self.trainer = create_trainer(
            max_time={"hours": max_time_hours},
            extra_callbacks=callbacks,
            **self.trainer_config,
        )

    # ------------------------------------------------------------------
    # Sampling & evaluation
    # ------------------------------------------------------------------

    def sample(self, exclude_keys: set[str] | None = None) -> list[Atoms]:
        """Sample structures from the current diffusion model.

        Uses :attr:`sample_config` as keyword arguments forwarded to
        :func:`agedi.sample`.  When a ``"template"`` is present the template
        atoms are frozen via :class:`ase.constraints.FixAtoms`.

        Parameters
        ----------
        exclude_keys : set of str or None
            Keys to temporarily omit from :attr:`sample_config` for this call.
            Useful for suppressing optional guidance on the first iteration.

        Returns
        -------
        list of ase.Atoms
            Sampled structures.
        """
        config = dict(self.sample_config)
        if exclude_keys:
            for key in exclude_keys:
                config.pop(key, None)
        n_samples = config.pop("n_samples", self.sample_batch_size)

        atoms_list = sample(self.diffusion, n_samples=n_samples, **config)

        template = config.get("template")
        if template is not None:
            template_len = len(template)
            for atoms in atoms_list:
                atoms.set_constraint(
                    FixAtoms(
                        mask=[atom.index < template_len for atom in atoms]
                    )
                )

        return atoms_list

    def evaluate(
        self,
        atoms_list: list[Atoms],
        iteration: int | None = None,
    ) -> list[Atoms]:
        """Evaluate energies and forces for a list of structures in-place.

        Each structure's calculator is replaced with a
        :class:`~ase.calculators.singlepoint.SinglePointCalculator` that
        stores the computed energy and forces so they can be retrieved later
        without re-running the (potentially expensive) calculator.

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            Structures to evaluate.
        iteration : int or None
            If provided, stored as the calculator name for bookkeeping.

        Returns
        -------
        list of ase.Atoms
            The same structures with updated calculators.
        """
        for atoms in atoms_list:
            atoms.calc = self.calculator
            e = atoms.get_potential_energy()
            f = atoms.get_forces(apply_constraint=False)
            atoms.calc = SPC(atoms, energy=e, forces=f)
            if iteration is not None:
                atoms.calc.name = f"Iteration{iteration}"

        return atoms_list

    # ------------------------------------------------------------------
    # Boltzmann weighting helpers
    # ------------------------------------------------------------------

    def compute_weights(self, data: list[Atoms]) -> np.ndarray:
        """Compute Boltzmann importance weights for a set of structures.

        Weights are normalised so that their sum equals ``len(data)``.
        Uses the current temperature from :attr:`temperature_schedule`.

        Parameters
        ----------
        data : list of ase.Atoms
            Structures with energies already attached.

        Returns
        -------
        np.ndarray of float, shape ``(len(data),)``
            Boltzmann importance weights, summing to ``len(data)``.
        """
        temperature = self.temperature_schedule.get_temperature()
        energies = np.array([atoms.get_potential_energy() for atoms in data])
        return boltzmann_weights(energies, temperature)

    def compute_ess(self, data: list[Atoms]) -> float:
        """Compute the Effective Sample Size (ESS) for a set of structures.

        Parameters
        ----------
        data : list of ase.Atoms
            Structures with energies attached.

        Returns
        -------
        float
            The ESS (between 1 and len(data)).
        """
        temperature = self.temperature_schedule.get_temperature()
        energies = np.array([atoms.get_potential_energy() for atoms in data])
        return effective_sample_size(energies, temperature)

    def get_properties(
        self,
        atoms_list: list[Atoms],
    ) -> list[dict]:
        """Compute Boltzmann weights, energies, and forces for each structure.

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            Structures with energies and forces already computed.

        Returns
        -------
        list of dict
            Each dict contains ``'weight'`` (float), ``'energy'`` (float),
            and ``'forces'`` (np.ndarray).
        """
        energies = np.array(
            [atoms.get_potential_energy() for atoms in atoms_list]
        )
        forces = [
            atoms.get_forces(apply_constraint=False) for atoms in atoms_list
        ]
        weights = self.compute_weights(atoms_list)

        return [
            {"weight": float(w), "energy": float(e), "forces": f}
            for w, e, f in zip(weights, energies, forces)
        ]

    # ------------------------------------------------------------------
    # Trajectory I/O
    # ------------------------------------------------------------------

    def save_trajectory(
        self,
        atoms_list: list[Atoms],
        path: str | Path | None = None,
        writer: object | None = None,
    ) -> None:
        """Save structures (sorted by energy) to a trajectory file.

        At least one of *path* or *writer* must be provided.

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            Structures to save.
        path : str or Path or None
            If provided, write the trajectory to this file path using
            :func:`ase.io.write`.
        writer : ase.io.Trajectory or None
            If provided, append each structure to this open trajectory writer.
        """
        if path is None and writer is None:
            raise ValueError("Either path or writer must be provided.")


        energies = [atoms.get_potential_energy() for atoms in atoms_list]
        traj = [deepcopy(atoms) for atoms in atoms_list]
        argsort = np.argsort(energies)
        traj = [traj[i] for i in argsort]

        if writer is not None:
            for atoms in traj:
                writer.write(atoms)

        if path is not None:
            write(str(path), traj)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _apply_valid_structure_filters(self, data: list[Atoms]) -> list[Atoms]:
        """Apply :attr:`valid_structure_filters` to *data* and return survivors.

        When :attr:`valid_structure_filters` is ``None`` (the default), all
        structures are returned unchanged.  Otherwise a structure is kept only
        when **every** filter returns ``True``.

        Parameters
        ----------
        data : list of ase.Atoms

        Returns
        -------
        list of ase.Atoms
        """
        if not self.valid_structure_filters:
            return data
        return [
            atoms for atoms in data
            if all(f(atoms) for f in self.valid_structure_filters)
        ]

    def update_buffer(self) -> None:
        """Rebuild :attr:`buffer` via Boltzmann-weighted prioritised sampling.

        :attr:`buffer_filters` are applied to :attr:`all_data` to obtain the
        candidate pool; structures rejected by any filter are excluded.  When
        fewer candidates than the current buffer size are available, all
        candidates are used.  Otherwise a stochastic prioritised subset is
        selected via reservoir-style key-based sorting.
        """
        # Apply buffer-level filters to the full accumulated dataset
        if self.buffer_filters:
            valid_data = [
                atoms for atoms in self.all_data
                if all(f(atoms) for f in self.buffer_filters)
            ]
        else:
            valid_data = self.all_data

        if not valid_data:
            print(
                "No valid structures at "
                f"T={self.temperature_schedule.temperature}"
            )
            self.buffer = []
            return

        if len(valid_data) <= self.buffer_controller.get_buffer_size():
            print(
                f"Not enough data, using all {len(valid_data)} valid structures."
            )
            self.buffer = valid_data
            return

        weights = self.compute_weights(valid_data)
        # Stochastic prioritised sampling: reservoir key trick
        keys = np.random.uniform(size=len(valid_data)) ** (1.0 / weights)
        sorter = np.argsort(keys)[::-1]
        self.buffer = [valid_data[i] for i in sorter[: self.buffer_controller.get_buffer_size()]]

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def sample_stage(
        self,
        iteration: int,
        data_writer: object,
        logdir: Path,
        exclude_sample_keys: set[str] | None = None,
    ) -> dict:
        """Run one sampling-and-evaluation stage.

        Samples structures until :attr:`sample_controller` is satisfied,
        evaluates them, filters unphysical structures, updates the temperature,
        and rebuilds the replay buffer.

        Parameters
        ----------
        iteration : int
            Current outer-loop iteration index (used for logging and file
            naming).
        data_writer : ase.io.Trajectory
            Open trajectory writer for all accumulated structures.
        logdir : pathlib.Path
            Directory for per-iteration trajectory files.
        exclude_sample_keys : set of str or None
            Keys to omit from :attr:`sample_config` for this stage (passed
            through to :meth:`sample`).

        Returns
        -------
        dict
            Keys: ``temperature``, ``new_energies``, ``new_forces``,
            ``all_energies``, ``all_forces``, ``buffer_energies``,
            ``buffer_forces``, ``ess``, ``sampling_wall_s``,
            ``evaluation_wall_s``.
        """
        current_temperature = self.temperature_schedule.temperature

        sampling_wall_s = 0.0
        evaluation_wall_s = 0.0
        iteration_data: list[Atoms] = []

        while self.sample_controller.continue_sampling(
            [a.get_potential_energy() for a in iteration_data],
            current_temperature,
        ):
            t0 = time.perf_counter()
            new_samples = self.sample(exclude_keys=exclude_sample_keys)
            sampling_wall_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            new_samples = self.evaluate(new_samples, iteration=iteration)
            evaluation_wall_s += time.perf_counter() - t0

            new_samples = self._apply_valid_structure_filters(new_samples)
            iteration_data.extend(new_samples)

        # Save this iteration's structures
        name = f"{current_temperature:.3f}" if current_temperature is not None else "initial"
        self.save_trajectory(
            iteration_data,
            path=str(logdir / f"iteration_data_T{name}.traj"),
            writer=data_writer,
        )

        # Compute ESS *before* updating the temperature
        ess: float | None = None
        if current_temperature is not None and len(iteration_data) > 0:
            ess = self.compute_ess(iteration_data)

        # Update temperature based on this iteration's energies
        if iteration_data:
            self.temperature_schedule.next(
                [a.get_potential_energy() for a in iteration_data]
            )

        # Accumulate global dataset
        self.all_data.extend(iteration_data)

        # Adapt buffer size and rebuild buffer
        if self.all_data:
            # Only call update_adaptive_buffer_size after temperature is set
            if self.temperature_schedule.temperature is not None:
                self.buffer_controller.update_buffer_size(
                    [a.get_potential_energy() for a in self.all_data],
                    self.temperature_schedule.temperature
                )
        self.update_buffer()

        # Save buffer
        if self.buffer:
            t_str = f"{self.temperature_schedule.temperature:.2f}"
            self.save_trajectory(
                self.buffer,
                path=str(logdir / f"buffer_T{t_str}.traj"),
            )

        # Extract arrays for the stage_info dict
        new_energies = (
            np.array([a.get_potential_energy() for a in iteration_data])
            if iteration_data else np.array([])
        )
        new_forces = (
            [a.get_forces(apply_constraint=False) for a in iteration_data]
            if iteration_data else []
        )
        all_energies = (
            np.array([a.get_potential_energy() for a in self.all_data])
            if self.all_data else np.array([])
        )
        all_forces = (
            [a.get_forces(apply_constraint=False) for a in self.all_data]
            if self.all_data else []
        )
        buffer_energies = (
            np.array([a.get_potential_energy() for a in self.buffer])
            if self.buffer else np.array([])
        )
        buffer_forces = (
            [a.get_forces(apply_constraint=False) for a in self.buffer]
            if self.buffer else []
        )

        stage_info = dict(
            temperature=self.temperature_schedule.temperature,
            new_energies=new_energies,
            new_forces=new_forces,
            all_energies=all_energies,
            all_forces=all_forces,
            buffer_energies=buffer_energies,
            buffer_forces=buffer_forces,
            ess=ess,
            sampling_wall_s=sampling_wall_s,
            evaluation_wall_s=evaluation_wall_s,
        )
        return stage_info

    def train_diffusion_stage(self) -> None:
        """Train the diffusion model on the current replay buffer.

        The number of additional epochs is computed from :attr:`batch_size` and
        :attr:`max_steps_per_loop` so that training runs for at most
        *max_steps_per_loop* gradient steps.  The training controller callback
        may stop training earlier.
        """
        temperature = self.temperature_schedule.get_temperature()
        print(f"\n=== Training Diffusion Model at T={temperature:.4f} ===")

        steps_per_epoch = max(1, len(self.buffer) // self.batch_size)
        epochs_to_add = max(1, self.max_steps_per_loop // steps_per_epoch)

        current_epoch = self.trainer.current_epoch
        self.trainer.fit_loop.max_epochs = current_epoch + epochs_to_add

        buffer_props = self.get_properties(self.buffer)

        dataset_config = dict(self.dataset_config)
        # Inject live all_data reference for regressor training if requested.
        if "regressor_data" in dataset_config:
            dataset_config["regressor_data"] = self.all_data

        dataset = create_dataset(
            self.buffer, properties=buffer_props, **dataset_config
        )
        train(self.diffusion, dataset, self.trainer)

        # Save checkpoint
        ckpt_path = str(
            Path(self.trainer.log_dir) / f"diffusion_model_T{temperature:.2f}.ckpt"
        )
        self.trainer.save_checkpoint(ckpt_path)
        print(f"Saved diffusion model checkpoint to {ckpt_path}")

        #ensure diffusion model stays on the correct device after training
        self.diffusion.to(self.device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        max_time_hours: float = 64,
        max_iterations: int = 100,
    ) -> str:
        """Run the complete GO-Diff training procedure.

        Iterates over sampling and training stages, annealing the temperature
        after each iteration, until *max_iterations* is reached or the
        wall-clock limit is hit.

        Parameters
        ----------
        max_time_hours : float
            Maximum total wall-clock time (hours) for the trainer.  Default: 64.
        max_iterations : int
            Maximum number of outer-loop iterations.  Default: 100.

        Returns
        -------
        str
            Path to the final saved model checkpoint.
        """
        self._get_trainer(max_time_hours)

        logdir = Path(self.trainer.log_dir)
        logdir.mkdir(parents=True, exist_ok=True)

        self._godiff_logger = GODiffLogger(self.trainer.logger.experiment)
        self.training_controller.set_logger(self._godiff_logger)
        self._flops_timing_cb.set_logger(self._godiff_logger)

        data_writer = Trajectory(str(logdir / "all_data.traj"), mode="w")

        # Disable force-field guidance on the first iteration (cold start)
        # by passing a temporary config without ForceFieldGuidanceConfig.
        ffg_config = self.sample_config.get("ForceFieldGuidanceConfig")

        i = 0
        while i < max_iterations:
            print(f"\n{'='*50}\nSTAGE {i}\n{'='*50}")

            t_iter_start = time.perf_counter()
            # Suppress force-field guidance on the first (cold-start) iteration.
            exclude = {"ForceFieldGuidanceConfig"} if i == 0 and ffg_config is not None else None
            stage_info = self.sample_stage(i, data_writer, logdir, exclude_sample_keys=exclude)

            temperature = self.temperature_schedule.temperature
            print(
                f"Current temperature: {temperature}, "
                f"Buffer size: {len(self.buffer)}, "
                f"Total data size: {len(self.all_data)}"
            )

            if len(self.buffer) == 0:
                print(f"Warning: Empty buffer at T={temperature}, skipping training.")
                if self._godiff_logger is not None:
                    self._godiff_logger.log_iteration(
                        i,
                        **stage_info,
                        iteration_wall_s=time.perf_counter() - t_iter_start,
                    )
                i += 1
                continue

            t_train_start = time.perf_counter()
            self._flops_timing_cb.reset()
            self.train_diffusion_stage()
            training_wall_s = time.perf_counter() - t_train_start
            iteration_wall_s = time.perf_counter() - t_iter_start

            if self._godiff_logger is not None:
                self._godiff_logger.log_iteration(
                    i,
                    **stage_info,
                    training_wall_s=training_wall_s,
                    iteration_wall_s=iteration_wall_s,
                )

            i += 1

        print("\nTraining completed.")

        final_ckpt = str(logdir / "final_model.ckpt")
        self.trainer.save_checkpoint(final_ckpt)
        print(f"Final model saved to: {final_ckpt}")

        return final_ckpt
