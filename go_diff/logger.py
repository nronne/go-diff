import time
import numpy as np


class GODiffLogger:
    """
    TensorBoard logger for GO-Diff run metrics.

    Logs metrics at two granularities:

    * **Iteration level** – one entry per GO-Diff outer loop iteration
      (temperature, energy statistics, buffer/data sizes, ESS, heat capacity).
    * **Training-step level** – one entry per Lightning training step for
      gradient-agreement tracking (written from the training controller
      callback via :py:meth:`log_training_step`).

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
        The underlying TensorBoard writer, typically obtained from a
        Lightning ``TensorBoardLogger`` via ``trainer.logger.experiment``.
    """

    def __init__(self, writer):
        self.writer = writer
        self._cumulative_wall_s = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_force_norm(forces, idx):
        """Return the maximum per-atom force norm for structure at *idx*.

        Returns ``None`` when *forces* is empty or does not contain the index.
        """
        if len(forces) > idx and forces[idx] is not None:
            return float(np.max(np.linalg.norm(forces[idx], axis=-1)))
        return None

    # ------------------------------------------------------------------
    # Iteration-level logging
    # ------------------------------------------------------------------

    def log_iteration(
        self,
        step,
        *,
        temperature,
        new_energies,
        new_forces,
        all_energies,
        all_forces,
        buffer_energies,
        buffer_forces,
        ess=None,
        ess_ratio=None,
        heat_capacity=None,
        iteration_wall_s=None,
        sampling_wall_s=None,
        evaluation_wall_s=None,
        training_wall_s=None,
    ):
        """Log all important metrics for a single GO-Diff iteration.

        Parameters
        ----------
        step : int
            Iteration index (used as the TensorBoard global step).
        temperature : float
            Current annealing temperature.
        new_energies : array-like
            Energies of structures sampled in this iteration.
        new_forces : list of np.ndarray
            Forces on each structure sampled in this iteration.
        all_energies : array-like
            Energies of all structures accumulated so far.
        all_forces : list of np.ndarray
            Forces on all accumulated structures.
        buffer_energies : array-like
            Energies of structures selected into the replay buffer.
        buffer_forces : list of np.ndarray
            Forces on buffer structures.
        ess : float or None
            Effective Sample Size computed over the new samples.
        ess_ratio : float or None
            ESS / n_new_samples ratio.
        heat_capacity : float or None
            Dimensionless heat capacity C(T) = Var(E) / T^2.
        iteration_wall_s : float or None
            Total wall-clock time (seconds) for the full iteration
            (sampling + evaluation + training).
        sampling_wall_s : float or None
            Wall-clock time (seconds) spent in diffusion model inference.
        evaluation_wall_s : float or None
            Wall-clock time (seconds) spent calling the energy/force calculator.
        training_wall_s : float or None
            Wall-clock time (seconds) spent in model training.
        """
        w = self.writer

        # --- temperature ---
        w.add_scalar("iteration/temperature", temperature, step)

        # --- new samples this iteration ---
        new_energies = np.asarray(new_energies) if len(new_energies) > 0 else np.array([])
        n_new = len(new_energies)
        w.add_scalar("iteration/n_new_samples", n_new, step)
        if n_new > 0:
            w.add_scalar("iteration/new_min_energy", float(np.min(new_energies)), step)
            w.add_scalar("iteration/new_mean_energy", float(np.mean(new_energies)), step)
            w.add_scalar("iteration/new_energy_std", float(np.std(new_energies)), step)

            # Max force norm of best new structure (proxy for relaxation quality)
            best_new_idx = int(np.argmin(new_energies))
            best_new_max_force = self._max_force_norm(new_forces, best_new_idx)
            if best_new_max_force is not None:
                w.add_scalar(
                    "iteration/new_best_energy_max_force", best_new_max_force, step
                )

        # --- accumulated / total data ---
        all_energies = np.asarray(all_energies) if len(all_energies) > 0 else np.array([])
        n_total = len(all_energies)
        w.add_scalar("iteration/total_data", n_total, step)
        if n_total > 0:
            best_idx = int(np.argmin(all_energies))
            best_energy = float(all_energies[best_idx])
            w.add_scalar("iteration/best_energy", best_energy, step)
            w.add_scalar("iteration/mean_energy", float(np.mean(all_energies)), step)
            w.add_scalar(
                "iteration/energy_std", float(np.std(all_energies)), step
            )
            w.add_scalar(
                "iteration/energy_variance", float(np.var(all_energies)), step
            )

            # Max force norm of the globally best structure
            best_max_force = self._max_force_norm(all_forces, best_idx)
            if best_max_force is not None:
                w.add_scalar(
                    "iteration/best_energy_max_force", best_max_force, step
                )

        # --- buffer ---
        buffer_energies = (
            np.asarray(buffer_energies) if len(buffer_energies) > 0 else np.array([])
        )
        w.add_scalar("iteration/buffer_size", len(buffer_energies), step)
        if len(buffer_energies) > 0:
            w.add_scalar(
                "iteration/buffer_best_energy", float(np.min(buffer_energies)), step
            )
            w.add_scalar(
                "iteration/buffer_mean_energy", float(np.mean(buffer_energies)), step
            )
            w.add_scalar(
                "iteration/buffer_energy_std", float(np.std(buffer_energies)), step
            )

            # Max force norm of buffer's best structure
            if len(buffer_forces) > 0:
                buf_best_idx = int(np.argmin(buffer_energies))
                buf_best_max_force = self._max_force_norm(buffer_forces, buf_best_idx)
                if buf_best_max_force is not None:
                    w.add_scalar(
                        "iteration/buffer_best_energy_max_force",
                        buf_best_max_force,
                        step,
                    )

        # --- ESS ---
        if ess is not None:
            w.add_scalar("iteration/ess", float(ess), step)
        if ess_ratio is not None:
            w.add_scalar("iteration/ess_ratio", float(ess_ratio), step)

        # --- heat capacity ---
        if heat_capacity is not None:
            w.add_scalar("iteration/heat_capacity", float(heat_capacity), step)

        # --- wall time ---
        if iteration_wall_s is not None:
            self._cumulative_wall_s += iteration_wall_s
            w.add_scalar("timing/iteration_wall_s", float(iteration_wall_s), step)
            w.add_scalar("timing/cumulative_wall_s", self._cumulative_wall_s, step)
        if sampling_wall_s is not None:
            w.add_scalar("timing/sampling_wall_s", float(sampling_wall_s), step)
        if evaluation_wall_s is not None:
            w.add_scalar("timing/evaluation_wall_s", float(evaluation_wall_s), step)
        if training_wall_s is not None:
            w.add_scalar("timing/training_wall_s", float(training_wall_s), step)

    # ------------------------------------------------------------------
    # Training-step level logging
    # ------------------------------------------------------------------

    def log_training_step(
        self,
        global_step,
        *,
        agreement_current=None,
        agreement_ema=None,
        agreement_max=None,
        patience_counter=None,
        step_wall_s=None,
        step_flops=None,
        cumulative_flops=None,
    ):
        """Log gradient-agreement metrics at the training-step level.

        Parameters
        ----------
        global_step : int
            Lightning global step counter.
        agreement_current : float or None
            Cosine similarity between the two gradient halves for this step.
        agreement_ema : float or None
            Exponential moving average of the gradient agreement.
        agreement_max : float or None
            Peak gradient agreement reached so far in this iteration.
        patience_counter : int or None
            Number of steps since the peak agreement was last updated.
        step_wall_s : float or None
            Wall-clock time (seconds) for this training step.
        step_flops : float or None
            Estimated FLOPs for this training step (profiled via
            ``torch.profiler`` for the first few steps per iteration; the
            running mean is used as an estimate for subsequent steps).
        cumulative_flops : float or None
            Running total of training FLOPs across all steps so far.
        """
        w = self.writer
        if agreement_current is not None:
            w.add_scalar(
                "train/gradient_agreement_current", float(agreement_current), global_step
            )
        if agreement_ema is not None:
            w.add_scalar(
                "train/gradient_agreement_ema", float(agreement_ema), global_step
            )
        if agreement_max is not None:
            w.add_scalar(
                "train/gradient_agreement_max", float(agreement_max), global_step
            )
        if patience_counter is not None:
            w.add_scalar(
                "train/patience_counter", int(patience_counter), global_step
            )
        if step_wall_s is not None:
            w.add_scalar("train/step_wall_s", float(step_wall_s), global_step)
        if step_flops is not None:
            w.add_scalar("train/step_flops", float(step_flops), global_step)
        if cumulative_flops is not None:
            w.add_scalar("train/cumulative_flops", float(cumulative_flops), global_step)
