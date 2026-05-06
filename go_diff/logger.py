import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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
        # History for cumulative analysis plots
        self._best_energy_history = []   # list of (n_total, best_energy)
        self._buffer_history = []        # list of (temperature, buffer_energies ndarray)
        self._landscape_history = []     # list of (step, min, mean, std)

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

        # --- custom analysis figures ---
        self.log_analysis(
            step,
            temperature=temperature,
            buffer_energies=buffer_energies,
            all_energies=all_energies,
            new_energies=new_energies,
        )

    # ------------------------------------------------------------------
    # Analysis plots
    # ------------------------------------------------------------------

    def log_analysis(self, step, *, temperature, buffer_energies, all_energies, new_energies):
        """Log custom matplotlib figures under the ``analysis/`` tag group.

        Four figures are produced each iteration:

        * **best_energy_vs_total_data** – running line plot of the global
          best energy against cumulative dataset size.
        * **buffer_energies_vs_temperature** – cumulative scatter plot of
          buffer energies for every temperature seen so far (log x-axis),
          with the per-temperature mean marked as a horizontal tick.
        * **new_sample_energy_distribution** – histogram of the energies
          sampled in the current iteration, with min and mean annotated.
        * **energy_landscape_overview** – running plot of min, mean and a
          ±1-std band of the full accumulated energy set vs. iteration.

        Parameters
        ----------
        step : int
            Current iteration index (TensorBoard global step).
        temperature : float or None
            Temperature at the current iteration.
        buffer_energies : array-like
            Energies of structures in the replay buffer.
        all_energies : array-like
            Energies of all accumulated structures.
        new_energies : array-like
            Energies of structures sampled this iteration.
        """
        w = self.writer

        all_energies = np.asarray(all_energies) if len(all_energies) > 0 else np.array([])
        buffer_energies = (
            np.asarray(buffer_energies) if len(buffer_energies) > 0 else np.array([])
        )
        new_energies = (
            np.asarray(new_energies) if len(new_energies) > 0 else np.array([])
        )

        # --- update running histories ---
        if len(all_energies) > 0:
            self._best_energy_history.append((len(all_energies), float(np.min(all_energies))))

        if temperature is not None and len(buffer_energies) > 0:
            self._buffer_history.append((float(temperature), buffer_energies.copy()))

        # -----------------------------------------------------------------
        # Plot 1: best energy as a function of total data
        # -----------------------------------------------------------------
        if len(self._best_energy_history) >= 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ns, bests = zip(*self._best_energy_history)
            ax.plot(ns, bests, "o-", color="steelblue", markersize=4, linewidth=1.5)
            ax.set_xlabel("Total data (structures)")
            ax.set_ylabel("Best energy [eV]")
            ax.set_title("Best energy vs. total data")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            w.add_figure("analysis/best_energy_vs_total_data", fig, global_step=step)
            plt.close(fig)

        # -----------------------------------------------------------------
        # Plot 2: buffer energies violin vs temperature (seaborn-style)
        # -----------------------------------------------------------------
        if len(self._buffer_history) >= 1:
            # Group energies per temperature across all iterations
            grouped = defaultdict(list)
            for t, e in self._buffer_history:
                grouped[t].extend(e.tolist())

            temps_sorted = sorted(grouped.keys(), reverse=True)  # high → low

            # Normalise to global min so y-axis starts near 0
            global_min = min(min(v) for v in grouped.values())

            # Colour map: RdYlBu_r → index 0 = red (high T), index 1 = blue (low T)
            n_temps = max(len(temps_sorted) - 1, 1)
            cmap = plt.cm.RdYlBu_r
            colors = {
                t: cmap(i / n_temps)
                for i, t in enumerate(temps_sorted)
            }

            fig, ax = plt.subplots(figsize=(8, 5))

            for t in temps_sorted:
                energies_t = np.asarray(grouped[t]) - global_min
                if len(energies_t) < 2:
                    # Not enough data for KDE – fall back to a single point
                    ax.scatter([t], [float(np.mean(energies_t))],
                               color=colors[t], s=40, zorder=5)
                    continue
                x_col = np.full(len(energies_t), t)
                _df = pd.DataFrame({"temperature": x_col, "energy": energies_t})
                sns.violinplot(
                    data=_df,
                    x="temperature",
                    y="energy",
                    ax=ax,
                    color=colors[t],
                    inner="point",
                    cut=0,
                    bw_adjust=0.5,
                    linewidth=0.5,
                    density_norm="width",
                    common_norm=True,
                    native_scale=True,
                    width=0.2,
                )

            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Energy [eV]")
            ax.set_ylim(-0.05, 5)
            ax.set_title("Buffer energies vs. temperature")
            ax.grid(True, alpha=0.3, which="both")
            fig.tight_layout()
            w.add_figure("analysis/buffer_energies_vs_temperature", fig, global_step=step)
            plt.close(fig)

        # -----------------------------------------------------------------
        # Plot 3: new-sample energy distribution (histogram)
        # -----------------------------------------------------------------
        if len(new_energies) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(new_energies, bins=max(1, min(20, len(new_energies))), color="steelblue",
                    edgecolor="white", alpha=0.75)
            ax.axvline(float(np.min(new_energies)), color="red", linestyle="--",
                       linewidth=1.5, label=f"min = {np.min(new_energies):.3f}")
            ax.axvline(float(np.mean(new_energies)), color="orange", linestyle="--",
                       linewidth=1.5, label=f"mean = {np.mean(new_energies):.3f}")
            ax.set_xlabel("Energy [eV]")
            ax.set_ylabel("Count")
            t_str = f"{temperature:.4f}" if temperature is not None else "initial"
            ax.set_title(f"New-sample energy distribution  (T = {t_str})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            w.add_figure("analysis/new_sample_energy_distribution", fig, global_step=step)
            plt.close(fig)

        # -----------------------------------------------------------------
        # Plot 4: energy landscape overview (min / mean ± std vs iteration)
        # -----------------------------------------------------------------
        if len(self._best_energy_history) >= 1 and len(all_energies) > 0:
            self._landscape_history.append(
                (step, float(np.min(all_energies)),
                 float(np.mean(all_energies)), float(np.std(all_energies)))
            )
            steps_l, mins_l, means_l, stds_l = zip(*self._landscape_history)
            steps_l = np.array(steps_l)
            mins_l = np.array(mins_l)
            means_l = np.array(means_l)
            stds_l = np.array(stds_l)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(
                steps_l, means_l - stds_l, means_l + stds_l,
                alpha=0.25, color="steelblue", label="mean ± std"
            )
            ax.plot(steps_l, means_l, "-", color="steelblue", linewidth=1.5)
            ax.plot(steps_l, mins_l, "o-", color="crimson", markersize=4,
                    linewidth=1.5, label="best")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Energy [eV]")
            ax.set_title("Energy landscape overview")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            w.add_figure("analysis/energy_landscape_overview", fig, global_step=step)
            plt.close(fig)

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
