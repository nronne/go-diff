"""Lightning callbacks that control training duration in GO-Diff."""

from __future__ import annotations

import time
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Batch


class MomentumConsensusStop(Callback):
    """Stop training when gradient momentum and current gradient diverge.

    After an initial warm-up phase (*min_steps*) the callback computes the
    cosine similarity between the current gradient and the first-moment
    estimate stored by the Adam/AdamW optimiser.  Training is halted when
    agreement has been below ``drop_factor × peak_agreement`` for *patience*
    consecutive steps.

    Parameters
    ----------
    min_steps : int
        Minimum number of training steps before the stopping criterion is
        evaluated.  Default: 50.
    patience : int
        Number of consecutive steps below the drop threshold before training
        is stopped.  Default: 100.
    drop_factor : float
        Fraction of the peak agreement that the current agreement must fall
        below (for *patience* steps) to trigger early stopping.  Default: 0.5.
    """

    def __init__(
        self,
        min_steps: int = 50,
        patience: int = 100,
        drop_factor: float = 0.5,
    ) -> None:
        super().__init__()
        self.min_steps = min_steps
        self.patience = patience
        self.drop_factor = drop_factor

        self.max_agreement: float = -1.0
        self.patience_counter: int = 0
        self.current_step: int = 0

        # Optional GODiffLogger for TensorBoard logging
        self._godiff_logger = None

    def set_logger(self, logger: Any) -> None:
        """Attach a GODiffLogger so agreement metrics are written to TensorBoard.

        Parameters
        ----------
        logger : GODiffLogger
            Logger instance to receive per-step metrics.
        """
        self._godiff_logger = logger

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        """Evaluate momentum–gradient agreement after each backward pass.

        Gradients are available in ``pl_module.parameters()`` at this point.
        Computes the cosine similarity between the current gradient vector and
        the Adam first-moment (momentum) vector and triggers early stopping
        when the agreement has been low for *patience* steps.
        """
        self.current_step += 1
        if self.current_step < self.min_steps:
            return

        # 1. Pull the optimizer (to access momentum states)
        # In manual optimization, you might have multiple; we'll take the first.
        opt = trainer.optimizers[0]
        
        grads = []
        momentums = []
        
        for p in pl_module.parameters():
            if p.grad is not None:
                # 'exp_avg' is the first moment (moving average) in Adam/AdamW
                state = opt.state[p]
                if 'exp_avg' in state:
                    grads.append(p.grad.detach().view(-1))
                    momentums.append(state['exp_avg'].detach().view(-1))

        if not grads or not momentums:
            return

        # 2. Vectorized Consensus Calculation
        g = torch.cat(grads)
        m = torch.cat(momentums)
        
        # Cosine similarity measures if the new gradient aligns with the trend
        agreement = torch.nn.functional.cosine_similarity(g, m, dim=0).item()

        # 3. Peak-to-Drop Logic
        if agreement > self.max_agreement:
            self.max_agreement = agreement
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # 4. Stopping logic
        if (self.patience_counter >= self.patience and 
            agreement < (self.drop_factor * self.max_agreement)):
            
            print(f"\n[Adaptive Stop] Consensus reached. "
                  f"Peak Agreement: {self.max_agreement:.3f} | Current: {agreement:.3f}")
            trainer.should_stop = True

        if self._godiff_logger is not None:
            self._godiff_logger.log_training_step(
                trainer.global_step,
                agreement_current=agreement,
                agreement_max=self.max_agreement,
                patience_counter=self.patience_counter,
            )
            
        
    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset internal state at the start of each GO-Diff training stage."""
        self.max_agreement = -1.0
        self.patience_counter = 0
        self.current_step = 0


class AdaptiveRefinementStop(Callback):
    """Stop training when the gradient-agreement EMA has significantly dropped.

    Computes gradient agreement by splitting each mini-batch into two halves,
    running independent backward passes, and measuring the cosine similarity of
    the resulting gradient vectors.  Training stops when the exponential moving
    average (EMA) of the agreement has peaked and then fallen to less than half
    its peak value for *patience* consecutive check-points.

    Parameters
    ----------
    min_steps : int
        Global training step at which checking begins.  Default: 100.
    patience : int
        Number of consecutive low-agreement check-points before stopping.
        Default: 50.
    smooth_factor : float
        EMA smoothing coefficient (between 0 and 1).  Higher values give a
        smoother but more lagging estimate.  Default: 0.75.
    check_interval : int
        Interval in training steps between agreement evaluations.  Default: 1.
    """

    def __init__(
        self,
        min_steps: int = 100,
        patience: int = 50,
        smooth_factor: float = 0.75,
        check_interval: int = 1,
    ) -> None:
        super().__init__()
        self._min_steps = min_steps
        self.min_steps = min_steps
        self.patience = patience
        self.smooth_factor = smooth_factor
        self.check_interval = check_interval

        self.ema_agreement: float = 0.0
        self.max_agreement: float = -1.0
        self.patience_counter: int = 0

        # Optional GODiffLogger for TensorBoard logging
        self._godiff_logger = None

    def set_logger(self, logger: Any) -> None:
        """Attach a GODiffLogger so agreement metrics are written to TensorBoard.

        Parameters
        ----------
        logger : GODiffLogger
            Logger instance to receive per-step metrics.
        """
        self._godiff_logger = logger

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Evaluate gradient agreement and optionally stop training."""
        if trainer.global_step < self.min_steps or trainer.global_step % self.check_interval != 0:
            return

        if len(batch) < 4:
            return

        # 1. Calculate current agreement (using the split logic from before)
        current_agreement = self._calculate_split_agreement(trainer, pl_module, batch, batch_idx)
        
        # 2. Apply Exponential Moving Average (EMA) to smooth the noise
        self.ema_agreement = (self.smooth_factor * self.ema_agreement) + \
                             ((1 - self.smooth_factor) * current_agreement)

        # 3. Track the peak agreement reached in this iteration
        if self.ema_agreement > self.max_agreement:
            self.max_agreement = self.ema_agreement
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # 4. Log gradient-agreement metrics to TensorBoard
        if self._godiff_logger is not None:
            self._godiff_logger.log_training_step(
                trainer.global_step,
                agreement_current=current_agreement,
                agreement_ema=self.ema_agreement,
                agreement_max=self.max_agreement,
                patience_counter=self.patience_counter,
            )

        # 5. Logic: If agreement has significantly dropped from its peak, stop.
        # This means the model has finished learning the "consensus" and is now over-fitting.
        if self.patience_counter >= self.patience and self.ema_agreement < (0.5 * self.max_agreement):
            print(f"\n[Adaptive Stop] Agreement peaked at {self.max_agreement:.4f} "
                  f"and dropped to {self.ema_agreement:.4f}. Stopping.")
            trainer.should_stop = True
            
    def _calculate_split_agreement(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> float:
        """Compute gradient cosine similarity by splitting the batch in two.

        Parameters
        ----------
        trainer : lightning.Trainer
        pl_module : lightning.LightningModule
        batch : list
            Current mini-batch (list of graph data objects).
        batch_idx : int

        Returns
        -------
        float
            Cosine similarity between the two half-batch gradients.
        """
        # 1. Split the batch into two independent halves
        half = len(batch) // 2

        opt = trainer.optimizers[0]
        # 2. Calculate Gradient A (First Half)
        pl_module.zero_grad()
        loss_a = pl_module.loss(Batch.from_data_list(batch[:half]), batch_idx)
        trainer.strategy.backward(loss_a["loss"], optimizer=opt)
        grad_a = self._get_flat_grad(pl_module)

        # 3. Calculate Gradient B (Second Half)
        pl_module.zero_grad()
        loss_b = pl_module.loss(Batch.from_data_list(batch[half:]), batch_idx)
        trainer.strategy.backward(loss_b["loss"], optimizer=opt)
        grad_b = self._get_flat_grad(pl_module)

        # 4. Compute Agreement (Cosine Similarity)
        grad_a /= torch.norm(grad_a)
        grad_b /= torch.norm(grad_b)
        agreement = cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0)).item()
        pl_module.zero_grad()
        
        return agreement



    def _get_flat_grad(self, pl_module: Any) -> torch.Tensor:
        """Flatten all model gradients into a single 1-D tensor."""
        grads = []
        for param in pl_module.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([])


    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset internal state at the start of each GO-Diff training stage.

        *min_steps* is offset by the current global step so the warm-up
        period is restarted relative to the current Lightning global step.
        """
        self.ema_agreement = 0.0
        self.max_agreement = -1.0
        self.patience_counter = 0
        self.min_steps = trainer.global_step + self._min_steps

    def reset(self, trainer: Any) -> None:
        """Reset internal state; *min_steps* is offset by the current global step.

        .. deprecated::
            Call :meth:`on_train_start` or rely on the Lightning callback
            mechanism instead.  This method is kept for backwards compatibility
            but will be removed in a future release.
        """
        self.on_train_start(trainer, None)

class FlopsAndTimingCallback(Callback):
    """Lightning callback that tracks per-training-step wall-time and estimates
    FLOPs via ``torch.profiler`` for the first ``profile_steps`` steps of each
    GO-Diff iteration.

    For each iteration, ``torch.profiler`` actively captures FLOPs for the first
    ``profile_steps`` training steps.  After that window the running-mean
    FLOPs/step estimate is reused for all remaining steps, keeping the profiling
    overhead negligible.  Calling :py:meth:`reset` at the start of a new
    iteration restarts the profiling window so the estimate stays up-to-date as
    the model evolves.

    Parameters
    ----------
    profile_steps : int
        Number of steps to actively profile with ``torch.profiler`` per
        iteration (default: 5).
    """

    def __init__(self, profile_steps: int = 5):
        super().__init__()
        self.profile_steps = profile_steps
        self._step_start: float = 0.0
        self._profiling_steps_done: int = 0
        self._flops_estimates: list = []
        self._flops_per_step: float = 0.0
        self._cumulative_flops: float = 0.0
        self._profiler = None
        self._profiler_available: bool = True
        self._godiff_logger = None

    def set_logger(self, logger) -> None:
        """Attach a GODiffLogger for TensorBoard logging."""
        self._godiff_logger = logger

    def reset(self) -> None:
        """Reset per-iteration profiling counters.

        Called at the start of each new GO-Diff iteration so the FLOPs
        estimate is refreshed for the new model state.
        """
        self._profiling_steps_done = 0
        self._flops_estimates = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_start = time.perf_counter()
        # Profile only the first `profile_steps` steps per iteration.
        # This callback is placed *after* AdaptiveRefinementStop in the
        # callbacks list so the gradient-agreement extra backward passes are
        # not included in the FLOPs estimate.
        if self._profiler_available and self._profiling_steps_done < self.profile_steps:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            try:
                self._profiler = torch.profiler.profile(
                    activities=activities,
                    with_flops=True,
                    record_shapes=True,
                )
                self._profiler.__enter__()
            except Exception:
                self._profiler_available = False
                self._profiler = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step_wall_s = time.perf_counter() - self._step_start
        step_flops = None

        if self._profiler is not None:
            try:
                self._profiler.__exit__(None, None, None)
                raw_flops = float(
                    sum(e.flops for e in self._profiler.key_averages() if e.flops > 0)
                )
                self._flops_estimates.append(raw_flops)
                self._profiling_steps_done += 1
                self._flops_per_step = (
                    sum(self._flops_estimates) / len(self._flops_estimates)
                )
                step_flops = raw_flops
            except Exception:
                self._profiler_available = False
            finally:
                self._profiler = None
        elif self._flops_per_step > 0:
            step_flops = self._flops_per_step

        if step_flops is not None:
            self._cumulative_flops += step_flops

        if self._godiff_logger is not None:
            self._godiff_logger.log_training_step(
                trainer.global_step,
                step_wall_s=step_wall_s,
                step_flops=step_flops,
                cumulative_flops=self._cumulative_flops if step_flops is not None else None,
            )

