import time
import torch
from lightning.pytorch.callbacks import Callback
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Batch


class MomentumConsensusStop(Callback):
    def __init__(self, min_steps=50, patience=100, drop_factor=0.5):
        """
        Args:
            min_steps: Minimum steps to allow for momentum to build up.
            patience: How many steps to wait after agreement starts dropping.
            drop_factor: Stop if agreement falls below (drop_factor * max_agreement).
        """
        super().__init__()
        self.min_steps = min_steps
        self.patience = patience
        self.drop_factor = drop_factor
        
        self.max_agreement = -1.0
        self.patience_counter = 0
        self.current_step = 0

        # Optional GODiffLogger for TensorBoard logging
        self._godiff_logger = None

    def set_logger(self, logger):
        """Attach a GODiffLogger so agreement metrics are written to TensorBoard."""
        self._godiff_logger = logger


    def on_after_backward(self, trainer, pl_module):
        """
        Called immediately after loss.backward(). 
        Gradients are now available in pl_module.parameters().
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
            
        
    def on_train_start(self, trainer, pl_module):
        """Reset state at the start of every GO-Diff iteration."""
        self.max_agreement = -1.0
        self.patience_counter = 0
        self.current_step = 0
        

class AdaptiveRefinementStop(Callback):
    def __init__(self, min_steps=100, patience=50, smooth_factor=0.75, check_interval=1):
        """
        Args:
            min_steps: Minimum global steps before starting to check for stopping.
            patience: Number of consecutive checks with low agreement before stopping.
            smooth_factor: EMA smoothing factor for agreement tracking (0 < smooth_factor < 1).
            check_interval: How often (in steps) to check the agreement.
        """
        super().__init__()
        self._min_steps = min_steps
        self.min_steps = min_steps
        self.patience = patience
        self.smooth_factor = smooth_factor
        self.check_interval = check_interval
        
        self.ema_agreement = 0.0
        self.max_agreement = -1.0
        self.patience_counter = 0

        # Optional GODiffLogger for TensorBoard logging
        self._godiff_logger = None

    def set_logger(self, logger):
        """Attach a GODiffLogger so agreement metrics are written to TensorBoard."""
        self._godiff_logger = logger

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
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
            
    def _calculate_split_agreement(self, trainer, pl_module, batch, batch_idx):
        # 1. Split the batch into two independent halves
        # Assumes batch is (x, energies) or similar. Adapt if your batch structure differs.
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



    def _get_flat_grad(self, pl_module):
        """Helper to flatten all model gradients into a single vector."""
        grads = []
        for param in pl_module.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([])


    def reset(self, trainer):
        """Resets the internal state of the callback."""
        self.ema_agreement = 0.0
        self.max_agreement = -1.0
        self.patience_counter = 0
        self.min_steps = trainer.global_step + self._min_steps


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

