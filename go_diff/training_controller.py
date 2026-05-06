import torch
from lightning.pytorch.callbacks import Callback
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Batch

class AdaptiveRefinementStop(Callback):
    def __init__(self, min_steps=100, patience=50, smooth_factor=0.9, check_interval=1):
        super().__init__()
        self.min_steps = min_steps
        self.patience = patience
        self.smooth_factor = smooth_factor
        self.check_interval = check_interval
        
        self.ema_agreement = 0.0
        self.max_agreement = -1.0
        self.patience_counter = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step < self.min_steps or trainer.global_step % self.check_interval != 0:
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

        # 4. Logic: If agreement has significantly dropped from its peak, stop.
        # This means the model has finished learning the "consensus" and is now over-fitting.
        if self.patience_counter >= self.patience and self.ema_agreement < (0.5 * self.max_agreement):
            print(f"\n[Adaptive Stop] Agreement peaked at {self.max_agreement:.4f} "
                  f"and dropped to {self.ema_agreement:.4f}. Stopping.")
            trainer.should_stop = True
        else:
            print(f"\n[Adaptive Stop] Step {trainer.global_step}: "
                  f"Current Agreement: {current_agreement:.4f}, "
                  f"EMA Agreement: {self.ema_agreement:.4f}, "
                  f"Max Agreement: {self.max_agreement:.4f}, "
                  f"Patience Counter: {self.patience_counter}")

            
    def _calculate_split_agreement(self, trainer, pl_module, batch, batch_idx):
        # 1. Split the batch into two independent halves
        # Assumes batch is (x, energies) or similar. Adapt if your batch structure differs.
        half = len(batch) // 2
        if half < 2: return # Need at least 2 samples per side


        opt = trainer.optimizers[0]
        # 2. Calculate Gradient A (First Half)
        pl_module.zero_grad()
        loss_a = pl_module.loss(Batch.from_data_list(batch[:half]), batch_idx)
        trainer.strategy.backward(loss_a["loss"], optimizer=opt)
        grad_a = self._get_flat_grad(pl_module)

        # 3. Calculate Gradient B (Second Half)
        pl_module.zero_grad()
        loss_b = pl_module.loss(Batch.from_data_list(batch[:half]), batch_idx)
        trainer.strategy.backward(loss_b["loss"], optimizer=opt)
        grad_b = self._get_flat_grad(pl_module)

        # 4. Compute Agreement (Cosine Similarity)
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
        self.min_steps += trainer.current_epoch
    
