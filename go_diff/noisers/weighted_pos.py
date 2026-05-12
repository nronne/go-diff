import torch
from typing import Dict
from agedi.diffusion.noisers import Positions

from typing import Dict
from agedi.data import AtomsGraph



class WeightedPositions(Positions):
    """
    Assumes that the batch has attribute `weight`.
    
    """
    
    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the loss for the weighted positions noiser.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        torch.Tensor
            The loss for the weighted positions noiser.

        """

        t = batch.time
        r_score = batch[self.key + "_score"]
        r_noise = batch[self.key + "_noise"]

        weights = batch.weight
        weights = weights.repeat_interleave(batch.n_atoms.view(-1), dim=0)

        var = self.sde.var(t)

        r_score = batch.apply_mask(r_score)

        loss = torch.mean(
            weights * torch.sum((r_noise + r_score * var) ** 2, dim=-1)
        )        


        return loss
        
