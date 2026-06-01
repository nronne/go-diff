import torch
from agedi.diffusion.noisers import ConfinedCellPositions, Positions
from agedi.data import AtomsGraph


class _WeightedLossMixin:
    """Mixin that replaces the standard position-score loss with a
    Boltzmann-weighted variant.

    Expects the batch to carry a ``weight`` attribute (one scalar weight per
    structure) that is broadcast to the per-atom level before computing the
    mean squared error between the predicted and target score.
    """

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Compute the Boltzmann-weighted denoising score-matching loss.

        Parameters
        ----------
        batch : AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        t = batch.time
        r_score = batch[self.key + "_score"]
        r_noise = batch[self.key + "_noise"]

        weights = batch.weight
        weights = weights.repeat_interleave(batch.n_atoms.view(-1), dim=0)

        var = self.sde.var(t)

        r_score = batch.apply_mask(r_score)

        return torch.mean(
            weights * torch.sum((r_noise + r_score * var) ** 2, dim=-1)
        )


class WeightedConfinedCellPositions(_WeightedLossMixin, ConfinedCellPositions):
    """Boltzmann-weighted variant of :class:`agedi.diffusion.noisers.ConfinedCellPositions`.

    Assumes that the batch has a ``weight`` attribute.
    """

    _key = "pos"


class WeightedPositions(_WeightedLossMixin, Positions):
    """Boltzmann-weighted variant of :class:`agedi.diffusion.noisers.Positions`.

    Assumes that the batch has a ``weight`` attribute.
    """

    _key = "pos"
    
