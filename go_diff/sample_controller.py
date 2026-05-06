import numpy as np

class SampleController:
    def __init__(
        self, 
        initial_N=64, 
        max_N=256, 
        target_ess_ratio=0.5, 
    ):
        """
        Args:
            initial_N: Starting number of samples.
            min_N: Minimum samples allowed (hardware/budget floor).
            max_N: Maximum samples allowed (VRAM/compute ceiling).
            target_ess_ratio: Ideal ratio of ESS to N (usually 0.1 - 0.4).
            growth_factor: Multiplier to increase N.
            shrink_factor: Multiplier to decrease N.
        """
        self.initial_N = initial_N
        self.max_N = max_N
        self.target_ess_ratio = target_ess_ratio

    def calculate_ess(self, energies, temperature):
        """Calculates Effective Sample Size from Boltzmann weights."""
        if len(energies) == 0:
            return 0
        
        # Calculate weights: w = exp(-E/T) / sum(exp(-E/T))
        # Use softmax for numerical stability
        energies = np.array(energies)
        e = np.exp(-energies/temperature)
        weights = e / np.sum(e)
        
        # ESS = 1 / sum(w^2)
        ess = 1.0 / np.sum(weights**2)
        return ess

    def continue_sampling(self, energies, temperature=None):
        if len(energies) == 0:
            return True

        if len(energies) >= self.max_N:
            return False
        
        if temperature is None:
            if len(energies) <= self.initial_N:
                return True
            else:
                return False
            
        """Determines whether to continue sampling based on ESS ratio."""
        current_ess = self.calculate_ess(energies, temperature)
        current_ess_ratio = current_ess / len(energies)
        
        # Continue sampling if ESS ratio is below target
        continue_sampling = current_ess_ratio < self.target_ess_ratio
        
        if continue_sampling:
            print(f"Continue sampling: ESS ratio {current_ess_ratio:.3f} does not meet target of {self.target_ess_ratio}.")                  
        else:
            print(f"Stopping sampling: ESS ratio {current_ess_ratio:.3f} meets target.")
        
            
        return continue_sampling
