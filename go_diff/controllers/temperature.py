import numpy as np

class TemperatureController:
    def __init__(self, k=1.0, fast=0.5, slow=0.95):
        self.k = k
        self.fast = fast
        self.slow = slow
        self.temperature = None
        
        self.history = []

    def get_temperature(self):
        if self.temperature is None:
            raise ValueError("Temperature schedule not initialized.")
        return self.temperature

    def next(self, energies):
        if self.temperature is None:
            self.temperature = np.std(energies) * self.k
        else:
            C = self.compute_heat_capacity(energies)
            print(f"Temperature: {self.temperature.item():.4f}, variance: {np.var(energies):.4f}, std: {np.std(energies):.4f}, C: {C.item():.4f}")
            if C > 1:
                self.temperature *= self.slow
            else:
                self.temperature *= self.fast
                
        
        self.history.append(self.temperature)
        
        return self.temperature.item()

    def compute_heat_capacity(self, energies):
        """
        Computes the effective Heat Capacity C(T) using energy variance.
        Args:
            energies: Tensor of shape (N,) containing evaluated energies.
            temperature: Current annealing temperature T.
        """
        variance = np.var(energies)
        return variance / (self.temperature**2 + 1e-8)
