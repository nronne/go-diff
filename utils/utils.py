import numpy as np
from ase.io import read

from agox.models.descriptors.voronoi import Voronoi


heptamer = read('heptamer.traj')
descripter = Voronoi(environment=None, indices=np.arange(len(heptamer))[-7:], covalent_bond_scale_factor=1.1)
heptamer_descriptor = descripter.convert_matrix_to_eigen_value_string(descripter.get_bond_matrix(heptamer))

def classify_heptamer(atoms):
    d = descripter.convert_matrix_to_eigen_value_string(descripter.get_bond_matrix(atoms))
    if d == heptamer_descriptor:
        return 1
    else:
        return 0


    
