import numpy as np
from ase.io import read

from agox.models.descriptors.voronoi import Voronoi


_descripter = None
_heptamer_descriptor = None


def _get_descriptor():
    global _descripter, _heptamer_descriptor
    if _descripter is None:
        heptamer = read('heptamer.traj')
        _descripter = Voronoi(
            environment=None,
            indices=np.arange(len(heptamer))[-7:],
            covalent_bond_scale_factor=1.1,
        )
        _heptamer_descriptor = _descripter.convert_matrix_to_eigen_value_string(
            _descripter.get_bond_matrix(heptamer)
        )
    return _descripter, _heptamer_descriptor


def classify_heptamer(atoms):
    descripter, heptamer_descriptor = _get_descriptor()
    atoms_descriptor = descripter.convert_matrix_to_eigen_value_string(descripter.get_bond_matrix(atoms))
    if atoms_descriptor == heptamer_descriptor:
        return 1
    else:
        return 0
