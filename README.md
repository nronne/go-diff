# GO-Diff

## Installation 
To run GO-Diff install the AGeDi package
```
git clone https://github.com/nronne/agedi.git
cd agedi
git checkout boltzmann-diffusion
pip install .
pip install matscipy schnetpack

cd ..
git clone https://github.com/nronne/go-diff.git
cd go-diff
pip install .
```

If you want to reproduce the results in the paper also install MACE and AGOX

```
pip install mace-torch
pip install agox['full']
```

## Scripts
Scripts to reproduce the results in the paper are located in `scripts`
with utils for identifying the Pt-heptamer structure is located in `utils`.

# CITE
If you use GO-Diff please also cite:

Rønne, Nikolaj, and Bjørk Hammer. “Atomistic Generative Diffusion for Materials Modeling.” arXiv:2507.18314. Preprint, arXiv, July 24, 2025. https://arxiv.org/abs/2507.18314

And if studying surface supported systems also:
Rønne, Nikolaj, Alán Aspuru-Guzik, and Bjørk Hammer. “Generative Diffusion Model for Surface Structure Discovery.” Physical Review B 110, no. 23 (2024): 235427. https://doi.org/10.1103/PhysRevB.110.235427

Optionally if using any AGOX functionality please also cite:
Christiansen, Mads-Peter V., Nikolaj Rønne, and Bjørk
Hammer. “Atomistic Global Optimization X: A Python Package for
Optimization of Atomistic Structures.” The Journal of Chemical Physics
157, no. 5 (2022): 054701. https://doi.org/10.1063/5.0094165
