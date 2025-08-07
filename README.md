# GO-Diff

## Installation 
To run GO-Diff install the AGeDi package
```
git clone https://github.com/nronne/agedi.git
cd agedi
git checkout boltzmann-diffusion
pip install .
pip install matscipy schnetpack
```

If you want to reproduce the results in the paper also install MACE and AGOX

```
pip install mace-torch
pip install agox['full']
```

## Scripts
Scripts to reproduce the results in the paper are located in `scripts` with utils for measuring success is located in `utils`
