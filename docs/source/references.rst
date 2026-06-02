References and citation
=======================

Primary publications
--------------------

1. N. Rønne, T. Vegge and A. Bhowmik
   *GO-Diff: Data-free and amortized global structure optimization*
   arXiv preprint **arXiv:2510.13448** (2025).
   URL: https://arxiv.org/abs/2510.13448

2. N. Rønne and B. Hammer,
   *Atomistic Generative Diffusion for Materials Modeling*,
   arXiv preprint **arXiv:2507.18314** (2025).
   URL: https://arxiv.org/abs/2507.18314

3. N. Rønne, A. Aspuru-Guzik, and B. Hammer,
   *Generative Diffusion Model for Surface Structure Discovery*,
   **Physical Review B** **110**, 235427 (2024).
   DOI: https://doi.org/10.1103/PhysRevB.110.235427

How this documentation maps to the papers
------------------------------------------

- GO-Diff including buffered training and Boltzmann-weighting is
  described in ref. 1.
- The atomistic diffusion model package AGeDi used as the diffusion
  model in GO-Diff is presented in ref. 2
- The surface supported diffusion model including its confinement and
  correspond methodology is introduced in ref. 3.


Suggested citation
------------------

If you use GO-Diff in academic work, please cite the GO-Diff preprint,
the AGeDi preprint and the PRB paper::

  @misc{ronne2025A,
      title={GO-Diff: Data-free and amortized global structure optimization}, 
      author={Nikolaj Rønne and Tejs Vegge and Arghya Bhowmik},
      year={2025},
      eprint={2510.13448},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2510.13448}, 
      }

  @misc{ronne2025B,
      title={Atomistic Generative Diffusion for Materials Modeling}, 
      author={Nikolaj Rønne and Bjørk Hammer},
      year={2025},
      eprint={2507.18314},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2507.18314}, 
  }
      

If investigating surface supported systems please also cite::
  
   @article{ronne2024,
     title   = {Generative Diffusion Model for Surface Structure Discovery},
     author  = {R{\o}nne, Nikolaj and Aspuru-Guzik, Al{\'a}n and Hammer, Bj{\o}rk},
     journal = {Physical Review B},
     volume  = {110},
     pages   = {235427},
     year    = {2024},
     doi     = {10.1103/PhysRevB.110.235427},
   }

If you make use of AGOX utilities please also cite::

   @article{christiansen2022,
     title   = {Atomistic Global Optimization X: A Python Package for
                Optimization of Atomistic Structures},
     author  = {Christiansen, Mads-Peter V. and R{\o}nne, Nikolaj and
                Hammer, Bj{\o}rk},
     journal = {The Journal of Chemical Physics},
     volume  = {157},
     pages   = {054701},
     year    = {2022},
     doi     = {10.1063/5.0094165},
   }
