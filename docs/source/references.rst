References and citation
=======================

Primary publications
--------------------

1. N. Rønne and B. Hammer,
   *Atomistic Generative Diffusion for Materials Modeling*,
   arXiv preprint **arXiv:2507.18314** (2025).
   URL: https://arxiv.org/abs/2507.18314

2. N. Rønne, A. Aspuru-Guzik, and B. Hammer,
   *Generative Diffusion Model for Surface Structure Discovery*,
   **Physical Review B** **110**, 235427 (2024).
   DOI: https://doi.org/10.1103/PhysRevB.110.235427

3. M.-P. V. Christiansen, N. Rønne, and B. Hammer,
   *Atomistic Global Optimization X: A Python Package for Optimization of
   Atomistic Structures*,
   **The Journal of Chemical Physics** **157**, 054701 (2022).
   DOI: https://doi.org/10.1063/5.0094165

How this documentation maps to the papers
------------------------------------------

- The GO-Diff outer loop (sample → evaluate → buffer → train) is the
  algorithm described in the AGeDi preprint (reference 1).
- The surface-template workflow, confinement, and Boltzmann-weighted diffusion
  correspond to the methodology introduced in the PRB paper (reference 2).
- AGOX functionality (reference 3) is used optionally for analysis utilities
  in ``utils/``.

Suggested citation
------------------

If you use GO-Diff in academic work, please cite the AGeDi preprint and the
PRB paper::

   @misc{ronne2025agedi,
     title  = {Atomistic Generative Diffusion for Materials Modeling},
     author = {R{\o}nne, Nikolaj and Hammer, Bjørk},
     year   = {2025},
     eprint = {2507.18314},
     archivePrefix = {arXiv},
   }

   @article{ronne2024generative,
     title   = {Generative Diffusion Model for Surface Structure Discovery},
     author  = {R{\o}nne, Nikolaj and Aspuru-Guzik, Al{\'a}n and Hammer, Bj{\o}rk},
     journal = {Physical Review B},
     volume  = {110},
     pages   = {235427},
     year    = {2024},
     doi     = {10.1103/PhysRevB.110.235427},
   }

If you also use AGOX utilities, cite reference 3 as well::

   @article{christiansen2022agox,
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
