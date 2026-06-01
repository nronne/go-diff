Installation
============

Requirements
------------

- Python >= 3.11
- A CUDA-capable GPU is strongly recommended for training and sampling.

Install GO-Diff
---------------

.. code-block:: console

   pip install go-diff

Step 3 – (Optional) Paper-reproduction dependencies
----------------------------------------------------

To reproduce the results from the paper you additionally need MACE and the
full AGOX package:

.. code-block:: console

   pip install mace-torch
   pip install "agox[full]"

Developer install
-----------------

Clone the repository and install in editable mode with the test extras:

.. code-block:: console

   git clone https://github.com/nronne/go-diff.git
   cd go-diff
   pip install -e ".[test]"

Verify installation
-------------------

.. code-block:: console

   python -c "from go_diff import GODiff; print('GO-Diff installed successfully')"
