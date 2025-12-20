============
Installation
============

Environment setting (optional)
==============================
For using scASprofiler, we recommend creating a separated `conda environment`_ for 
easier and cleaner management of dependent packages, particularly pytorch, 
which are under active development.

The following command line in terminal (Linux or MacOS) will create a conda 
environment with name ``scasp``.(tested on Python 3.10.9):

install
============

.. code-block:: bash

    git clone https://github.com/1027victor/scASprofiler.git
    cd scASprofiler
    conda env create -f environment.yaml
    conda activate scasp
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
    pip install .

Test
====

In order to test the installation, you could type ``scASprofiler-perp, scASprofiler-impute, scASprofiler-quantify``. 
If you have any issues, please report them to the issue on `scASprofiler issues`_.

.. _scASprofiler issues: https://github.com/1027victor/scASprofiler/issue
