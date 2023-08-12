Installing QimPy
================

Prerequisites
-------------

QimPy requires a python environment with PyTorch for the core calculations,
along with mpi4py and h5py for communication and disk I/O respectively
(in addition to standard libaries such as NumPy, SciPy and matplotlib).
QimPy's installation below will pull in most of these dependencies automatically,
but the installation of mpi4py and h5py typically require care and should be done beforehand.


Using pip (recommended in HPC environments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new python virtual environment called, say, qimpy (recommended)
within the directory where you store virtual environments, say `venvs`:

.. code-block:: bash

    venvs$ python -m venv qimpy
    venvs$ source qimpy/bin/activate
    (qimpy) venvs$

or activate the existing environment you intend to use.
(See the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_.)

Make sure you have MPI and a parallel HDF5 library installed through the package manager
on your system, or on a HPC cluster, load the appropriate modules for these.
In particular, you should now have `mpicc` accessible from your command line.

Build and install mpi4py linked to your system/HPC MPI library following the instructions
on the `mpi4py website <https://mpi4py.readthedocs.io/en/stable/install.html>`_
(or check your HPC resource if they have modified instructions):

.. code-block:: bash

    (qimpy) $ python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

Build and install h5py linked to your system/HPC *parallel* HDF5 library following the
instructions on the `h5py website <https://docs.h5py.org/en/stable/mpi.html>`_
(or check your HPC resource if they have modified instructions):

.. code-block:: bash

    (qimpy) $ CC=mpicc HDF5_MPI="ON" python -m pip install --no-cache-dir --no-binary=h5py h5py

Verify that mpi4py and h5py are working correctly in parallel mode following the examples on
the `parallel h5py page <https://docs.h5py.org/en/stable/mpi.html#using-parallel-hdf5-from-h5py>`_.


Using conda
^^^^^^^^^^^

Create a new conda environment (recommended) called, say, qimpy:

.. code-block:: bash

    $ conda create -n qimpy python
    $ conda activate qimpy
    (qimpy) $

or activate the existing environment you intend to use.
The path from where you run these commands should not matter.
(See `Managing environments <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
in the conda documentation.)

Install mpi4py and *parallel* h5py:

.. code-block:: bash

    (qimpy) $ conda install -c conda-forge mpi4py h5py=*=mpi*

Verify that mpi4py and h5py are working correctly in parallel mode following the examples on
the `parallel h5py page <https://docs.h5py.org/en/stable/mpi.html#using-parallel-hdf5-from-h5py>`_.

We may as well install the other core dependencies from conda as well:

.. code-block:: bash

    (qimpy) $ conda install numpy scipy pyyaml
    (qimpy) $ conda install pytorch -c pytorch -c nvidia

See `the PyTorch getting-started page <https://pytorch.org/get-started/locally/>`_
for options on selecting the PyTorch build suitable for your GPU / CUDA configuration.


Installation
------------

To install the last versioned QimPy from PyPI, simply:

.. code-block:: bash

    (qimpy) $ pip install qimpy

For the latest version from git:

.. code-block:: bash

    (qimpy) codes$ git clone https://github.com/shankar1729/qimpy.git
    (qimpy) codes$ cd qimpy
    (qimpy) qimpy$ python setup.py install

Do this within the directory where you want to keep the code, say `codes`
(just make sure you don't create a venv called `qimpy` and fetch
the code called `qimpy` from git into the same directory).

If you want to set-up for development, replace that last line:

.. code-block:: bash

    (qimpy) qimpy$ python setup.py develop

which will allow you to modify the code, and have it take effect
directly in the active environment (without installing again).
