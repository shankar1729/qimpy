Getting started
===============

Follow :doc:`/install` to setup QimPy in a python virtual environment
or conda environment, and make sure that environment is active
before running any of the tutorials.
We'll assume you called that environment `qimpy`,
indicated by the prefix (qimpy) on all the shells shown here.

The tutorials will guide you through the construction of a YAML input file, say `in.yaml`.
To run QimPy using a single process and all available CPU threads:

.. code-block:: bash

    (qimpy) $ python -m qimpy.dft -i in.yaml

To use multiple processes using MPI, *e.g.*, using 4 processes and assuming OpenMPI:

.. code-block:: bash

    (qimpy) $ mpirun -n 4 python -m qimpy.dft -i in.yaml

With MPI and to leverage GPUs, *e.g.*, assuming 4 GPUs available on the system:

.. code-block:: bash

    (qimpy) $ CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun -n 4 python -m qimpy.dft -i in.yaml

Note that QimPy will not use GPUs unless explicitly instructed to using CUDA_VISIBLE_DEVICES.

Within a SLURM batch job file, request cores and GPUs as specified for your HPC resource and:

.. code-block:: bash

    (qimpy) $ srun python -m qimpy.dft -i in.yaml

In this case, SLURM will set all required environment variables, including CUDA_VISIBLE_DEVICES
and SLURM_CPUS_PER_TASK, which QimPy will use to select the appropriate GPU and CPUs.


Pseudopotentials
----------------

QimPy currently supports norm-conserving UPF pseudopotentials,
but does not distribute any pseudopotentials with the code.
You can refer to pseudopotentials with relative or absolute paths
in each calculation, but this can be cumbersome.
The instructions below will get you started with the
`SG15 pseudopotentials <http://www.quantum-simulation.org/potentials/sg15_oncv/>`_
in a path specified to QimPy with an environment variable,
so that you do not need to specify absolute paths for each calculation.
The tutorials assume that you have the SG15 pseudopotentials set up this way.

Create a directory, say /path/to/pseudos where you want to store your pseudopotentials:

.. code-block:: bash

    $ cd /path/to/pseudos
    pseudos$ wget https://github.com/shankar1729/jdftx/blob/master/jdftx/pseudopotentials/SG15.tgz
    pseudos$ wget https://github.com/shankar1729/jdftx/blob/master/jdftx/pseudopotentials/SG15-pulay.tgz
    pseudos$ tar xvzf SG15.tgz
    pseudos$ tar xvzf SG15-pulay.tgz

Alternately, if you have `JDFTx <https://jdftx.org>`_ installed, you already have these files
and /path/to/pseudos can just be taken as jdftx-build-dir/pseudopotentials.

Set the environment variable before running QimPy calculations:

.. code-block:: bash

    $ export QIMPY_PSEUDO_DIR=/path/to/pseudos

and add this to your .bashrc so that this takes effect in all subsequent terminal sessions.
