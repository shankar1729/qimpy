QimPy: Quantum-Integrated Multi-PhYsics
=======================================

QimPy is an open-source electronic structure software designed to enable
tight integration with classical multi-physics and multi-scale modeling.
A key focus of this code is facilitating simultaneous performance and rapid technique development,
making it as easy to develop new electronic-structure-integrated features,
as it is to apply to materials and chemistry modeling.

Designed from the ground up in 2021, this code takes advantage of modern Python
for ease of development and PyTorch for high performance on a wide range of computing hardware.
It is intended as a successor of `JDFTx <https://jdftx.org>`_ and will develop a full feature set
for first-principles electrochemistry, carrier dynamics and transport in 2023 and 2024.
At the moment, QimPy is a fully-functional plane-wave DFT code with norm-conserving pseudopotentials,
supporing electronic structure, geometry optimization and *ab initio* molecular dynamics calciulations.

QimPy is built on PyTorch as a hardware abstraction layer, and fully supports CPUs,
NVIDIA GPUs and likely also AMD GPUs through the corresponding PyTorch device layers.
The use of ML libraries as the underlying layer presents a unique advantage:
use of the specialized tensor cores in the GPUs without the need for hand-tuned kernels.
QimPy is designed to scale to large numbers of GPUs  by efficiently overlapping
all inter-GPU communications with the primary computation involving wavefunction transforms.


.. toctree::
    :maxdepth: 1

    install
    tutorials/index
    inputfile
    api
    development/index

Auxiliary packages
------------------

.. toctree::
    :maxdepth: 1

    transport/index
