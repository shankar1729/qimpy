Silicon AIMD
===================

We will show an ab-initio molecular dynamics simulation with a silicon crystal.

TODO: insert example input
.. code-block:: yaml

    lattice:
      system: cubic
      a: 10.0

    ions:
      pseudopotentials:
        - ../../../../JDFTx/build_testing/pseudopotentials/SG15/$ID_ONCV_PBE.upf
      fractional: no
      coordinates:
        - [H, 0., -1.432, +0.6]
        - [H, 0., +1.432, +0.6]
        - [O, 0.,  0.000, -0.6]

    checkpoint: water.h5

