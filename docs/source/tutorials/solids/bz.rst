Brillouin-zone sampling
=======================

TODO: describe why we need to do :math:`\int_{BZ}d\vec{k} \ldots`


Here's an example input file for silicon

.. code-block:: yaml

    # Diamond-cubic silicon
    lattice:
      system: cubic
      modification: face-centered
      a: 10.26  # bohrs
      
    ions:
      pseudopotentials:
        - ../../../JDFTx/build_testing/pseudopotentials/SG15/$ID_ONCV_PBE.upf
      coordinates:
        - [Si, 0.125, 0.07, 0.01]
        - [Si, 0.375, 0.32, 0.26]
    
    electrons:
      k-mesh:
        offset: [0.5, 0.5, 0.5] #Monkhorst-Pack
        size: [8, 8, 8]
    
    grid:
      ke-cutoff: 100
    
    checkpoint: null  # disable reading checkpoint
    checkpoint-out: Si.h5  # but still create it

