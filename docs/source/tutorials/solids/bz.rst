Brillouin-zone sampling
=======================

TODO: describe why we need to do :math:`\int_{BZ}d\vec{k} \ldots`


Here's an example input file for silicon

.. code-block:: yaml

    # Diamond-cubic silicon
    lattice:
      system: cubic
      modification: face-centered
      a: 5.43 Å

    ions:
      pseudopotentials:
        - SG15/$ID_ONCV_PBE.upf
      coordinates:
        - [Si, 0.00, 0.00, 0.00]
        - [Si, 0.25, 0.25, 0.25]
    
    electrons:
      k-mesh:
        offset: [0.5, 0.5, 0.5] #Monkhorst-Pack
        size: [8, 8, 8]
    
    grid:
      ke-cutoff: 100
    
    checkpoint: Si.h5
