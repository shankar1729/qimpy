Brillouin-zone sampling
=======================

TODO: describe why we need to do :math:`\int_{BZ}d\vec{k} \ldots`

The previous section dealt entirely with molecular calculations,
where we calculate properties of isolated systems,
surrounded by vacuum or liquid in all three dimensions.
Now we move to crystalline materials, which are periodic in all three dimensions,
starting with Brillouin zone sampling using the example of silicon.

Previously, we used the lattice command primarily to create a large enough box
to contain our molecules, and used %Coulomb truncation to isolate periodic images.
Now, in a crystalline solid, the lattice vectors directly specify
the periodicity of the crystal and are not arbitrary.
Silicon has a diamond lattice structure with a cubic lattice constant
of 5.43 Angstroms (10.263 bohrs).
In each cubic unit cell, there would be 8 silicon atoms:
at vertices, face centers and two half-cell body centers.
However, this does not capture all the spatial periodicity of the lattice
and we can work with the smaller unit cell of the face-centered Cubic lattice,
which will contain only two silicon atoms.

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
