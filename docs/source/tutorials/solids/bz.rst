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


Next, we see how the Brillouin zone sampling affects the total energies. 

Change the line

.. code-block:: yaml
	
	size: [8, 8, 8]

to 

.. code-block:: yaml

	size: [${nk}, ${nk}, ${nk}]
	
Then create the following bash script

.. code-block:: yaml

	#!/bin/bash
	for nk in 1 2 4 8 12 16; do
		export nk  #Export adds shell variable nk to the enviornment
               #Without it, nk will not be visible to jdftx below
		mpirun -n 4 python -m qimpy.dft -i Si.yaml | tee Si-$nk.out
		done

	for nk in 1 2 4 8 12 16; do
		grep "Relax" Si-$nk.out
	done

This should then give an output like

.. code-block:: yaml

	Relax: 0  F: -7.78898685689    fmax: +5.100e-18  t[s]: 6.03
	Relax: 0  F: -7.87689519309    fmax: +2.343e-18  t[s]: 6.46
	Relax: 0  F: -7.88283692282    fmax: +3.018e-19  t[s]: 7.66
	Relax: 0  F: -7.88293690637    fmax: +2.803e-19  t[s]: 15.24
	Relax: 0  F: -7.88293685116    fmax: +2.520e-19  t[s]: 38.74
	Relax: 0  F: -7.88293695436    fmax: +2.803e-19  t[s]: 78.41
