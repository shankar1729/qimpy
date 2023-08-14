Brillouin-zone sampling
=======================

The previous section dealt entirely with molecular calculations,
where we calculate properties of isolated systems,
surrounded by vacuum or liquid in all three dimensions.
Now we move to crystalline materials, which are periodic in all three dimensions,
starting with Brillouin zone sampling using the example of silicon.

Previously, we used the lattice command primarily to create a large enough box
to contain our molecules, and used Coulomb truncation to isolate periodic images.
Now, in a crystalline solid, the lattice vectors directly specify
the periodicity of the crystal and are not arbitrary.
Silicon has a diamond lattice structure with a cubic lattice constant
of 5.43 Angstroms (10.263 bohrs).
In each cubic unit cell, there are 8 silicon atoms:
at vertices, face centers and two half-cell body centers.
However, this does not capture all the spatial periodicity of the lattice
and we can work with the smaller unit cell of the face-centered Cubic lattice,
which will contain only two silicon atoms.

Here's an example input file for silicon, which you may save as Si.yaml.

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

Run the above input file using 4 mpi processes as follows: 

.. code-block::yaml

	mpirun -n 4 python -m qimpy.dft -i Si.yaml | tee Si.out 

And inspect the resulting output file, Si.out. First note the symmetry initialization: the Bravais lattice, in this case the face-centered Cubic structure, has 48 point group symmetries,
and 48 space group symmetries (defined with translations modulo unit cell) after including the basis, in this case the two atoms per unit cell. The number of kpoints is 8x8x8=512 kpoints, 
which is the result of the k-mesh command included above. Kpoints correspond to Bloch wave-vectors which set the phase that the wavefunction picks up when moving from one unit cell to another. 
The default is a single kpoint with wavevector [0,0,0] (also called the Gamma-point), which means that the wavefunction picks up no phase or is periodic on the unit cell. 
This was acceptable for the molecules, where we picked large enough unit cells that the wavefunctions went to zero in each cell anyway and this periodicity didn't matter. 
But now, we need to account for all possible relative phases of wavefunctions in neighbouring unit cells, which corresponds to integrating over the wave vectors in the reciprocal space unit cell, 
or equivalently the Brillouin zone. Essentially, k-mesh replaces the specified kpoint(s) (default Gamma in this case) with a uniform mesh of kpoints (8 x 8 x 8 in this case), 
covering the reciprocal space unit cell. Next, the code reduces the number of kpoints that need to be calculated explicitly using symmetries, from 512 to 60 in this case. 

To visualize the Silicon unit cell as well as its ground state density, run: 

.. code-block:: yaml

	python -m qimpy.interfaces.xsf -c Si.h5 -x Si.xsf --data-symbol n

and visualize the resulting xsf file with Vesta. 

Next, we see how the Brillouin zone sampling affects the total energies. Change the line

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
