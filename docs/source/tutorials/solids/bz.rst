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

Here's an example input file for silicon, which you may save as ``Si.yaml``.

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
        size: [8, 8, 8]
     
    checkpoint: Si.h5

Run the above input file using 4 mpi processes as follows: 

.. code-block:: yaml

    mpirun -n 4 python -m qimpy.dft -i Si.yaml | tee Si.out

And inspect the resulting output file, ``Si.out``. First note the symmetry initialization: the Bravais lattice, in this case the face-centered Cubic structure, has 48 point group symmetries,
and 48 space group symmetries (defined with translations modulo unit cell) after including the basis, in this case the two atoms per unit cell. The number of kpoints is 8x8x8=512 kpoints, 
which is the result of the k-mesh command included above. Kpoints correspond to Bloch wave-vectors which set the phase that the wavefunction picks up when moving from one unit cell to another. 
The default is a single kpoint with wavevector [0,0,0] (also called the Gamma-point), which means that the wavefunction picks up no phase or is periodic on the unit cell. 
This was acceptable for the molecules, where we picked large enough unit cells that the wavefunctions went to zero in each cell anyway and this periodicity didn't matter. 
But now, we need to account for all possible relative phases of wavefunctions in neighbouring unit cells, which corresponds to integrating over the wave vectors in the reciprocal space unit cell, 
or equivalently the Brillouin zone. Essentially, k-mesh replaces the specified kpoint(s) (default Gamma in this case) with a uniform mesh of kpoints (8 x 8 x 8 in this case), 
covering the reciprocal space unit cell. Next, the code reduces the number of kpoints that need to be calculated explicitly using symmetries, from 512 to 29 in this case. 

To visualize the Silicon unit cell as well as its ground state density, run: 

.. code-block:: yaml

    python -m qimpy.interfaces.xsf -c Si.h5 -x Si.xsf --data-symbol n

and visualize the resulting xsf file with Vesta. 

Convergence with respect to k-point sampling
-------------------------------------------

Next, we see how the Brillouin zone sampling affects the total energies. Change the line

.. code-block:: yaml
    
    size: [8, 8, 8]

to 

.. code-block:: yaml

    size: [${nk}, ${nk}, ${nk}]
    
In addition, change the line ``checkpoint: Si.h5`` to ``checkpoint: Si-$nk.h5``. Then create the following bash script and save it as ``run.sh``: 

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

To run this script, do ``chmod +x run.sh && ./run.sh``. This should then give an output like

.. code-block:: yaml

    Relax: 0  F: -7.25985524162    fmax: +2.383e-23  t[s]: 7.58
    Relax: 0  F: -7.78880323458    fmax: +1.562e-18  t[s]: 9.00
    Relax: 0  F: -7.87596489290    fmax: +2.054e-18  t[s]: 9.89
    Relax: 0  F: -7.88279086578    fmax: +3.219e-18  t[s]: 14.09
    Relax: 0  F: -7.88293043013    fmax: +1.637e-18  t[s]: 20.86
    Relax: 0  F: -7.88293650650    fmax: +1.562e-18  t[s]: 32.93


K-point offsets (Monkhorst-Pack)
--------------------------------

We implement a k-point offset by adding an offset command to the k-mesh block of the input file, changing it from: 

.. code-block:: yaml

    k-mesh:
      size: [${nk}, ${nk}, ${nk}]

to: 

.. code-block:: yaml

    k-mesh:
      offset: [0.5, 0.5, 0.5] #Monkhorst-Pack
      size: [${nk}, ${nk}, ${nk}]

Now, running the same script to calculate the total energies as a function of k-point sampling, we obtain: 

.. code-block:: yaml

    Relax: 0  F: -7.78898667517    fmax: +6.247e-18  t[s]: 10.03
    Relax: 0  F: -7.87689497983    fmax: +1.562e-18  t[s]: 10.89
    Relax: 0  F: -7.88283670473    fmax: +4.024e-19  t[s]: 13.06
    Relax: 0  F: -7.88293668812    fmax: +4.647e-19  t[s]: 20.81
    Relax: 0  F: -7.88293663292    fmax: +4.392e-19  t[s]: 41.32    
    Relax: 0  F: -7.88293673612    fmax: +3.508e-19  t[s]: 91.51

Note also that for the 8x8x8 sampling we examined at the outset of this tutorial, we now have 60 (not 29) kpoints under symmetries. 
