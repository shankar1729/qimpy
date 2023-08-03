A first calculation
===================

QimPy is an electronic density-functional theory (DFT) software,
which means that its primary functionality is to calculate
the quantum-mechanical energy of electrons in an external potential,
typically from nuclei (ions) in a molecule or solid.
Specifically, the underlying theory is Kohn-Sham DFT
which involves solving the single-particle Schrodinger equation
in a self-consistent potential determined from the electron density.
This tutorial demonstrates a DFT calculation of the energy
and electron density of a water molecule.

A lot of effort in the research behind QimPy has involved water:
its dielectric response, equation of state, free energy functional etc.
Therefore, a water molecule is a fitting first calculation.
Save the following to `water.yaml`:

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

See :doc:`/inputfile` for details on all the settings that can be specified in YAML input.

TODO: change. This input-file illustrates the bare minimum commands needed to set up a calculation.
The lattice and ion commands set up the unit cell and geometry,
and elec-cutoff controls the resolution of the plane-wave basis
that is used to represent the wavefunctions and densities. 
See \ref Commands for a list of all available input file commands and their options.

TODO: change. In a plane-wave basis, bare nuclei are replaced by an effective potential
due to the nucleus and core electrons, termed the pseudopotential,
and only the remaining valence electrons are included explicitly in the DFT calculation.
The ion-species commands select the GBRV pseudopotentials distributed with QimPy.
These are installed to the build directory, and QimPy will automatically look for them there.
If you use pseudopotentials not built into QimPy, specify the absolute or relative paths instead.
See the \ref Pseudopotentials page for supported pseudopotential formats,
other sets of pseudopotentials distributed with QimPy, and a wildcard syntax
for selecting an entire set of pseudopotentials (which we shall use henceforth).

Now, that basic input file can be run with

.. code-block:: bash

    python -m qimpy.dft -i water.yaml | tee water.out

TODO: change. That should complete in a few seconds and create files water.out and water.h5.
Note that the default behavior of qimpy is to concatenate output files.
If you wish to overwrite a previous water.out file, then add option -d to the command line. 

TODO: change. Have a look at water.out.
It lists the commands that were issued in the input file along with several more which have sensible defaults.
For instance, the exchange functional defaults to GGA (<b>elec-ex-corr gga-PBE</b>).
To use LDA instead, you would add command <b>elec-ex-corr lda</b> to the input file.
Other commands do not have defaults, such as \ref CommandVanDerWaals to include dispersion
interactions which standard DFT functionals miss (discussed later in \ref Dispersion tutorial);
so don't forget to see \ref Commands for the full list of available commands.

TODO: change. The commands section is followed by initialization of the plane-wave grid, symmetries, pseudopotentials etc.,
and then the electronic minimization which logs the progress of the conjugate gradients minimizer
(lines starting with ElecMinimize).
The default is to minimize for 100 iterations or an energy difference between
consecutive iterations of 10<sup>-8</sup> Hartrees, whichever comes first.
This example converges to that accuracy in around 15 iterations.
Note that the ions have not been moved and the end of the output file lists the forces at the initial position.

TODO: change. Additional output files are written based on the options provided by the dump command.
Here, we have requested a dump of energy components (in water.Ecomponents)
and electron density (in water.n) at the end of the run, 
with filenames specified by dump-name.

TODO: change. Finally, let's visualize the electron density output by this calculation.
Use the createXSF script to create water.xsf containing the
ionic geometry (extracted from water.out) and electron density (from water.n):

.. code-block:: bash

    python -m qimpy.interfaces.xsf -c water.h5 -x water.xsf -d n

TODO: change. Note that the script recognizes dump-name, so n will be understood to be water.n,
but you can also specify the file name directly.
Now open the XSF file using the visualization program VESTA
(or another program that supports XSF such as XCrysDen).
You should initially see the water molecule torn between the
corners of the box since it was centered at [0,0,0].
Change the visualization boundary settings from [0,1) to [-0.5,0.5)
to see the (intact molecule) image at the top of the page!

