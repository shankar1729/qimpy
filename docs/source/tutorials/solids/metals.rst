Metals
======

Silicon, the crystalline solid we have worked with is a semiconductor with a band gap which we identified in the tutorials on :doc:`band_structures` calculations. This gap clearly demarcates occupied states from unoccupied states for all k-points, so electron 'fillings' are straightforward: the lowest (nElectrons/2) bands of all k-points are filled, and the remaining are empty. This is no longer true for metals, where one or more bands are partially filled, and these fillings (or occupation factors) must be optimized self-consistently. This tutorial introduces such a calculation for platinum.

Platinum is a face-centered cubic metal with a cubic lattice constant of 3.92 Angstroms (7.41 bohrs), which we can specify easily in the input file Pt.in:

.. code-block:: yaml

    lattice:
      system: cubic
        modification: face-centered
        a: 3.92 Å
     
    ions:
      pseudopotentials:
        - SG15/$ID_ONCV_PBESOL-1.0.upf
      coordinates:
        - [Pt, 0, 0, 0] #just one atom per unit cell
     
    electrons:
      k-mesh: 
        size: [12, 12, 12]
       
      fillings:
        smearing: fermi
        kT: 0.01 #Hartree
      
      basis:
        grid:
          ke-cutoff: 100 #Hartree
      
      xc:
        functional: gga_pbesol
      
      checkpoint_out: Pt_out.h5

The only new command is elec-smearing which specifies that the fillings must be set using a Fermi function based on the current Kohn-Sham eigenvalues, rather than determined once at startup (the default we implicitly used so far). The first parameter of this command specifies the functional form for the occupations, in this case selecting a Fermi function (see :doc:`fillings </yamldoc/qimpy.dft.electrons.Fillings>` for other options). The second parameter determines the width in energy over which the Fermi function goes from one to zero (occupied to unoccupied).

Metals are much more sensitive to k-point sampling, because for some bands, a part of the Brillouin zone is filled and the remainder is empty, separated by a Fermi surface. The number of k-points directly determines how accurately the Fermi surface(s) can be resolved. The Fermi temperature is essentially a smoothing parameter that allows us to resolve the Fermi surface accurately at moderate k-point counts. Note that the tempreature we chose is around ten times room temperature in order to increase the smoothing and use a practical number of k-points.

Finally, we have selected the PBEsol exchange-correlation functional using the **functional** command, which is generally more accurate than PBE for solids (especially for metals). Correspondingly, we selected PBEsol pseudopotentials from the SG15 set.

Run the calculation using 

.. code-block:: bash

  (qimpy) $ python -m qimpy.dft -i Pt.in -o Pt.out

and examine the output file. The main difference is that every SCF line is preceded by a line starting with FillingsUpdate. This line reports the chemical potential, mu, of the Fermi function that produces the correct number of electrons with the current Kohn-Sham eigenvalues, and the number of electrons, nElectrons, which of course stays constant.

Also notice that the energy in the SCF lines is called F instead of Etot. The total energy Etot satisfies a variational theorem at fixed occupations, which we dealt with so far. However, now the occupations equilibrate at a specified temperature T, and the Helmholtz free energy F = E - TS, where S is the electronic entropy, is the appropriate variational free energy to work with. Note the corresponding changes in the energy component printout at the end as well.

Additionally, in the initialization, note that nBands is now larger than nElectrons/2, since the code needs a few empty states to use Fermi fillings. Examine the fillings in the HDF5 checkpoint file using :code:`h5dump -g /electrons/fillings Pt_out.h5`

TODO: Explain what these numbers mean and show the bandstructure calcultion
