Spin-orbit Coupling
==================

Most density-functional theory calculations are based in non-relativistic
quantum mechanics, where spin decouples completely from the spatial degrees
of freedom and only enter in the orbital occupations.
Far from the atoms, the electron velocities are indeed much smaller than the
speed of light and the non-relativistic approximation is valid.
This is however no longer true close to the nuclei for electrons
with non-zero angular momentum.
Importantly, even valence electrons (with l > 0) in heavy atoms
pick up relativsitic effects - especially spin-orbit coupling -
from the regions of space close to the nuclei.
This tutorial demonstrates the effect of spin-orbit coupling
on the band structure of metallic platinum.

In plane-wave calculations using pseudopotentials,
these relativistic effects can be built into the pseudopotentials.
The GBRV pseudopotentials we have used so far do not have relativistic versions available,
so we will use another pseudopotential from the 
`Quantum Espresso pseudopotential library <http://www.quantum-espresso.org/pseudopotentials>`_.
Download `this relativistic pseudopotential <http://www.quantum-espresso.org/wp-content/uploads/upf_files/Pt.rel-pbe-n-rrkjus_psl.0.1.UPF>`_,
and also `this conventional non-relativistic pseudopotential <http://www.quantum-espresso.org/wp-content/uploads/upf_files/Pt.pbe-n-rrkjus_psl.0.1.UPF>`_,
into your current working directory.
We will use a different non-relativistic pseudopotential as well to compare with
and identify the relativistic effects, and for this we will use the downloaded one
instead of GBRV because its generation parameters are much closer to the relativistic one
and we will see less discrepancies simply due to pseudopotential differences.

You should now have two UPF pseudopotential files in your current directory:

.. code-block:: yaml
      
      Pt.rel-pbe-n-rrkjus_psl.0.1.UPF   #the relativistic one
      Pt.pbe-n-rrkjus_psl.0.1.UPF       #the non-relativistic one

We first run a non-relativistic calculation (but with the new pseudopotential)


.. code-block:: yaml

    lattice:
      system: cubic
      modification: face-centered
      a: 7.41

    ions:
      pseudopotentials:
        - $ID.pbe-n-rrkjus_psl.0.1.UPF
      coordinates:
        - [Pt, 0, 0, 0]

    electrons:
      spinorial: no
      fillings:
         smearing: Gauss
         sigma: 0.02
      k-mesh:
        size: [12, 12, 12]

      xc:
        functional: gga_pbe

checkpoint_out: Pt_out.h5
