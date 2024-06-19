Open-shell systems
==================

This tutorial covers the basics for spin-polarized and open-shell calculations in QimPy,
using the most basic system of all, a hydrogen atom.

First, lets set up a hydrogen atom calculation exactly as we would based on the previous tutorials.
Save the following to `Hatom.yaml`:

.. code-block:: yaml

    lattice:
      system: cubic
      modification: face-centered
      a: 20.  # bohrs

    ions:
      pseudopotentials:
        - SG15/$ID_ONCV_PBE.upf
      coordinates:
        - [H, 0., 0., 0.]

    electrons:
      basis:
        ke-cutoff: 30.0

    checkpoint: null  # disable reading checkpoint
    checkpoint-out: Hatom.h5  # but still create it

and run

.. code-block:: bash

    (qimpy) $ python -m qimpy.dft -i Hatom.yaml | tee Hatom.out

Since there is only one atom, we don't need geometry optimization.
Notice that the final energy F = -0.4601 Hartrees, which is rather different
from the analytical exact energy -0.5 Hartree (= -1 Rydberg = -13.6 eV).

The reason for this disrepancy is that, by default, this DFT calculation is spin-unpolarized,
that is it assumes an equal number of up and down spin electrons.
This assumption is correct for the water molecule with a closed shell of 8 valence electrons
that we dealt with so far, but is incorrect for the hydrogen atom which has only one electron.
This electron must be either an up or down spin, so that the magnetization (Nup - Ndn) is +1 or -1.
We can invoke a spin-polarized calculation and specify the magnetization by adding the following
key-value pairs to `Hatom.yaml` and rerun QimPy:

.. code-block:: yaml

    electrons:
        basis:
            ke-cutoff: 30.0
        spin-polarized: yes
        fillings:
            M: 1

Now we find F = -0.4997 Hartrees, in much better agreement with the analytical result.
Check that using magnetization -1 produces exactly the same result.
