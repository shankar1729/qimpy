# Calculation analogous to NaCl-ClVac.py with Python input:
import qimpy as qp
qp.log_config()  # default set up to log from MPI head alone
qp.log.info('Using QimPy '+qp.__version__)

# Create lattice object explicitly (eg. shared between two systems)
lattice = qp.Lattice(
    vector1=[0, 0.5, 0.5],  # Note: vectors in rows like other codes.
    vector2=[0.5, 0, 0.5],  # Also, not ambiguous as separate vectors,
    vector3=[0.5, 0.5, 0],  # compared to the 3x3 matrix of JDFTx.
    scale=21.48)  # bohrs, for 2x2x2 supercell (could be length-3 list/tuple)

# To demo, create ions using dict (as it would have been from YAML)
ion_params = {
  'pseudopotentials': [
     '../../../JDFTx/build_testing/pseudopotentials/SG15/Cl_ONCV_PBE-1.0.upf',
     '../../../JDFTx/build_testing/pseudopotentials/SG15/$ID_ONCV_PBE.upf'],
  'coordinates': [
     ['Na', 0.25, 0.25, 0.25],
     ['Na', 0.25, 0.25, 0.75],
     ['Na', 0.25, 0.75, 0.25],
     ['Na', 0.25, 0.75, 0.75],
     ['Na', 0.75, 0.25, 0.25],
     ['Na', 0.75, 0.25, 0.75],
     ['Na', 0.75, 0.75, 0.25],
     ['Na', 0.75, 0.75, 0.75],
     ['Cl', 0.00, 0.50, 0.50],
     ['Cl', 0.50, 0.00, 0.50],
     ['Cl', 0.50, 0.50, 0.00],
     ['Cl', 0.50, 0.00, 0.00],
     ['Cl', 0.00, 0.50, 0.00],
     ['Cl', 0.00, 0.00, 0.50],
     ['Cl', 0.50, 0.50, 0.50]]}

system = qp.System(
    lattice=lattice,
    ions=ion_params)
