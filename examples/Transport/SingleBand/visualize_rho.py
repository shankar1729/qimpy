import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="HDF5 file to read data from")
parser.add_argument("quad", type=int, help="Quad number to plot")
parser.add_argument("x", type=int, help="Grid-point x index to plot")
parser.add_argument("y", type=int, help="Grid-point y index to plot")
parser.add_argument("--diff", type=str, nargs=1, help="Filename to difference against")
args = parser.parse_args()

with h5py.File(args.filename) as fp:
    k = np.array(fp["/material/k"])
    Rbasis = np.array(fp["/material/lattice"].attrs["Rbasis"])
    Gbasis = np.linalg.inv(Rbasis.T) * (2 * np.pi)
    kx, ky, _ = Gbasis @ k.T
    rho = np.array(fp[f"/geometry/quad{args.quad}/rho"][args.x, args.y])

if args.diff:
    with h5py.File(args.diff[0]) as fp:
        rho -= np.array(fp[f"/geometry/quad{args.quad}/rho"][args.x, args.y])

plt.scatter(kx, ky, c=rho, cmap="RdBu")
plt.gca().set_aspect("equal")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.colorbar(label=r"$\rho$")
plt.show()
