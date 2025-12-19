import glob

import h5py
import numpy as np
from matplotlib import pyplot as plt


t = []
N = []  # total count
for filename in sorted(glob.glob("animation/*.h5")):
    with h5py.File(filename) as fp:
        n_quads = len(fp["/geometry/quads"])
        N_cur = []
        for i_quad in range(n_quads):
            g = np.array(fp[f"/geometry/quad{i_quad}/g"])
            n = np.array(fp[f"/geometry/quad{i_quad}/observables"][..., 0])
            N_cur.append((g * n).sum(axis=(-2, -1)))
        N.append(sum(N_cur))
        t.append(np.array(fp["/geometry/t"]))
N = np.concatenate(N)
t = np.concatenate(t)

plt.plot(t, N)
plt.xlabel("Time")
plt.ylabel("Norm")
plt.savefig("norm_vs_t.pdf")
plt.show()
