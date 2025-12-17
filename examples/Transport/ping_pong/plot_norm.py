import glob

import h5py
import numpy as np
from matplotlib import pyplot as plt


t = []
N = []  # total count
for filename in sorted(glob.glob("animation/*.h5")):
    with h5py.File(filename) as fp:
        g = np.array(fp["/geometry/quad0/g"])
        n = np.array(fp["/geometry/quad0/observables"][..., 0])
        N.append((g * n).sum(axis=(-2, -1)))
        t.append(np.array(fp["/geometry/t"]))
N = np.concatenate(N)
t = np.concatenate(t)

plt.plot(t, N)
plt.xlabel("Time")
plt.ylabel("Norm")
plt.savefig("norm_vs_t.pdf")
plt.show()
