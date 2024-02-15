import struct
import argparse

import h5py
import numpy as np
from tqdm import tqdm

from qimpy.io import log_config
from qimpy.profiler import stopwatch, StopWatch


def read_double(file, count=1):
    return struct.unpack(f"{count}d", file.read(8 * count))


def read_size_t(file, count=1):
    return struct.unpack(f"{count}N", file.read(8 * count))


def read_int(file, count=1):
    return struct.unpack(f"{count}i", file.read(4 * count))


def read_byte(file, count=1):
    return struct.unpack(f"{count}b", file.read(count))


def read_complex_array(file, dims):
    """Read numpy array of complex numbers with specified dimensions."""
    data = np.array(read_double(file, np.prod(dims) * 2))
    return data.reshape(-1, 2).view(np.complex128).reshape(dims)


def read_header(file) -> dict:
    assert file.read(4) == b"LDBD"
    # mu, T and pump/probe frequency range accounted for
    dmuMin, dmuMax, Tmax, pumpOmegaMax, probeOmegaMax = read_double(file, 5)
    # number of selected k-points and original total k-points
    nk, nkTot = read_size_t(file, 2)
    # e-ph and spinorial info available
    ePhEnabled, spinorial = read_byte(file, 2)
    spinWeightFlags = read_int(file)[0]
    spinWeight = spinWeightFlags & 3  # degeneracy for spin
    haveL = bool(
        int(format(spinWeightFlags & 4, "04b")[1])
    )  # whether angular momentum is included in data
    R = np.array(read_double(file, 9)).reshape((3, 3))  # lattice vectors
    return {
        "dmuMin": dmuMin,
        "dmuMax": dmuMax,
        "Tmax": Tmax,
        "pumpOmegaMax": pumpOmegaMax,
        "probeOmegaMax": probeOmegaMax,
        "nk": nk,
        "nkTot": nkTot,
        "ePhEnabled": bool(ePhEnabled),
        "spinorial": bool(spinorial),
        "spinWeight": spinWeight,
        "haveL": haveL,
        "R": R,
    }


@stopwatch
def read_k_point(file, header, ik, k, E, P, S, L, G, omega_ph, ikpair):
    assert file.read(4) == b"\nKPT"
    k[ik] = read_double(file, 3)  # k-point in reciprocal lattice coordinates
    nInner, nOuter, innerStart = read_int(file, 3)
    E[ik] = read_double(file, nOuter)  # energies: diagonal matrix
    P[ik] = read_complex_array(file, (3, nInner, nOuter)).swapaxes(-2, -1)
    if header["spinorial"]:
        S[ik] = read_complex_array(file, (3, nInner, nInner)).swapaxes(-2, -1)
    if header["haveL"]:
        L[ik] = read_complex_array(file, (3, nInner, nInner)).swapaxes(-2, -1)
    if header["ePhEnabled"]:
        Gk_size = read_size_t(file)[0]
        Gi = []
        omega_ph_i = []
        ikpair_i = []
        for i in range(Gk_size):
            read_eph(file, ik, Gi, omega_ph_i, ikpair_i, nInner)
        G.append(np.stack(Gi))
        omega_ph.append(np.array(omega_ph_i))
        ikpair.append(np.array(ikpair_i))


def read_eph(file, ik, G, omega_ph, ikpair, nInner):
    assert file.read(4) == b"GEPH"
    jk = read_size_t(file)[0]  # index of second k-point
    ikpair.append((ik, jk))  # index pair of k-points for following properties
    omega_ph.append(read_double(file)[0])  # phonon frequency
    Gn_size = read_size_t(file)[0]  # number of band pairs at this k-pair
    G_cur = np.zeros((nInner, nInner), dtype=complex)
    for i1, i2, Gre, Gim in struct.iter_unpack("2i2d", file.read(Gn_size * 24)):
        G_cur[i1, i2] = complex(Gre, Gim)
    G.append(G_cur)  # dense e-ph matrix for k-pair (but sparse in ik, jk)


def read_ldbd(ldbd_file, n_bands, h5_file=None):
    with open(ldbd_file, mode="rb") as file:
        header = read_header(file)
        nk = header["nk"]
        _ = header["Tmax"]  # Temp: not used
        _ = read_size_t(file, nk)  # byte_offsets: not used anymore
        # Intializing:
        k = np.zeros((nk, 3))
        E = np.zeros((nk, n_bands))
        P = np.zeros((nk, 3, n_bands, n_bands), dtype=complex)
        S = np.zeros((nk, 3, n_bands, n_bands), dtype=complex)
        L = np.zeros((nk, 3, n_bands, n_bands), dtype=complex)
        ikpair = []
        omega_ph = []
        G = []
        for ik in tqdm(range(nk), f"Reading {ldbd_file}"):
            read_k_point(file, header, ik, k, E, P, S, L, G, omega_ph, ikpair)

    G = np.concatenate(G, axis=0)
    omega_ph = np.concatenate(omega_ph, axis=0)
    ikpair = np.concatenate(ikpair, axis=0)

    if h5_file is not None:
        write_checkpoint(h5_file, header, k, E, P, S, L, G, omega_ph, ikpair)


@stopwatch
def write_checkpoint(h5_file, header, k, E, P, S, L, G, omega_ph, ikpair):
    with h5py.File(h5_file, "w") as fp:
        for key, value in header.items():
            fp.attrs[key] = value
        fp["k"] = k
        fp["E"] = E
        fp["P"] = P
        fp["S"] = S
        fp["L"] = L
        fp["G"] = G
        fp["omega_ph"] = omega_ph
        fp["ikpair"] = ikpair


def main():
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.transport.material.read",
        description="Convert data file from FeynWann to HDF5 checkpoint file",
    )
    parser.add_argument(
        "-d",
        "--data-file",
        type=str,
        help="data file from FeynWann",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint-file",
        default="checkpoint.h5",
        type=str,
        help="checkpoint file in HDF5 format (checkpoint.h5 if unspecified)",
    )
    parser.add_argument(
        "-n",
        "--n_bands",
        help="number of bands for each kpoint in FeynWann data file",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    log_config()
    ldbd_file = args.data_file
    h5_file = args.checkpoint_file
    n_bands = args.n_bands
    read_ldbd(ldbd_file, n_bands, h5_file)
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
