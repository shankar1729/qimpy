"""Unit-test the divergence dump WITHOUT qimpy, a GPU, or Jetstream.

Same trick as outputs/test_kick_logic.py: ast-extract the real functions from
the repo file and drive them with a stub `collision_dot`, so the code under
test is the exact text that will be deployed -- not a paraphrase.
"""
import ast
import os
import sys
import tempfile

import numpy as np
import torch
import h5py

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_time_evolution.py")
tree = ast.parse(open(SRC).read())

WANT_FN = {"_amax", "_patches", "_clone", "_amax_where"}
WANT_METH = {"_collision_kick", "_check_collision_rho", "_dump_collision_state"}


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))


log = _Log()
class Geometry: pass
class TensorList(list): pass
ns = {"torch": torch, "np": np, "h5py": h5py, "os": os, "log": log,
      "Geometry": Geometry, "TensorList": TensorList}
exec("from __future__ import annotations", ns)

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FN:
        exec(compile(ast.Module([node], []), SRC, "exec"), ns)
    if isinstance(node, ast.ClassDef):
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name in WANT_METH:
                exec(compile(ast.Module([sub], []), SRC, "exec"), ns)
            if isinstance(sub, ast.Assign):  # _MAX_SUBSTEPS etc.
                for t in sub.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        exec(compile(ast.Module([sub], []), SRC, "exec"), ns)

missing = (WANT_FN | WANT_METH) - set(ns)
assert not missing, f"failed to extract {missing}"
print(f"extracted {sorted(WANT_FN | WANT_METH)}")


class Geom:
    """Stub geometry: rho plus a collision operator we choose."""

    def __init__(self, rho, dot):
        self.rho = rho
        self._dot = dot
        self.n_applies = 0

    def collision_dot(self, rho):
        self.n_applies += 1
        return self._dot(rho)


class TE:
    """Stub TimeEvolution carrying only what _collision_kick touches."""
    _RHO_ATOL = ns["_RHO_ATOL"]
    _MAX_SUBSTEPS = ns["_MAX_SUBSTEPS"]
    _WARN_SUBSTEPS = ns["_WARN_SUBSTEPS"]
    _LOUD_SUBSTEPS = ns["_LOUD_SUBSTEPS"]
    _collision_kick = ns["_collision_kick"]
    _check_collision_rho = ns["_check_collision_rho"]
    _dump_collision_state = ns["_dump_collision_state"]

    def __init__(self, s_max=0.02, rho_max=1.0):
        self.collision_s_max = s_max
        self.collision_rho_max = rho_max
        self.collision_interval = 40
        self.collision_fuse = False
        self.i_step = 182519
        self.t = 1.113e8
        self.dt = 610.0


fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp()
os.environ["QIMPY_COLLISION_DUMP"] = os.path.join(tmp, "dump")

# ---------------------------------------------------------------- 1. bit-identity
print("\n[1] s_max = 0 stays bit-identical (2 applies, no clone, no dump)")
rho0 = torch.linspace(-0.3, 0.3, 24, dtype=torch.float64).reshape(4, 6)
gamma = 3.0e-7


def lin(r):
    return -gamma * r


te = TE(s_max=0.0)
g = Geom(rho0.clone(), lin)
te._collision_kick(g, 2.44e4)
dt = 2.44e4
ref = rho0 + dt * lin(rho0 + 0.5 * dt * lin(rho0))
check("2 applies", g.n_applies == 2)
check("bit-identical to plain midpoint", torch.equal(g.rho, ref))
check("no dump written", not os.listdir(tmp))

# ---------------------------------------------------------------- 2. linear h
print("\n[2] linear operator: h = 2*s_max/gamma exactly, decays like exp(-gamma t)")
te = TE(s_max=0.02)
g = Geom(rho0.clone(), lin)
te._collision_kick(g, 2.44e4)
exact = rho0 * np.exp(-gamma * 2.44e4)
check("converged to exp(-gamma t)", torch.allclose(g.rho, exact, rtol=1e-6))
check("substep log mentions max rate", any("max rate" in m for _, m in log.lines))

# ---------------------------------------------------------------- 3. rho_max in loop
print("\n[3] rho_max trips INSIDE the loop and dumps (not after ~4096 substeps)")
os.environ["QIMPY_COLLISION_DUMP"] = os.path.join(tmp, "d3")


def grow(r):  # strongly amplifying, drives |rho| through 1
    return 4.0e-4 * r * (1.0 + 50.0 * r * r)


te = TE(s_max=0.02, rho_max=1.0)
g = Geom(torch.full((3, 3), 0.9, dtype=torch.float64), grow)
raised = None
try:
    te._collision_kick(g, 2.44e4)
except RuntimeError as e:
    raised = str(e)
check("raised", raised is not None and "max|rho|" in raised)
check("raised at a substep, not after the cap", "at substep" in (raised or ""))
d3 = sorted(f for f in os.listdir(tmp) if f.startswith("d3"))
check("dump written", len(d3) == 1)
if d3:
    with h5py.File(os.path.join(tmp, d3[0]), "r") as fp:
        check("reason=rho_max", fp.attrs["reason"] == "rho_max")
        check("has rho_entry", "rho_entry" in fp)
        check("has rho", "rho" in fp)
        check("has k1", "k1" in fp)
        check("has substep_history", "substep_history" in fp)
        check("i_step recorded", int(fp.attrs["i_step"]) == 182519)
        ent = fp["rho_entry/patch0"][()]
        check("rho_entry is the ENTRY state (0.9), not the blown-up one",
              np.allclose(ent, 0.9))
        check("rho at failure exceeds rho_max",
              float(fp["rho"].attrs["amax"]) > 1.0)
        hist = fp["substep_history"][()]
        check("history has 5 columns", hist.shape[1] == 5)
        check("history rate column rises",
              hist[-1, 3] >= hist[0, 3])
        check("amax_index recorded", len(fp["rho"].attrs["amax_index"]) == 2)

# ---------------------------------------------------------------- 4. substep cap
print("\n[4] substep cap still fires (rate that cannot be outrun) and dumps")
os.environ["QIMPY_COLLISION_DUMP"] = os.path.join(tmp, "d4")


def stiff(r):  # huge rate, tiny amplitude change -> h collapses, |rho| stays small
    return -1.0e3 * torch.sign(r) * torch.ones_like(r) * 1e-9


te = TE(s_max=0.02, rho_max=1.0)
g = Geom(torch.full((2, 2), 1e-9, dtype=torch.float64), stiff)
raised = None
try:
    te._collision_kick(g, 1e12)
except RuntimeError as e:
    raised = str(e)
check("raised on cap", raised is not None and "substeps after" in (raised or ""))
d4 = sorted(f for f in os.listdir(tmp) if f.startswith("d4"))
# The warn dump fires first (n_sub == _WARN_SUBSTEPS) and sorts BEFORE the cap
# dump alphabetically, so select by name rather than by position.
check("both warn and cap dumps written", len(d4) == 2)
check("warn dump present", any("_warn" in f for f in d4))
d4cap = [f for f in d4 if "max_substeps" in f]
check("cap dump present", len(d4cap) == 1)
if d4cap:
    with h5py.File(os.path.join(tmp, d4cap[0]), "r") as fp:
        check("reason=max_substeps", fp.attrs["reason"] == "max_substeps")
        check("n_sub at cap", int(fp.attrs["n_sub"]) > TE._MAX_SUBSTEPS)
        check("history capped in length",
              fp["substep_history"].shape[0] == int(fp.attrs["n_sub"]))

# ---------------------------------------------------------------- 5. TensorList
print("\n[5] multi-patch TensorList path")
os.environ["QIMPY_COLLISION_DUMP"] = os.path.join(tmp, "d5")


class TL(list):
    def __add__(self, o):
        return TL(a + b for a, b in zip(self, o))

    def __radd__(self, o):
        return self.__add__(o)

    def __rmul__(self, c):
        return TL(c * a for a in self)


te = TE(s_max=0.02, rho_max=1.0)
start = TL([torch.full((2, 2), 0.5, dtype=torch.float64),
            torch.full((3,), 0.95, dtype=torch.float64)])
g = Geom(start, lambda r: TL(grow(x) for x in r))
raised = None
try:
    te._collision_kick(g, 2.44e4)
except RuntimeError as e:
    raised = str(e)
check("raised on TensorList", raised is not None)
d5 = sorted(f for f in os.listdir(tmp) if f.startswith("d5"))
if d5:
    with h5py.File(os.path.join(tmp, d5[0]), "r") as fp:
        check("both patches dumped",
              "rho_entry/patch0" in fp and "rho_entry/patch1" in fp)
        check("amax located in patch 1 (the 0.95 one)",
              int(fp["rho_entry"].attrs["amax_patch"]) == 1)

# ---------------------------------------------------------------- 6. dump never masks
print("\n[6] an unwritable dump path must NOT mask the RuntimeError")
os.environ["QIMPY_COLLISION_DUMP"] = "/nonexistent-dir-xyz/dump"
te = TE(s_max=0.02, rho_max=1.0)
g = Geom(torch.full((3, 3), 0.9, dtype=torch.float64), grow)
raised = None
try:
    te._collision_kick(g, 2.44e4)
except RuntimeError as e:
    raised = str(e)
check("still raised the physics error", raised is not None and "max|rho|" in raised)
check("warned about the failed write",
      any(lvl == "warning" and "FAILED to write" in m for lvl, m in log.lines))

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
