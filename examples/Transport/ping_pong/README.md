(1) Check geometry:
`python -m qimpy.transport.geometry.test_svg rect-domain.svg 1 --grid_spacing 0.01`

(2) Run:
`time mpirun -np 8 python -m qimpy.transport.geometry.test_advect --h 0.005 --Ntheta 128 --specularity 0 --sigma 0.1 --q0 0.5 0.5 --v0 0 1 --dt_save 0.01 --t_max 10 --svg rect-domain.svg`

(3) Plot:
`time mpirun -np 8 python -m qimpy.transport.plot plot.yaml`

---

For testing norm conservation upon reflection, compare the results of `python3 plot_norm.py` after the following runs:

```
mpirun python -m qimpy.transport.geometry.test_advect --h 0.005 --Ntheta 4 --sigma 0.1 --q0 0.3 0.7 --specularity 1 --v0 1.000 1.000 --dt_save 0.01 --t_max 2 --svg rect-domain.svg
mpirun python -m qimpy.transport.geometry.test_advect --h 0.005 --Ntheta 4 --sigma 0.1 --q0 0.3 0.7 --specularity 1 --v0 1.002 0.998 --dt_save 0.01 --t_max 2 --svg rect-domain.svg
```

---

Test using periodic curved tile, along with check of more complex data exchanges at pass-through boundaries:

```
export QIMPY_CYCLE_PERMUTE
mpirun python -m qimpy.transport.geometry.test_advect --h 0.25 --Ntheta 4 --sigma 5 --q0 40 40 --specularity 1 --v0 0.998 1.002 --dt_save 1 --t_max 100 --svg ../curved-tile-split.svg
```
