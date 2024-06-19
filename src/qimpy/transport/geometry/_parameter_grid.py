from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from qimpy import rc, log, io
from qimpy.io import (
    CheckpointPath,
    InvalidInputException,
    TensorCompatible,
    cast_tensor,
    CheckpointContext,
)
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import Material
from . import TensorList, Geometry, QuadSet


class ParameterGrid(Geometry):
    """Geometry specification."""

    shape: tuple[int, int]  #: Dimensions of parameter grid
    parameters: dict[str, torch.Tensor]  #: Parameters broadcastable with grid

    def __init__(
        self,
        *,
        material: Material,
        shape: tuple[int, int],
        dimension1: Optional[dict[str, dict[str, TensorCompatible]]] = None,
        dimension2: Optional[dict[str, dict[str, TensorCompatible]]] = None,
        save_rho: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize parameter grid parameters.

        Parameters
        ----------
        shape
            :yaml:`Dimensions of parameter grid (always 2D).`
        dimension1
            :yaml:`Parameter names and values to sweep along dimension 1.`
            The values can be specified with "loop" over explicit values,
            or "sweep" linearly from the initial to the final value.
        dimension2
            :yaml:`Parameter names and values to sweep along dimension 2.`
            Specification is the same as for `dimension1`.
        save_rho
            :yaml:`Whether to write the full density matrices to the checkpoint file.`
            If not (default), only observables are written to the checkpoint file.
        """
        assert len(shape) == 2
        self.shape = tuple(shape)

        # Create fake geometry for parameter grid
        quad_set = QuadSet(
            vertices=_GRID_VERTICES * np.array(shape),
            quads=_GRID_QUADS,
            adjacency=np.full((1, 4, 2), -1),
            displacements=np.zeros((1, 4, 2)),
            grid_size=np.array([shape]),
            contacts=np.zeros((0, 3)),
            apertures=np.zeros((0, 3)),
            aperture_names=[],
            has_apertures=np.full((1, 4), False),
        )
        super().__init__(
            material=material,
            process_grid=process_grid,
            quad_set=quad_set,
            grid_spacing=1,
            grid_size_max=0,
            contacts={},
            save_rho=save_rho,
            checkpoint_in=checkpoint_in,
        )
        self.dt_max = 0  # disable transport dt limit

        # Prepare all parameter values:
        self.parameters: dict[str, torch.Tensor] = {}
        if checkpoint_in:
            cp_parameters = checkpoint_in.relative("parameters")
            cp, path = cp_parameters
            if path in cp:
                for name in cp[path].keys():
                    self.parameters[name] = cp_parameters.read(name)
        else:
            for i_dim, dimension in enumerate([dimension1, dimension2]):
                if dimension is not None:
                    for key, values in io.dict.key_cleanup(dimension).items():
                        self.parameters[key] = self.create_values(i_dim, **values)
        log.info(f"Initialized parameter grid of dimensions: {shape}")

        # Initialize material for parameter subsets in each patch:
        patches_mine = slice(self.patch_division.i_start, self.patch_division.i_stop)
        for patch, grid_start, grid_stop in zip(
            self.patches,
            self.sub_quad_set.grid_start[patches_mine],
            self.sub_quad_set.grid_stop[patches_mine],
        ):
            # Initialize parameter subsets:
            slice0 = slice(grid_start[0], grid_stop[0])
            slice1 = slice(grid_start[1], grid_stop[1])
            parameters_sub = {
                key: values[
                    slice(None) if (values.shape[0] == 1) else slice0,
                    slice(None) if (values.shape[1] == 1) else slice1,
                ]
                for key, values in self.parameters.items()
            }
            rho_initial = patch.rho.clone().detach() if checkpoint_in else patch.rho
            material.initialize_fields(rho_initial, parameters_sub, id(patch))

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["shape"] = self.shape
        attrs["save_rho"] = self.save_rho
        for name, parameter in self.parameters.items():
            cp_path.write(f"parameters/{name}", parameter)
        saved_list = [*attrs.keys(), "parameters"]
        return saved_list + super()._save_checkpoint(cp_path, context)

    def create_values(
        self,
        i_dim: int,
        *,
        loop: Optional[list] = None,
        sweep: Optional[list] = None,
    ) -> torch.Tensor:
        if (loop is None) == (sweep is None):
            raise InvalidInputException(
                "Exactly one of loop or sweep should be specified."
            )

        # Prepare scan of values based on loop or sweep:
        values: Optional[torch.Tensor] = None
        if loop is not None:
            values = cast_tensor(loop)
            if len(values) != self.shape[i_dim]:
                raise InvalidInputException(
                    f"Number of entries in loop = {len(values)} should match"
                    f" length {self.shape[i_dim]} of dimension {i_dim + 1}"
                )
        if sweep is not None:
            limits = cast_tensor(sweep)
            if len(limits) != 2:
                raise InvalidInputException(
                    f"Number of entries in sweep = {len(limits)} must be 2"
                )
            t = torch.linspace(0, 1, self.shape[i_dim], device=rc.device)
            values = limits[0] + torch.einsum("t,...->t...", t, limits[1] - limits[0])

        # Reshape to broadcast along appropriate dimension:
        assert values is not None
        if i_dim == 0:
            return values[:, None]
        else:  # i_dim == 1
            return values[None]

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        return TensorList(
            self.material.rho_dot(rho_i, t, id(patch))
            for rho_i, patch in zip(rho, self.patches)
        )


_GRID_VERTICES = (1.0 / 3) * np.array(
    [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 3],
        [3, 2],
        [3, 1],
        [3, 0],
        [2, 0],
        [1, 0],
    ]
)
_GRID_QUADS = np.array([[[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9, 10, 11, 0]]])
