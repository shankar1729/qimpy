from qimpy import log
from qimpy.io import CheckpointPath
from .. import Wavefunction


class PlusU:
    """DFT+U correction."""

    u_parameters: list[dict]
    required_fields = ["species", "orbital", "UminusJ"]
    allowed_fields = required_fields + ["Vext"]

    def validate(self, paramsU):
        """Ensure that given DFT+U settings are valid"""
        if paramsU is not None:
            for paramU in paramsU:
                for field in self.required_fields:
                    if field not in paramU.keys():
                        raise ValueError(f"Missing required field for plus-U: {field}")
                for key, value in paramU.items():
                    if key not in self.allowed_fields:
                        raise ValueError(f"Invalid field for plus-U: {key}")

    def __init__(
        self, plusU, *, checkpoint_in: CheckpointPath = CheckpointPath()
    ) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        plusU
            Dictionary of U values by species and orbital names.
        """
        super().__init__()
        self.validate(plusU)
        self.u_parameters = plusU

    def __bool__(self) -> bool:
        return bool(self.u_parameters)

    def __call__(self, C: Wavefunction) -> Wavefunction:
        """TODO."""
        raise NotImplementedError
