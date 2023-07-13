"""Math functions extending the core torch set."""
# List exported symbols for doc generation
__all__ = (
    "prime_factorization",
    "fft_suitable",
    "ceildiv",
    "cis",
    "abs_squared",
    "dagger",
    "accum_norm_",
    "accum_prod_",
    "ortho_matrix",
    "eighg",
)

from .integer import prime_factorization, fft_suitable, ceildiv
from .linalg import (
    cis,
    abs_squared,
    dagger,
    accum_norm_,
    accum_prod_,
    ortho_matrix,
    eighg,
)
