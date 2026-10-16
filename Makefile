.PHONY: precommit
precommit:
	pre-commit run --all-files

.PHONY: typecheck
typecheck:
	cd src && mypy -p qimpy

.PHONY: test-nompi
test-nompi:
	python -m pytest

.PHONY: test-mpi
test-mpi:
	mpirun ./mpi_print_from_head.sh python -m pytest --with-mpi

.PHONY: test
test: test-nompi test-mpi

# The brute-force cross-checks against the reduction-free collision integral.
# Deselected from `make test` (see the `validate` marker in pyproject.toml)
# because they cost minutes each and want a GPU; run them before a merge, on a
# release branch, or after touching anything under material/fermi_surface.
.PHONY: test-validate
test-validate:
	python -m pytest -m validate

.PHONY: check
check: precommit typecheck test
