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

.PHONY: check
check: precommit typecheck test
