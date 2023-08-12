Development
===========

To get started with QimPy development, fetch QimPy from git
and install it in develop mode as discussed in :doc:`/install`.
QimPy maintains a consistent object heirarchy in the API,
input and checkpoint files, so once you get familiar with
where to specify a setting in the input, you now also know
where to find the corresponding outputs in the HDF5 checkpoint
as well as the underlying source code related to that setting.

In fact, the :doc:`/inputfile` and :doc:`/api` are both generated
from the same documentation strings within the code.
The main objects in the object heirarchy for QimPy all derive from
:class:`qimpy.TreeNode`, which sets up a consistent tree structure for
the objects in memory, the checkpoint and the YAML input file.
See the particularly detailed doc strings for the `__init__` of
any such class, *e.g.*, starting with :class:`qimpy.dft.System`,
the root object created for DFT calculations.
The parameters whose documentation contain a `:yaml:` tag
are those that can be specified from the input file,
while the rest are used internally in the code alone.

To get started with QimPy development, a great place to start is the
`QimPy issues <https://github.com/shankar1729/qimpy/issues>`_ page.
In particular, look for issues labeled with `good first issue`.
We of course greatly appreciate any and all feature contributions.
If you like using the code but are not yet comfortable modifying it,
expansion of the tutorials and improvement of the documentation
will also be invaluable contributions!


Coding style
------------

The repository provides a .editorconfig with indentation and line-length rules,
and a pre-commit configuration to run black and flake8 to enforce and verify style.
Please install this pre-commit hook by running `pre-commit install`
within the working directory.
While this hook will run automatically on filed modified in each commit,
you can also use `make precommit` to manually run it on all code files.

Function/method signatures and class attributes must use type hints.
Document class attributes using doc comments on the type hints when possible.
Run `make typecheck` to perform a static type check using mypy before pushing code.

For all log messages, use f-strings as far as possible for maximum readability.

Run `make test` to invoke all configured pytest tests. To only run mpi or
non-mpi tests specifically, use `make test-mpi` or `make test-nompi`.

Best practice: run `make check` to invoke the precommit, typecheck
and test targets before commiting code.
