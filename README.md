![QimPy](docs/qimpy.svg)

---

QimPy (pronounced [/'kɪm-paɪ'/](https://en.wikipedia.org/wiki/Help:IPA/English))
is a Python package for Quantum-Integrated Multi-PhYsics.

# Coding style

All Python code must be PEP-8 compliant.
The repository provides a .editorconfig with indentation and line-length rules,
and a pre-commit configuration to run flake8.
Please install this pre-commit hook by running `pre-commit install`
within the working directory.

Function/method signatures and class attributes must use type hints.
Document class attributes using doc comments on the type hints when possible.

For all log messages, use f-strings as far as possible for maximum readability.
