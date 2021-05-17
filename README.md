![QimPy](docs/qimpy.svg)

---

QimPy (pronounced [/'kɪm-paɪ'/](https://en.wikipedia.org/wiki/Help:IPA/English))
is a Python package for Quantum-Integrated Multi-PhYsics.

# Coding style

All Python code must be PEP-8 compliant.
The repository provides a .editorconfig with indentation and line-length rules,
and a pre-commit hook within .githooks to run pycodestyle.
Please install this pre-commit hook by creating a link to it within .git/hooks,
and make sure you have pycodestyle installed.

Function/method signatures and class attributes must use type hints.
Document class attributes using doct comments on the type hints when possible.

For all log messages, use f-strings as far as possible for maximum readability.
