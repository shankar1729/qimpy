# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.append(os.path.abspath("../../src/"))
sys.path.append(os.path.abspath("./_ext/"))


# -- Project information -----------------------------------------------------

project = "QimPy"
copyright = "2021, QimPy Collaboration"
author = "QimPy Collaboration"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.coverage",  # Report missing documentation
    "sphinx.ext.napoleon",  # NumPy style docstrings
    "yamldoc",  # Extract YAML input file documentation from docstrings
]
autosummary_generate = True
autosummary_imported_members = True
coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Move type hints from call signature to description
autodoc_typehints = "description"

# Order entries by type:
autodoc_member_order = "groupwise"

# Suppress unnecessary paths in class / function names:
add_module_names = False
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {"style_nav_header_background": "#000000"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "../qimpy.svg"
html_favicon = "../qimpy.ico"
