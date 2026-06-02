# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

project = 'GO-Diff'
copyright = '2025, Nikolaj Rønne'
author = 'Nikolaj Rønne'

_version_file = os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION')
try:
    with open(_version_file) as _f:
        release = _f.read().strip()
except OSError as e:
    raise RuntimeError(f"Could not read VERSION file at {_version_file}: {e}") from e

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoapi_dirs = ['../../go_diff']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
