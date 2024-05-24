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
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../api'))
sys.path.insert(0, os.path.abspath('../autocomplete/app'))
sys.path.insert(0, os.path.abspath('../chatbot'))
sys.path.insert(0, os.path.abspath('../indexing/app'))
sys.path.insert(0, os.path.abspath('../rag/app'))
# sys.path.insert(0, os.path.abspath('../other/app/path'))
# missing survey pipeline and gui

# -- Project information -----------------------------------------------------

project = 'EAK-Copilot'
copyright = '2024, CdC-SI'
author = 'CdC-SI'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    "sphinx_copybutton",
    "myst_parser",          # .md support
    "sphinx_inline_tabs",
    'sphinx.ext.napoleon',  # NumPy & Google docstrings
]
autodoc_mock_imports = ["fastapi"]
todo_include_todos = True
myst_enable_extensions = ["deflist"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_theme_options = {
    "source_repository": "https://github.com/CdC-SI/eak-copilot",
    "source_branch": "main/",
    "source_directory": "doc/",
    "light_css_variables": {
        "color-brand-primary": "rgb(31, 41, 55)",
        "color-brand-content": "#d8232a",
        "color-brand-visited": "var(--color-brand-content)",
        "color-link--visited--hover": "rgb(153, 25, 30)",
        "color-link--hover": "rgb(153, 25, 30)",
    },
    "dark_css_variables": {
        "color-brand-primary": "white",
        "color-brand-content": "#d8232a",
        "color-brand-visited": "var(--color-brand-content)",
        "color-background-primary": "#1c2834",
        "color-background-secondary": "#263645",
        "color-sidebar-item-background--hover": "#2f4356",
        "color-link--visited--hover": "rgb(153, 25, 30)",
        "color-link--hover": "rgb(153, 25, 30)",
    },
}

html_css_files = [
    'css/custom.css',
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = "_static/img/favicon.ico"
html_title = "ZAS/EAK-Copilot documentation"
