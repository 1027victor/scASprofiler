# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scASprofiler'
copyright = '2025, hpw'
author = 'hpw'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



templates_path = ['_templates']
exclude_patterns = []

extensions = [
    # "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "sphinx.ext.coverage",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.napoleon",
    # "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    # "sphinx_autodoc_typehints",
    "nbsphinx",
    # "edit_on_github",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
pygments_style = 'sphinx'
html_theme_options = dict(navigation_depth=1, titles_only=True)
