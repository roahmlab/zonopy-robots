# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
]
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "html_image",
]

sd_fontawesome_latex = True

katex_prerender = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ["torch", "scipy", "zonopy", "urchin", "trimesh", "networkx"]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath('../..'))
main_ns = {}
ver_path = os.path.join(os.path.abspath('../..'), ('zonopyrobots/properties.py'))
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

DEVELOP = os.environ.get("DEVELOP", False)
if DEVELOP:
    project = 'zonopy-robots (development)'
    version = f"{main_ns['__version__']}-dev"
    version_match = "dev"
else:
    project = 'zonopy-robots'
    version = main_ns['__version__']
    version_match = f"v{version}"


copyright = '2024, ROAHM Lab'
author = 'ROAHM Lab'
release = version
html_title = f"{project} v{version}"

# Version switcher code
json_url = "https://roahmlab.github.io/zonopy-robots/versions.json"

html_theme_options = {
    "path_to_docs": version_match,
    "repository_url": "https://github.com/roahmlab/zonopy-robots",
    "repository_branch": "gh-pages",
    "use_repository_button": True,
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "article_header_end": ["version-switcher", "article-header-buttons"],
}

html_css_files = [
    "css/custom.css",
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ('https://pytorch.org/docs/main/', None),
    "zonopy": ('https://roahmlab.github.io/zonopy/latest/', None),
    "trimesh": ('https://trimesh.org/', None),
}

