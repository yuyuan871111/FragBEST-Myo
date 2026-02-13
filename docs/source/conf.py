# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# The original settings code is based on ProLIF.
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add the project directory to the system path
sys.path.insert(0, str(Path("../..", "utils").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FragBEST-Myo"
author = "Yu-Yuan (Stuart) Yang"
copyright = f"2024-{datetime.now().year}, {author}"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "myst_nb",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = []

# The myst_nb extension is used to support Jupyter notebooks
# https://myst-nb.readthedocs.io/en/latest/
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "myst-nb",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

html_theme_options = {
    "repository_url": "https://github.com/yuyuan871111/FragBEST-Myo",
    "path_to_docs": "docs",
    "use_source_button": True,
    "use_download_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yuyuan871111/FragBEST-Myo",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}
