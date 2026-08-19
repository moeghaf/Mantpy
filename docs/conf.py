# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import shutil
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from sphinxcontrib import katex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("mantpy")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
# ``get_all`` returns None when an editable install's cached metadata predates
# the [project.urls] entries; fall back to the known repository URL.
urls = dict(pu.split(", ") for pu in (info.get_all("Project-URL") or []))
repository_url = urls.get("Source", "https://github.com/moeghaf/Mantpy")

# The full version, including alpha/beta/rc tags
release = info["Version"]

templates_path = ["_templates"]
nitpicky = False  # external/inherited cross-refs are noisy; re-enable after a docstring polish pass
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "moeghaf",
    "github_repo": "Mantpy",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    # myst_nb supersedes myst_parser: it handles plain Markdown identically and
    # additionally executes `{code-cell}` blocks, which is how the tutorials
    # under docs/tutorials/ are built. Do not list both — they register the
    # same source suffixes and collide.
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

source_suffix = {
    ".rst": "restructuredtext",
    # Bound to myst-nb rather than "markdown": myst-nb owns both suffixes and
    # registers them with override=True, so naming a different parser here
    # silently loses execution for every page.
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Execute tutorials at build time rather than committing pre-run outputs.
# This is only affordable because `mt.datasets.toy_ecm_roi` needs no download —
# the real loaders pull 38-194 MB each, which would not fit Read the Docs'
# build budget. Executing means the tutorials cannot silently rot against the
# library: a broken example fails the build instead of shipping stale output.
nb_execution_mode = "cache"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
# Notebook-level stderr (e.g. tqdm) is not a documentation warning; without
# this, -W turns any incidental stderr into a build failure.
nb_output_stderr = "remove"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project

html_logo = "_static/mantpy_logo.png"
html_favicon = "_static/mantpy_logo.png"
html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    "logo": {
        "image_light": "_static/mantpy_logo.png",
        "image_dark": "_static/mantpy_logo.png",
        "text": "",
    },
}

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
]
