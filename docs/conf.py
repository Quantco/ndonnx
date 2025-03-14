# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import datetime
import importlib
import inspect
import os
import subprocess
import sys
from subprocess import CalledProcessError
from typing import cast

_mod = importlib.import_module("ndonnx")


project = "ndonnx"
copyright = f"{datetime.date.today().year}, QuantCo, Inc"
author = "QuantCo, Inc."

extensions = [
    "numpydoc",
    "sphinxcontrib.apidoc",
    "sphinx_toolbox.collapse",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "nbsphinx",
]

apidoc_module_dir = "../ndonnx"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_extra_args = ["--implicit-namespaces", "--follow-links"]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autosummary_generate = True
autodoc_type_aliases = {"npt.ArrayLike": "numpy.typing.ArrayLike"}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False


class ObjectInfo:
    def __init__(self, module_name, full_name):
        self.module_name = module_name
        self.full_name = full_name

    def linkcode_resolve(self):
        _submodule = sys.modules.get(self.module_name)
        if _submodule is None:
            return None

        _object = _submodule
        for _part in self.full_name.split("."):
            try:
                _object = getattr(_object, _part)
            except AttributeError:
                return None

        try:
            fn = inspect.getsourcefile(inspect.unwrap(_object))  # type: ignore
        except TypeError:
            fn = None
        if not fn:
            return None

        try:
            source, line_number = inspect.getsourcelines(_object)
        except OSError:
            line_number = None  # type: ignore

        if line_number:
            linespec = f"#L{line_number}-L{line_number + len(source) - 1}"
        else:
            linespec = ""

        fn = os.path.relpath(fn, start=os.path.dirname(cast(str, _mod.__file__)))

        try:
            # See https://stackoverflow.com/a/21901260
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except CalledProcessError:
            # If subprocess returns non-zero exit status
            commit = "main"

        return (
            "https://github.com/quantco/ndonnx"
            f"/blob/{commit}/{_mod.__name__.replace('.', '/')}/{fn}{linespec}"
        )


# Copied and adapted from
# https://github.com/pandas-dev/pandas/blob/4a14d064187367cacab3ff4652a12a0e45d0711b/doc/source/conf.py#L613-L659
# Required configuration function to use sphinx.ext.linkcode
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    return ObjectInfo(info["module"], info["fullname"]).linkcode_resolve()
