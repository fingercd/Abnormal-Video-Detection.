"""Compatibility import for :mod:`vadbench.workflows.feature_resume`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.workflows.feature_resume")
