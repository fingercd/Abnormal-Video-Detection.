"""Compatibility import for :mod:`vadbench.workflows.quality_comparison`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.workflows.quality_comparison")
