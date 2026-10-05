"""Compatibility import for :mod:`vadbench.workflows.efficiency`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.workflows.efficiency")
