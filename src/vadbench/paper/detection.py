"""Compatibility import for :mod:`vadbench.workflows.detection`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.workflows.detection")
