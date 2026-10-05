"""Compatibility import for :mod:`vadbench.workflows.urdmu_quality_export`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.workflows.urdmu_quality_export")
