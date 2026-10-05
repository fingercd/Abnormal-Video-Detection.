"""Compatibility import for :mod:`vadbench.integrations.detectors.urdmu.inference`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.integrations.detectors.urdmu.inference")
