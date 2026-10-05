"""Compatibility import for :mod:`vadbench.data.feature_contracts`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.data.feature_contracts")
