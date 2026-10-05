"""Compatibility import for :mod:`vadbench.workflows.video_efficiency`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("vadbench.workflows.video_efficiency")
