"""CPU-only DSANet quality entry, without unrelated encoder registration.

Loads the exact snapshot workflow and shared comparison modules. Registry
assets are irrelevant to sealed prediction evaluation; all score/head/input
SHA checks remain in quality-one. Requires Python3.10+ as those modules do.
"""
import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType


def main():
    if sys.version_info < (3,10):
        raise RuntimeError('DSANet quality requires Python3.10+')
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--snapshot',type=Path,required=True)
    options,rest=parser.parse_known_args()
    if 'vadbench' in sys.modules:
        raise RuntimeError('quality snapshot must run in a fresh process')
    package=ModuleType('vadbench')
    package.__path__=[str(options.snapshot.resolve()/'src/vadbench')]
    package.__package__='vadbench'
    sys.modules['vadbench']=package
    importlib.import_module('vadbench.workflows.dsanet.__main__').main(['quality-one',*rest])


if __name__=='__main__':
    main()
