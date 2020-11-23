"""Python package to build simple toy accelerator.
"""
import logging

from . import elements
from .beam import Beam
from .constraints import TargetDispersion, TargetGlobal, TargetPhasespace, TargetTwiss
from .elements import *
from .lattice import Lattice

__all__ = [
    "Beam",
    "Lattice",
    "TargetPhasespace",
    "TargetTwiss",
    "TargetDispersion",
    "TargetGlobal",
]
__all__.extend(elements.__all__)

__version__ = "0.1.4"

logger = logging.getLogger("pyaccelerator")
