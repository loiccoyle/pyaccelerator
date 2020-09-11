"""Python package to build simple toy accelerator.
"""
from . import elements
from .beam import Beam
from .elements import *
from .lattice import Lattice

__all__ = ["Beam", "Lattice"]
__all__.extend(elements.__all__)

__version__ = "0.1.0"
