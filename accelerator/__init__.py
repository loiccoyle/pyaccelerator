import logging

from . import elements
from .beam import Beam
from .elements import *
from .lattice import Lattice

# TODO: remove this eventually
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

__all__ = ["Beam", "Lattice"]
__all__.extend(elements.__all__)
