import logging

import matplotlib.pyplot as plt

from .beam import Beam
from .lattice import Lattice
from .transfer_matrix import TransferMatrix

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


plt.rcParams["figure.facecolor"] = "white"
