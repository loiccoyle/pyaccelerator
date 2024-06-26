from typing import Any, Dict

import numpy as np

from .. import elements
from .base import BaseElement


def straight_element(theta: float, d_s: float) -> np.ndarray:
    return np.array([np.cos(theta) * d_s, np.sin(theta) * d_s, 0])


def bent_element(theta: float, d_s: float, rho: float) -> np.ndarray:
    phi = d_s / rho
    R = 2 * rho * np.sin(phi / 2)
    d_x = R * np.cos(theta + phi / 2)
    d_y = R * np.sin(theta + phi / 2)
    d_theta = phi
    return np.array([d_x, d_y, d_theta])


def deserialize(dic: Dict[str, Any]) -> BaseElement:
    """Deserialize a element dictionary into an element instance.

    Args:
        dic: Element dict.

    Returns:
        Element instance.
    """
    element = dic.pop("element")
    return getattr(elements, element)(**dic)
