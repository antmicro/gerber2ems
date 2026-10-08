"""Contains common helper classes and functions."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from serde import coerce, serde

from gerber2ems.constants import BASE_UNIT, UNIT_MULTIPLIER


def mm_to_sim_space(val: float) -> float:
    return val / 1000 / BASE_UNIT * UNIT_MULTIPLIER


@serde(type_check=coerce)
@dataclass
class Point:
    """Coordinates of 2D point."""

    x: float
    y: float

    def mm_to_sim_space(self) -> Point:
        return Point(mm_to_sim_space(self.x), mm_to_sim_space(self.y))

    def mirror(self, axis: str) -> None:
        """Mirror point using `axis` axis."""
        setattr(self, axis, -getattr(self, axis))

    def rotate(self, angle: float) -> None:
        """Rotate point around (0,0)."""
        (self.x, self.y) = (self.x * cos(angle) + self.y * sin(angle), self.x * sin(angle) + self.y * cos(angle))

    def scale(self, scale: float) -> None:
        """Scale position by `scale`."""
        (self.x, self.y) = (self.x * scale, self.y * scale)

    def move(self, offset: "Point") -> None:
        """Move point by `offset`."""
        (self.x, self.y) = (self.x + offset.x, self.y + offset.y)
