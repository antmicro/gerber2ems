"""Contains common helper classes and functions."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import Self

from serde import coerce, serde

from gerber2ems.constants import BASE_UNIT, UNIT_MULTIPLIER

MM2SIM_FACTOR = UNIT_MULTIPLIER / BASE_UNIT / 1000


def mm_to_sim(val: float) -> float:
    """Convert value in millimeters to simulation space units."""
    return val * MM2SIM_FACTOR


def sim_to_mm(val: float) -> float:
    """Convert value in simulation space units to millimeters."""
    return val / MM2SIM_FACTOR


@serde(type_check=coerce)
@dataclass
class Point:
    """Coordinates of 2D point."""

    x: float
    y: float

    def mm_to_sim(self) -> Point:
        """Convert coordinates in millimeters to simulation space units."""
        return Point(mm_to_sim(self.x), mm_to_sim(self.y))

    def sim_to_mm(self) -> Point:
        """Convert coordinates in simulation space units to millimeters."""
        return Point(sim_to_mm(self.x), sim_to_mm(self.y))

    def mirror(self, axis: str) -> None:
        """Mirror point using `axis` axis."""
        setattr(self, axis, -getattr(self, axis))

    def rotate(self, angle: float) -> None:
        """Rotate point around (0,0)."""
        (self.x, self.y) = (self.x * cos(angle) + self.y * sin(angle), self.x * sin(angle) + self.y * cos(angle))

    def scale(self, scale: float) -> None:
        """Scale position by `scale`."""
        (self.x, self.y) = (self.x * scale, self.y * scale)

    def move(self, offset: Self) -> None:
        """Move point by `offset`."""
        (self.x, self.y) = (self.x + offset.x, self.y + offset.y)

    def vector_to(self, other: Self) -> Self:
        """Vector distance to `other` point."""
        return self.__class__(other.x - self.x, other.y - self.y)

    def distance_to(self, other: Self) -> float:
        """Euclidean distance to `other` point."""
        vec = self.vector_to(other)
        return sqrt(vec.x**2 + vec.y**2)

    def bbox_min(self, other: Self) -> Self:
        """Point with minimal coordinate values in bounding box that wraps two points."""
        return self.__class__(min(self.x, other.x), min(self.y, other.y))

    def bbox_max(self, other: Self) -> Self:
        """Point with maximal coordinate values in bounding box that wraps two points."""
        return self.__class__(max(self.x, other.x), max(self.y, other.y))


@serde(type_check=coerce)
@dataclass
class PadProp:
    """Basic properties of signal pad required for simulation."""

    position: Point
    """Position of pad that is reference point for port"""
    size: Point
    """Size of pad that is reference point for port"""
    id: str | None = None
    """Expected name format: `J15,9` (component-designator,pin-number)
    This will be matched to pad designated with `%TO.P,J15,9,net_name*%` in gerber file"""

    def sim_pad(self) -> PadProp:
        """Convert pad coordinates to simulation space."""
        return PadProp(self.position.mm_to_sim(), self.size.mm_to_sim())

    def mm_pad(self) -> PadProp:
        """Convert pad coordinates from simulation space to real world millimeters."""
        return PadProp(self.position.sim_to_mm(), self.size.sim_to_mm())
