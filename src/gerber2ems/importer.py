"""Module containing functions for importing gerbers."""

from __future__ import annotations

import enum
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import PIL.Image
from nanomesh import Image, Mesher2D

from gerber2ems.common import Point, mm_to_sim
from gerber2ems.config import Config, get_ports_from_all_pos_files
from gerber2ems.constants import BASE_UNIT, FAB_DIR, GEOMETRY_DIR, STACKUP_FORMAT_VERSION, UNIT_MULTIPLIER

logger = logging.getLogger(__name__)

PIL.Image.MAX_IMAGE_PIXELS = None
cfg = Config()


def process_gbrs_to_pngs() -> None:
    """Process all gerber files to PNG's.

    Finds edge cuts gerber as well as copper gerbers in `fab` directory.
    Processes copper gerbers into PNG's using edge_cuts for framing.
    Output is saved to `ems/geometry` folder
    """
    logger.info("Processing gerber files (may take a while for larger boards)")

    fab = FAB_DIR
    edge = next(fab.glob("*Edge_Cuts.gbr"), None)
    if edge is None:
        logger.error("No edge_cuts gerber found")
        sys.exit(1)

    layers = list(fab.glob("*_Cu.gbr"))
    if len(layers) == 0:
        logger.warning("No copper gerbers found")

    with Pool(initargs=(cfg._config,), initializer=Config.set_config) as p:
        p.map(partial(gbr_to_png, edge), layers)


def gbr_to_png(edge_filename: Path, gerber_filename: Path) -> None:
    """Generate PNG from gerber file.

    Generates PNG of a gerber using gerbv.
    Edge cuts gerber is used to crop the image correctly.
    Output DPI is based on config.pixel_size constant.
    """
    output_filename = GEOMETRY_DIR / gerber_filename.with_suffix(".png").name.rpartition("-")[2]

    dpi = 1 / (cfg.pixel_size * BASE_UNIT / 0.0254)
    logger.debug("Generating PNG (DPI: %d) for %s", dpi, gerber_filename)

    not_cropped_name = output_filename.with_stem(output_filename.stem + "_not_cropped")
    if not dpi.is_integer():
        logger.warning("DPI is not an integer number: %f", dpi)
    gerbv_command = [
        "gerbv",
        edge_filename,
        gerber_filename,
        "--background=#000000",
        "--foreground=#0000ffff",
        "--foreground=#00ff00ff",
        "-o",
        not_cropped_name,
        "--dpi",
        f"{dpi}",
        "--border=0",
        "--export=png",
        "-a",
    ]

    subprocess.run(gerbv_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    not_cropped_image = PIL.Image.open(not_cropped_name)
    mono_img = not_cropped_image.getchannel("B")
    bbox = mono_img.getbbox()
    if not bbox:
        raise Exception(f"Bounding box of edge cuts in {not_cropped_name} can not be identified!")
    edge_width = 0
    v_probe = int((bbox[1] + bbox[3]) / 2)
    px_access = mono_img.load()
    assert px_access
    for i in range(mono_img.width):
        px = px_access[i, v_probe]
        if px > 0x3F:
            # px belongs to edge
            edge_width += 1
            continue
        if edge_width != 0:
            break
    ew2 = int(edge_width / 2)
    cropped_image = not_cropped_image.crop((bbox[0] + ew2, bbox[1] + ew2, bbox[2] - ew2, bbox[3] - ew2))
    cropped_image.save(output_filename)

    if not cfg.arguments.debug:
        os.remove(not_cropped_name)


def get_dimensions(input_filename: str) -> tuple[int, int]:
    """Return board dimensions based on png.

    Opens PNG found in `ems/geometry` directory,
    gets it's size and subtracts border thickness to get board dimensions
    """
    pixel_size = cfg.pixel_size
    path = GEOMETRY_DIR / input_filename
    image = PIL.Image.open(path)
    image_width, image_height = image.size
    height = image_height * pixel_size * UNIT_MULTIPLIER
    width = image_width * pixel_size * UNIT_MULTIPLIER
    logger.debug("Board dimensions read from file are: height:%f width:%f", height, width)
    return (width, height)


@dataclass(frozen=True)
class Edge:
    """Starting/ending points defining a segment in 2D space."""

    start: Point
    end: Point

    def rev(self) -> Edge:
        """Return reversed segment."""
        return Edge(self.end, self.start)


Polygon = list[Point]
PolygonEdges = list[Edge]


class TriangleMerger:
    """Class for merging list of triangles into larger polygons."""

    class __TripletOrientation(IntEnum):
        CLOCKWISE = enum.auto()
        COUNTERCLOCKWISE = enum.auto()
        COLINEAR = enum.auto()
        COLINEAR_DISJOINT = enum.auto()

    @classmethod
    def _orientation(cls, p1: Point, p2: Point, p3: Point) -> __TripletOrientation:
        """Determine the orientation of three points (counter-clockwise test)."""
        cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        if cross > 0:
            return TriangleMerger.__TripletOrientation.CLOCKWISE
        if cross < 0:
            return TriangleMerger.__TripletOrientation.COUNTERCLOCKWISE
        if TriangleMerger._on_segment(p3, p1, p2):
            return TriangleMerger.__TripletOrientation.COLINEAR
        return TriangleMerger.__TripletOrientation.COLINEAR_DISJOINT

    @classmethod
    def _on_segment(cls, p: Point, a: Point, b: Point) -> bool:
        """Check if point p lies on segment ab (assuming p, a, b are collinear).

        Performs a bounding box check to determine if the point is within the
        segment's extent. Used to detect shared vertices in collinear cases.

        Returns:
        True if p lies on segment ab, False otherwise.

        """
        return min(a.x, b.x) <= p.x <= max(a.x, b.x) and min(a.y, b.y) <= p.y <= max(a.y, b.y)

    @classmethod
    def _segments_intersect(cls, e1: Edge, e2: Edge) -> bool:
        """Determine if segment `e1` intersects segment `e2` using the CCW orientation method.

        Returns:
        True if the segments intersect, False otherwise.

        """
        d1 = TriangleMerger._orientation(e1.start, e2.start, e1.end)
        d2 = TriangleMerger._orientation(e1.start, e2.end, e1.end)
        d3 = TriangleMerger._orientation(e2.start, e1.start, e2.end)
        d4 = TriangleMerger._orientation(e2.start, e1.end, e2.end)

        # Collinear cases
        if any(d == TriangleMerger.__TripletOrientation.COLINEAR for d in (d1, d2, d3, d4)):
            return True

        # General case:
        # The endpoints of first segment lie on opposite sides of the other segment's line.
        if d1 != d2 and d3 != d4:
            return True

        return False

    @classmethod
    def _polygon_self_intersects(cls, polygon: PolygonEdges) -> bool:
        """Check if a polygon self-intersects by testing all non-adjacent edge pairs.

        Returns:
        True if the polygon self-intersects, False otherwise.

        """
        n = len(polygon)
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent edges that share a vertex
                if i == 0 and j == n - 1:
                    continue
                if cls._segments_intersect(polygon[i], polygon[j]):
                    return True
        return False

    @classmethod
    def _polygon_edges(cls, vertices: Polygon) -> PolygonEdges:
        """Return the directed edges of a polygon.

        Edges are returned in counterclockwise order as tuples of (start, end) points.

        Arguments:
        vertices: List of vertices defining the polygon

        Returns:
        List of (start, end) edge tuples in CCW order.

        """
        n = len(vertices)
        return [Edge(vertices[k], vertices[(k + 1) % n]) for k in range(n)]

    @classmethod
    def _merge_two_polygons(cls, fa: PolygonEdges, fb: PolygonEdges, edge: Edge) -> PolygonEdges | None:
        """Merge two adjacent polygons that share a common edge.

        The polygons are represented as dictionaries mapping edges to None values.
        The merge rotates each polygon to start at the shared edge, removes the shared
        edge, and concatenates the remaining vertices in the correct order.

        Arguments:
        fa: List of edges for the first polygon
        fb: List of edges for the second polygon
        edge: The shared edge (start, end) being merged along

        Returns:
        The merged polygon as a list of vertices, or None if the result
        self-intersects.

        """
        edge_rev = edge.rev()
        a_rev = False  # a_rev is always false due to _find_common_edge impl
        b_rev = edge_rev in fb

        a_idx = fa.index(edge_rev if a_rev else edge)
        b_idx = fb.index(edge_rev if b_rev else edge)

        # Rotate each polygon to start at the shared edge
        a = fa[a_idx + 1 :] + fa[:a_idx]
        b = fb[b_idx + 1 :] + fb[:b_idx]

        # Determine rotation for b based on edge orientation
        b = list(reversed(b)) if b_rev == a_rev else b

        # Combine the polygons, removing the shared edge
        merged = a + b

        # Verify the merged polygon doesn't self-intersect
        if cls._polygon_self_intersects(merged):
            return None

        return merged

    @staticmethod
    def _find_common_edge(poly1: PolygonEdges, poly2: PolygonEdges) -> Edge | None:
        """Find common edge between two polygons.

        Returns:
        Common edge if there is common edge, else None

        """
        for p2e in chain(poly2, (e.rev() for e in poly2)):
            if p2e in poly1:
                return p2e
        return None

    @classmethod
    def merge_triangles(cls, triangles: list[Polygon]) -> list[Polygon]:
        """Merge a list of triangles into maximal non-self-intersecting polygons.

        Two triangles can be merged if they share exactly one edge. The merging
        process continues until no more merges are possible, producing polygons
        that exactly fill the same space as the input triangles without holes.

        Arguments:
        triangles: NumPy array of shape (n, 3, 2) where each triangle has 3
        vertices with (x, y) coordinates

        Returns:
        List of merged polygons.

        """
        polygons = [(set((p.x, p.y) for p in tri), cls._polygon_edges(tri)) for tri in triangles]

        for iteration in range(12):
            if iteration % 3 == 2:
                polygons = [poly for poly in reversed(polygons) if poly]

            for poly1_idx, (poly1_set, poly1) in enumerate(polygons):
                if not poly1 or len(poly1) > 200:
                    continue
                for poly2_set, poly2 in polygons[poly1_idx + 1 :]:
                    if not poly2 or len(poly1) > 200 or len(poly2) > 200:
                        continue

                    if len(poly1_set & poly2_set) != 2:
                        # check for points common between poly as fast failing common_edge check
                        continue

                    common_edge = cls._find_common_edge(poly1, poly2)
                    if not common_edge:
                        continue

                    # Try to merge the polys
                    merged = cls._merge_two_polygons(poly1, poly2, common_edge)
                    if merged is not None:
                        poly1.clear()
                        poly1_set |= poly2_set
                        poly1.extend(merged)
                        poly2.clear()
                        break

        # Convert final polys to polygon vertex lists
        return [[edge.start for edge in poly] for (_poly_set, poly) in polygons if poly]


def get_triangles(input_filename: str) -> list[Polygon]:
    """Triangulate image.

    Processes file from `ems/geometry`.
    Converts to grayscale, thresholds it to remove border
    and then uses Nanomesh to create a triangular mesh of the copper.
    Returns a list of triangles, where each triangle consists of coordinates for each vertex.
    """
    img_path = GEOMETRY_DIR / input_filename
    image = PIL.Image.open(img_path)
    gray = image.getchannel("G")
    if gray.getextrema()[1] < 230:  # type:ignore
        # Image is empty - no cooper features on this layer
        return []
    thresh = gray.point(lambda p: 255 if p < 230 else 0)
    cooper_np = np.array(thresh)
    copper = Image(cooper_np)

    mesher = Mesher2D(copper)
    # These constants are set so there won't be to many triangles.
    # If in some case triangles are too coarse they should be adjusted
    max_edge_dist = int(6000000 / cfg.pixel_size**2)
    mesher.generate_contour(max_edge_dist=max_edge_dist, precision=1, group_regions=False)
    mesher.plot_contour()
    mesh = mesher.triangulate(opts=f"pAq10a{max_edge_dist}S0u")

    points = mesh.get("triangle").points
    cells = mesh.get("triangle").cells

    # Selecting only triangles that represent copper
    cu_triangles = []
    for cell in cells:
        t = (points[cell[0]], points[cell[1]], points[cell[2]])
        center = np.average(t, axis=0)
        is_copper = int(cooper_np[int(center[0]), int(center[1])] < 127)
        if not is_copper:
            continue
        cu_triangles.append(
            Polygon(
                [
                    image_to_board_coordinates(t[0]),
                    image_to_board_coordinates(t[1]),
                    image_to_board_coordinates(t[2]),
                ]
            )
        )

    brd_size = image_to_board_coordinates(image.size)
    logger.debug("Found %d triangles for %s", len(cu_triangles), input_filename)

    if cfg.arguments.debug:
        plot_polygons(img_path, cu_triangles, brd_size, "")

    polygons = TriangleMerger.merge_triangles(cu_triangles)

    logger.debug(f"Reduced to {len(polygons)} for {input_filename}")
    if cfg.arguments.debug:
        plot_polygons(img_path, polygons, brd_size, "final")

    return polygons


def plot_polygons(img_path: Path, polygons: list[Polygon], brd_size: Point, name_suffix: str) -> None:
    """Plot cooper mesh to file."""
    from matplotlib import pyplot
    from matplotlib.patches import Polygon as MplPolygon

    filename = img_path.with_stem(img_path.stem + "_" + name_suffix + "mesh")
    logger.debug("Saving mesh to file: %s", filename)
    # Create plot
    fig, ax = pyplot.subplots()

    # Set limits and aspect
    ax.set_xlim(0, brd_size.x / 1000)
    ylim = brd_size.y / 1000
    ax.set_ylim(0, ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("Slice X position [mm]")
    ax.set_ylabel("Slice Y position [mm]")
    ax.set_title(f"{img_path.stem} {name_suffix} mesh")

    for poly in polygons:
        spoly = [(p.y / 1000, -p.x / 1000 + ylim) for p in poly]
        patch = MplPolygon(spoly, closed=True, lw=0.05, facecolor="skyblue", edgecolor="black")
        ax.add_patch(patch)

    fig.savefig(filename, dpi=2400, bbox_inches="tight")
    pyplot.close()


def image_to_board_coordinates(point: tuple[float, float]) -> Point:
    """Transform point coordinates from image to board coordinates."""
    return Point(point[0] * cfg.pixel_size * UNIT_MULTIPLIER, point[1] * cfg.pixel_size * UNIT_MULTIPLIER)


def get_vias() -> list[list[float]]:
    """Get via information from excellon file.

    Looks for excellon file in `fab` directory. Its filename should end with `-PTH.drl`
    It then processes it to find all vias.
    """
    files = os.listdir(os.path.join(os.getcwd(), "fab"))
    drill_filename = next(filter(lambda name: "-PTH.drl" in name, files), None)
    if drill_filename is None:
        logger.error("Couldn't find drill file")
        sys.exit(1)

    drills = {0: 0.0}  # Drills are numbered from 1. 0 is added as a "no drill" option
    current_drill = 0
    vias: list[list[float]] = []
    with open(FAB_DIR / drill_filename, "r", encoding="utf-8") as drill_file:
        for line in drill_file.readlines():
            # Regex for finding drill sizes (in mm)
            match = re.fullmatch("T([0-9]+)C([0-9]+.[0-9]+)\\n", line)
            if match is not None:
                drills[int(match.group(1))] = mm_to_sim(float(match.group(2)))

            # Regex for finding drill switches (in mm)
            match = re.fullmatch("T([0-9]+)\\n", line)
            if match is not None:
                current_drill = int(match.group(1))

            # Regex for finding hole positions (in mm)
            match = re.fullmatch("X([0-9]+.[0-9]+)Y([0-9]+.[0-9]+)\\n", line)
            if match is not None:
                if current_drill in drills:
                    logger.debug(
                        f"Adding via at: X{mm_to_sim(float(match.group(1)))} \
                        Y{mm_to_sim(float(match.group(2)))}"
                    )
                    vias.append(
                        [
                            mm_to_sim(float(match.group(1))),
                            mm_to_sim(float(match.group(2))),
                            drills[current_drill],
                        ]
                    )
                else:
                    logger.warning("Drill file parsing failed. Drill with specifed number wasn't found")
    logger.debug("Found %d vias", len(vias))
    return vias


def import_stackup() -> None:
    """Import stackup information from `fab/stackup.json` file and load it into config object."""
    filename = "fab/stackup.json"
    with open(filename, "r", encoding="utf-8") as file:
        try:
            stackup = json.load(file)
        except json.JSONDecodeError as error:
            logger.error(
                "JSON decoding failed at %d:%d: %s",
                error.lineno,
                error.colno,
                error.msg,
            )
            sys.exit(1)
        ver = stackup["format_version"]
        if (
            ver is not None
            and ver.split(".")[0] == STACKUP_FORMAT_VERSION.split(".", maxsplit=1)[0]
            and ver.split(".")[1] >= STACKUP_FORMAT_VERSION.split(".", maxsplit=1)[1]
        ):
            cfg.load_stackup(stackup)
        else:
            logger.error(
                "Stackup format (%s) is not supported (supported: %s)",
                ver,
                STACKUP_FORMAT_VERSION,
            )
            sys.exit()


def import_port_positions() -> None:
    """Import port positions from PnP .csv files.

    Looks for all PnP files in `fab` folder (files ending with `-pos.csv`)
    Parses them to find port footprints and inserts their position information to config object.
    """
    for number, position, direction in get_ports_from_all_pos_files():
        if len(cfg.all_ports) > number:
            port = cfg.all_ports[number]
            if port.position is None:
                cfg.all_ports[number].position = position
                cfg.all_ports[number].direction = direction
            else:
                logger.warning(
                    "Port #%i is defined twice on the board. Ignoring the second instance",
                    number,
                )
    for index, port in enumerate(cfg.all_ports):
        if port.position is None:
            logger.error("Port #%i is not defined on board. It will be skipped", index)
