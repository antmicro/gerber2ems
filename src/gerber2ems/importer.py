"""Module containing functions for importing gerbers."""

import csv
import json
import subprocess
import os
import logging
from typing import List, Tuple
import sys
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import PIL.Image
import numpy as np
from nanomesh import Image, Mesher2D
import matplotlib as mpl

from gerber2ems.config import Config
from gerber2ems.constants import (
    GEOMETRY_DIR,
    UNIT,
    STACKUP_FORMAT_VERSION,
)

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

    fab = Path.cwd() / "fab"
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
    output_filename = Path.cwd() / GEOMETRY_DIR / gerber_filename.with_suffix(".png").name.rpartition("-")[2]

    dpi = 1 / (cfg.pixel_size * UNIT / 0.0254)
    logger.debug("Generating PNG (DPI: %d) for %s", dpi, gerber_filename)

    not_cropped_name = output_filename.with_stem(output_filename.stem + "_not_cropped")
    if not dpi.is_integer():
        logger.warning("DPI is not an integer number: %f", dpi)
    gerbv_command = [
        "gerbv",
        gerber_filename,
        edge_filename,
        "--background=#000000",
        "--foreground=#ffffffff",
        "--foreground=#00007f",
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

    edge_width = 0
    v_probe = not_cropped_image.height / 2
    px_access = not_cropped_image.load()
    for i in range(not_cropped_image.width):
        px = px_access[i, v_probe]
        if 0xBF > px[2] > 0x3F:
            # px belongs to edge
            edge_width += 1
            continue
        if edge_width != 0:
            break
    ew2 = int(edge_width / 2)
    cropped_image = not_cropped_image.crop((ew2, ew2, not_cropped_image.width - ew2, not_cropped_image.height - ew2))
    cropped_image.save(output_filename)

    if not cfg.arguments.debug:
        os.remove(not_cropped_name)


def get_dimensions(input_filename: str) -> Tuple[int, int]:
    """Return board dimensions based on png.

    Opens PNG found in `ems/geometry` directory,
    gets it's size and subtracts border thickness to get board dimensions
    """
    pixel_size = cfg.pixel_size
    path = os.path.join(GEOMETRY_DIR, input_filename)
    image = PIL.Image.open(path)
    image_width, image_height = image.size
    height = image_height * pixel_size
    width = image_width * pixel_size
    logger.debug("Board dimensions read from file are: height:%f width:%f", height, width)
    return (width, height)


def get_triangles(input_filename: str) -> np.ndarray:
    """Triangulate image.

    Processes file from `ems/geometry`.
    Converts to grayscale, thresholds it to remove border
    and then uses Nanomesh to create a triangular mesh of the copper.
    Returns a list of triangles, where each triangle consists of coordinates for each vertex.
    """
    img_path = Path(GEOMETRY_DIR) / input_filename
    image = PIL.Image.open(img_path)
    gray = image.convert("L")
    thresh = gray.point(lambda p: 255 if p < 230 else 0)
    cooper_np = np.array(thresh)
    copper = Image(cooper_np)

    mesher = Mesher2D(copper)
    # These constans are set so there won't be to many triangles.
    # If in some case triangles are too coarse they should be adjusted
    max_edge_dist = int(60000 / (cfg.pixel_size**2 / 25))
    mesher.generate_contour(max_edge_dist=max_edge_dist, precision=1, group_regions=False)
    mesher.plot_contour()
    mesh = mesher.triangulate(opts=f"epAq15a{max_edge_dist}")

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
            [
                image_to_board_coordinates(t[0]),
                image_to_board_coordinates(t[1]),
                image_to_board_coordinates(t[2]),
            ]
        )
    cu_triangles_np = np.stack(cu_triangles, axis=0)

    logger.debug("Found %d triangles for %s", len(cu_triangles), input_filename)

    if cfg.arguments.debug:
        plot_mesh(img_path, image.size, cu_triangles_np)

    return cu_triangles_np


def plot_mesh(img_path: Path, img_size: Tuple[int, int], cu_triangles_np: np.ndarray) -> None:
    """Plot cooper mesh to file."""
    filename = img_path.with_stem(img_path.stem + "_mesh")
    logger.debug("Saving mesh to file: %s", filename)
    fig, ax = mpl.pyplot.subplots()

    ax.set_xlim(0, image_to_board_coordinates(img_size[0]) / 1000)
    ylim = image_to_board_coordinates(img_size[1]) / 1000
    ax.set_ylim(0, ylim)
    x, y = cu_triangles_np[:, :, 1].flatten() / 1000, -cu_triangles_np[:, :, 0].flatten() / 1000 + ylim
    cu_count = cu_triangles_np.shape[0]
    ax.set_aspect("equal")
    ax.set_xlabel("Slice X position [mm]")
    ax.set_ylabel("Slice Y position [mm]")
    ax.set_title(f"{img_path.stem} triangle mesh")
    c = np.ones(cu_count)
    cmap = mpl.colors.ListedColormap("#B87333")
    ax.tripcolor(x, y, c, triangles=np.arange(cu_count * 3).reshape((cu_count, 3)), edgecolor="k", lw=0.05, cmap=cmap)
    fig.savefig(
        filename,
        dpi=2400,
        bbox_inches="tight",
    )
    mpl.pyplot.close()


def image_to_board_coordinates(point: np.ndarray | int) -> np.ndarray:
    """Transform point coordinates from image to board coordinates."""
    return point * cfg.pixel_size


def get_vias() -> List[List[float]]:
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
    vias: List[List[float]] = []
    with open(os.path.join(os.getcwd(), "fab", drill_filename), "r", encoding="utf-8") as drill_file:
        for line in drill_file.readlines():
            # Regex for finding drill sizes (in mm)
            match = re.fullmatch("T([0-9]+)C([0-9]+.[0-9]+)\\n", line)
            if match is not None:
                drills[int(match.group(1))] = float(match.group(2)) / 1000 / UNIT

            # Regex for finding drill switches (in mm)
            match = re.fullmatch("T([0-9]+)\\n", line)
            if match is not None:
                current_drill = int(match.group(1))

            # Regex for finding hole positions (in mm)
            match = re.fullmatch("X([0-9]+.[0-9]+)Y([0-9]+.[0-9]+)\\n", line)
            if match is not None:
                if current_drill in drills:
                    logger.debug(
                        f"Adding via at: X{float(match.group(1)) / 1000 / UNIT}Y{float(match.group(2)) / 1000 / UNIT}"
                    )
                    vias.append(
                        [
                            float(match.group(1)) / 1000 / UNIT,
                            float(match.group(2)) / 1000 / UNIT,
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
    ports: List[Tuple[int, Tuple[float, float], float]] = []
    for filename in os.listdir(os.path.join(os.getcwd(), "fab")):
        if filename.endswith("-pos.csv"):
            ports += get_ports_from_file(os.path.join(os.getcwd(), "fab", filename))

    for number, position, direction in ports:
        if len(cfg.ports) > number:
            port = cfg.ports[number]
            if port.position is None:
                cfg.ports[number].position = position
                cfg.ports[number].direction = direction
            else:
                logger.warning(
                    "Port #%i is defined twice on the board. Ignoring the second instance",
                    number,
                )
    for index, port in enumerate(cfg.ports):
        if port.position is None:
            logger.error("Port #%i is not defined on board. It will be skipped", index)


def get_ports_from_file(filename: str) -> List[Tuple[int, Tuple[float, float], float]]:
    """Parse pnp CSV file and return all ports in format (number, (x, y), direction)."""
    ports: List[Tuple[int, Tuple[float, float], float]] = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers
        for row in reader:
            if "Simulation_Port" in row[2] or "Simulation-Port" in row[2]:
                number = int(row[0][2:])
                ports.append(
                    (
                        number - 1,
                        (float(row[3]) / 1000 / UNIT, float(row[4]) / 1000 / UNIT),
                        float(row[5]),
                    )
                )
                logging.debug("Found port #%i position in pos file", number)

    return ports
