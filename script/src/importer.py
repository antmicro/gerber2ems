"""Module containing functions for importing gerbers."""
import csv
import json
import subprocess
import os
import logging
from typing import List, Tuple
import sys
import re

import PIL.Image
import numpy as np
from nanomesh import Image
from nanomesh import Mesher2D
import matplotlib.pyplot as plt
from config import Config

from constants import (
    GEOMETRY_DIR,
    UNIT,
    PIXEL_SIZE,
    BORDER_THICKNESS,
    STACKUP_FORMAT_VERSION,
)

logger = logging.getLogger(__name__)


def process_gbrs_to_pngs():
    """Process all gerber files to PNG's.

    Finds edge cuts gerber as well as copper gerbers in `fab` directory.
    Processes copper gerbers into PNG's using edge_cuts for framing.
    Output is saved to `ems/geometry` folder
    """
    logger.info("Processing gerber files")

    files = os.listdir(os.path.join(os.getcwd(), "fab"))

    edge = next(filter(lambda name: "Edge_Cuts.gbr" in name, files), None)
    if edge is None:
        logger.error("No edge_cuts gerber found")
        sys.exit(1)

    layers = list(filter(lambda name: "_Cu.gbr" in name, files))
    if len(layers) == 0:
        logger.warning("No copper gerbers found")

    for name in layers:
        output = name.split("-")[-1].split(".")[0] + ".png"
        gbr_to_png(
            os.path.join(os.getcwd(), "fab", name),
            os.path.join(os.getcwd(), "fab", edge),
            os.path.join(os.getcwd(), GEOMETRY_DIR, output),
        )


def gbr_to_png(gerber_filename: str, edge_filename: str, output_filename: str) -> None:
    """Generate PNG from gerber file.

    Generates PNG of a gerber using gerbv.
    Edge cuts gerber is used to crop the image correctly.
    Output DPI is based on PIXEL_SIZE constant.
    """
    logger.debug("Generating PNG for %s", gerber_filename)
    not_cropped_name = f"{output_filename.split('.')[0]}_not_cropped.png"

    dpi = 1 / (PIXEL_SIZE * UNIT / 0.0254)
    if not dpi.is_integer():
        logger.warning("DPI is not an integer number: %f", dpi)

    gerbv_command = f"gerbv {gerber_filename} {edge_filename}"
    gerbv_command += " --background=#000000 --foreground=#ffffffff --foreground=#0000ff"
    gerbv_command += f" -o {not_cropped_name}"
    gerbv_command += f" --dpi={dpi} --export=png -a"

    subprocess.call(gerbv_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    not_cropped_image = PIL.Image.open(not_cropped_name)

    # image_width, image_height = not_cropped_image.size
    cropped_image = not_cropped_image.crop(not_cropped_image.getbbox())
    cropped_image.save(output_filename)

    if not Config.get().arguments.debug:
        os.remove(not_cropped_name)


def get_dimensions(input_filename: str) -> Tuple[int, int]:
    """Return board dimensions based on png.

    Opens PNG found in `ems/geometry` directory,
    gets it's size and subtracts border thickness to get board dimensions
    """
    path = os.path.join(GEOMETRY_DIR, input_filename)
    image = PIL.Image.open(path)
    image_width, image_height = image.size
    height = image_height * PIXEL_SIZE - BORDER_THICKNESS
    width = image_width * PIXEL_SIZE - BORDER_THICKNESS
    logger.debug("Board dimensions read from file are: height:%f width:%f", height, width)
    return (width, height)


def get_triangles(input_filename: str) -> np.ndarray:
    """Triangulate image.

    Processes file from `ems/geometry`.
    Converts to grayscale, thresholds it to remove border
    and then uses Nanomesh to create a triangular mesh of the copper.
    Returns a list of triangles, where each triangle consists of coordinates for each vertex.
    """
    path = os.path.join(GEOMETRY_DIR, input_filename)
    image = PIL.Image.open(path)
    gray = image.convert("L")
    thresh = gray.point(lambda p: 255 if p < 230 else 0)
    copper = Image(np.array(thresh))

    mesher = Mesher2D(copper)
    # These constans are set so there won't be to many triangles.
    # If in some case triangles are too coarse they should be adjusted
    mesher.generate_contour(max_edge_dist=10000, precision=2)
    mesher.plot_contour()
    mesh = mesher.triangulate(opts="a100000")

    if Config.get().arguments.debug:
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, input_filename + "_mesh.png")
        logger.debug("Saving mesh to file: %s", filename)
        mesh.plot_mpl()
        plt.savefig(filename, dpi=300)

    points = mesh.get("triangle").points
    cells = mesh.get("triangle").cells
    kinds = mesh.get("triangle").cell_data["physical"]

    triangles: np.ndarray = np.empty((len(cells), 3, 2))
    for i, cell in enumerate(cells):
        triangles[i] = [
            image_to_board_coordinates(points[cell[0]]),
            image_to_board_coordinates(points[cell[1]]),
            image_to_board_coordinates(points[cell[2]]),
        ]

    # Selecting only triangles that represent copper
    mask = kinds == 2.0

    logger.debug("Found %d triangles for %s", len(triangles[mask]), input_filename)

    return triangles[mask]


def image_to_board_coordinates(point: np.ndarray) -> np.ndarray:
    """Transform point coordinates from image to board coordinates."""
    return (point * PIXEL_SIZE) - [BORDER_THICKNESS / 2, BORDER_THICKNESS / 2]


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
                        f"Adding via at: X:{float(match.group(1)) / 1000 / UNIT} Y:{float(match.group(2)) / 1000 / UNIT}"
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


def import_stackup():
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
            Config.get().load_stackup(stackup)
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
        if len(Config.get().ports) > number:
            port = Config.get().ports[number]
            if port.position is None:
                Config.get().ports[number].position = position
                Config.get().ports[number].direction = direction
            else:
                logger.warning(
                    "Port #%i is defined twice on the board. Ignoring the second instance",
                    number,
                )
    for index, port in enumerate(Config.get().ports):
        if port.position is None:
            logger.error("Port #%i is not defined on board. It will be skipped", index)


def get_ports_from_file(filename: str) -> List[Tuple[int, Tuple[float, float], float]]:
    """Parse pnp CSV file and return all ports in format (number, (x, y), direction)."""
    ports: List[Tuple[int, Tuple[float, float], float]] = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers
        for row in reader:
            if "Simulation_Port" in row[2]:
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
