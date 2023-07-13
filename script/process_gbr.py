""" Module containing functions for importing gerbers """
import subprocess
import os
import logging
from typing import Tuple
import sys

import cv2
import numpy as np

from constants import TMP_DIR, UNIT, PIXEL_SIZE

logger = logging.getLogger(__name__)


def process():
    """Processes all gerber files"""
    logger.info("Processing gerber files")

    files = os.listdir(os.getcwd())

    edge = next(filter(lambda name: "Edge_Cuts.gbr" in name, files), None)
    if edge is None:
        logger.error("No edge_cuts gerber found")
        sys.exit(1)

    layers = list(filter(lambda name: "_Cu.gbr" in name, files))
    if len(layers) == 0:
        logger.warning("No copper gerbers found")

    for name in layers:
        output = name.split("-")[1].split(".")[0] + ".png"
        gbr_to_png(name, edge, os.path.join(os.getcwd(), TMP_DIR, output))


def gbr_to_png(gerber: str, edge: str, output: str) -> None:
    """Generate PNG from gerber file"""
    logger.debug("Generating PNG for %s", gerber)
    not_cropped_name = os.path.join(os.getcwd(), TMP_DIR, "not_cropped.png")

    dpi = 1 / (PIXEL_SIZE * UNIT / 0.0254)
    if not dpi.is_integer():
        logger.warning("DPI is not an integer number: %f", dpi)

    subprocess.call(
        f"gerbv {gerber} {edge} --background=#ffffff --foreground=#000000ff --foreground=#ff00000f -o {not_cropped_name} --dpi={dpi} --export=png -a",
        shell=True,
    )
    subprocess.call(f"convert {not_cropped_name} -trim {output}", shell=True)
    os.remove(not_cropped_name)


def get_contours(input_name: str) -> Tuple[np.ndarray, ...]:
    """Finds outlines in the image"""

    path = os.path.join(TMP_DIR, input_name)
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
