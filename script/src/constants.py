"""Module containing constans used in the app."""
import os

UNIT = 1e-6
PIXEL_SIZE = 10
BASE_DIR = "ems"
SIMULATION_DIR = os.path.join(BASE_DIR, "simulation")
GEOMETRY_DIR = os.path.join(BASE_DIR, "geometry")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOT_STYLE = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "antmicro.mplstyle"
)
BORDER_THICKNESS = 100
VIA_POLYGON = 8

STACKUP_FORMAT_VERSION = "1.0"
CONFIG_FORMAT_VERSION = "1.0"
