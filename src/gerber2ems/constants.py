"""Module containing constans used in the app."""

import os

MULTIPLIER = 10
BASE_UNIT = 1e-6  # Length units used in the whole script are microns
BASE_DIR = "ems"  # Name of the directory that outputs will be stored in
SIMULATION_DIR = os.path.join(BASE_DIR, "simulation")
GEOMETRY_DIR = os.path.join(BASE_DIR, "geometry")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOT_STYLE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "antmicro.mplstyle")
DEFAULT_CONFIG_PATH = "./simulation.json"

# Via geometry is approximated using n-sided right prism
VIA_POLYGON = 12

STACKUP_FORMAT_VERSION = "1.0"
CONFIG_FORMAT_VERSION = "1.2"
