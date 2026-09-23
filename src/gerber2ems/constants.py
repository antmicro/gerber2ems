"""Module containing constants used in the app."""

from pathlib import Path

UNIT_MULTIPLIER = 10
BASE_UNIT = 1e-6  # Length units used in the whole script are microns
BASE_DIR = Path.cwd() / "ems"  # Name of the directory that outputs will be stored in
SIMULATION_DIR = BASE_DIR / "simulation"
GEOMETRY_DIR = BASE_DIR / "geometry"
GEOMETRY_FILE = GEOMETRY_DIR / "geometry.xml"
RESULTS_DIR = BASE_DIR / "results"
PLOT_STYLE = Path(__file__).parent.absolute() / "antmicro.mplstyle"
DEFAULT_CONFIG_PATH = "./simulation.json"

# Via geometry is approximated using n-sided right prism
VIA_POLYGON = 12

STACKUP_FORMAT_VERSION = "1.0"
CONFIG_FORMAT_VERSION = "1.3"
