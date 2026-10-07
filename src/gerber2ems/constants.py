"""Module containing constants used in the app."""

from pathlib import Path

UNIT_MULTIPLIER = 10
BASE_UNIT = 1e-6  # Length units used in the whole script are microns
FAB_DIR = Path.cwd() / "fab"  # Name of the directory where input/production files are located
BASE_DIR = Path.cwd() / "ems"  # Name of the directory that outputs will be stored in
SIMULATION_DIR = BASE_DIR / "simulation"
GEOMETRY_DIR = BASE_DIR / "geometry"
GEOMETRY_FILE = GEOMETRY_DIR / "geometry.xml"
RESULTS_DIR = BASE_DIR / "results"
PLOT_STYLE = Path(__file__).parent.absolute() / "antmicro.mplstyle"
GUIDELINES_FILE = Path(__file__).parent.absolute() / "guidelines.yml"
DEFAULT_CONFIG_PATH = "./simulation.json"

# Via geometry is approximated using n-sided right prism
VIA_POLYGON = 12

STACKUP_FORMAT_VERSION = "1.0"
CONFIG_FORMAT_VERSION = "1.4"
"""`simulation.json` config file format specification revision

# Changelog:

## v1.4
* `dB_margin` -> `db_margin` (`dB_margin` support is kept as alias)

## v1.3
* Add `sim_length`
"""


class ObjectPriority:
    """Class defining CSXCAD priority of objects."""

    COOPER: int = 1
    VIA: int = 50
    VIA_FILLING: int = 51
    PORT_V2_REF_STITCH: int = 70
    PORT: int = 100

    @staticmethod
    def substrate(idx: int) -> int:
        """Priority for PCB substrate primitive."""
        return -1 - idx
