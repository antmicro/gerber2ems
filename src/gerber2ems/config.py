"""Contains classes that describe the configuration."""

from __future__ import annotations

import sys
import logging
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum
from math import sqrt
from serde import serde, field, coerce, from_dict
from serde.json import to_json
from pathlib import Path
import json
from argparse import Namespace
import csv

from gerber2ems.constants import CONFIG_FORMAT_VERSION, UNIT_MULTIPLIER, BASE_UNIT, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)
port_count: int = 0


@serde(type_check=coerce)
class PortConfig:
    """Class representing and parsing port config."""

    name: str = field(default="Unnamed")
    position: Optional[Tuple[float, float]] = field(default=None, skip=True)
    direction: Optional[float] = field(default=None, skip=True)
    width: float = field(default=200)
    length: float = field(default=1000)
    impedance: float = field(default=50)
    layer: int = field(default=0)
    plane: int = field(default=1)
    dB_margin: float = field(default=-15)  # noqa: N815
    excite: bool = field(default=False)

    @staticmethod
    def with_name(p: str) -> PortConfig:
        """Create PortConfig with specified name."""
        pc = PortConfig()
        pc.name = p
        return pc


@serde(type_check=coerce)
class DifferentialPairConfig:
    """Class representing and parsing differential pair config."""

    start_p: int = field(default=0)
    stop_p: int = field(default=1)
    start_n: int = field(default=2)
    stop_n: int = field(default=3)
    name: Optional[str] = field(default=None)
    nets: List[str] = field(default_factory=list)
    correct: bool = field(default=True, skip=True)

    def __post_init__(self) -> None:
        """Validate trace config."""
        global port_count
        if self.name is None:
            self.name = f"{self.start_p}_{self.stop_p}_{self.start_n}_{self.stop_n}"
        for f in ["start_p", "stop_p", "start_n", "stop_n"]:
            pn = getattr(self, f)
            if pn >= port_count:
                logger.warning(f"Differential pair {self.name} is defined to use not existing port number {pn} as {f}")
                self.correct = False


@serde(type_check=coerce)
class SingleEndedConfig:
    """Class representing and parsing single-ended config."""

    start: int = field(default=0)
    stop: int = field(default=1)
    name: Optional[str] = field(default=None)
    nets: List[str] = field(default_factory=list)
    correct: bool = field(default=True, skip=True)

    def __post_init__(self) -> None:
        """Validate trace config."""
        global port_count
        if self.name is None:
            self.name = f"{self.start}_{self.stop}"
        for f in ["start", "stop"]:
            pn = getattr(self, f)
            if pn >= port_count:
                logger.warning(f"Trace {self.name} is defined to use not existing port number {pn} as {f}")
                self.correct = False


class LayerConfig:
    """Class representing and parsing layer config."""

    def __init__(self, config: Any) -> None:
        """Initialize LayerConfig based on passed json object."""
        self.kind = self.parse_kind(config["type"])
        self.thickness = 0
        self.name = config["name"]
        if config["thickness"] is not None:
            self.thickness = config["thickness"] / 1000 / BASE_UNIT * UNIT_MULTIPLIER
        if self.kind == LayerKind.METAL:
            self.file = config["name"].replace(".", "_")
        elif self.kind == LayerKind.SUBSTRATE:
            self.epsilon = config["epsilon"]

    def __repr__(self) -> str:
        """Get human-readable string describing layer."""
        return f"Layer kind:{self.kind} thickness: {self.thickness}"

    @staticmethod
    def parse_kind(kind: str) -> LayerKind:
        """Parse type name to enum."""
        if kind in ["core", "prepreg"]:
            return LayerKind.SUBSTRATE
        if kind == "copper":
            return LayerKind.METAL
        return LayerKind.OTHER


class LayerKind(Enum):
    """Enum describing layer type."""

    SUBSTRATE = 1
    METAL = 2
    OTHER = 3


@serde(type_check=coerce)
class Frequency:
    """Frequency config."""

    start: float = field(default=1e6)
    stop: float = field(default=6e9)


@serde(type_check=coerce)
class Via:
    """Via config."""

    plating_thickness: float = field(default=50)
    filling_epsilon: float = field(default=1)


@serde(type_check=coerce)
class Margin:
    """Margin config (how far outside area of interest should grid span)."""

    xy: float = field(default=1000)
    z: float = field(default=1000)
    from_trace: bool = field(default=True)


@serde(type_check=coerce)
class CellRatio:
    """Cell Ratio config (Optimal scaling between neighboring grid cell sizes)."""

    xy: float = field(default=1.2)
    z: float = field(default=1.5)


@serde(type_check=coerce)
class Grid:
    """Grid generation config (configures simulation grid density)."""

    inter_layers: int = field(default=4)
    optimal: float = field(default=50)
    diagonal: float = field(default=50)
    perpendicular: float = field(default=200)
    max: float = field(default=500)
    margin: Margin = field(default_factory=Margin)
    cell_ratio: CellRatio = field(default_factory=CellRatio)


@serde(type_check=coerce)
class _Config:
    """Main config class."""

    format_version: str = field(default=CONFIG_FORMAT_VERSION)
    ports: List[PortConfig] = field(default_factory=list)
    frequency: Frequency = field(default_factory=Frequency)
    max_steps: int = field(default=100e3)
    pixel_size: int = field(default=5.0)
    via: Via = field(default_factory=Via)
    grid: Grid = field(default_factory=Grid)
    traces: List[SingleEndedConfig] = field(default_factory=list)
    diff_pairs: List[DifferentialPairConfig] = field(default_factory=list, rename="differential_pairs")

    pcb_width: int = field(default=0, skip=True)
    pcb_height: int = field(default=0, skip=True)
    layers: List[LayerConfig] = field(default_factory=list, skip=True)
    arguments: Namespace = field(
        default_factory=Namespace, skip=True, deserializer=lambda _: None, serializer=lambda _: None
    )

    def __post_init__(self) -> None:
        """Validate grid setting."""
        min_wavelength = 3e8 * 1e6 / sqrt(4.13) / self.frequency.stop  # min wavelength (in microns)
        self.grid.max = min(int(min_wavelength / 10), self.grid.max)
        self.grid.perpendicular = min(self.grid.perpendicular, self.grid.max)
        self.grid.diagonal = min(self.grid.diagonal, self.grid.perpendicular)
        self.grid.optimal = min(self.grid.diagonal, self.grid.optimal)
        self.format_version = CONFIG_FORMAT_VERSION

    def _apply_unit_multiplier(self) -> None:
        self.grid.max *= UNIT_MULTIPLIER
        self.grid.diagonal *= UNIT_MULTIPLIER
        self.grid.optimal *= UNIT_MULTIPLIER
        self.grid.perpendicular *= UNIT_MULTIPLIER
        self.grid.margin.xy *= UNIT_MULTIPLIER
        self.grid.margin.z *= UNIT_MULTIPLIER
        self.via.plating_thickness *= UNIT_MULTIPLIER

    def _correct_trace_config(self) -> None:
        for trace in self.traces:
            if not self.ports[trace.start].excite and self.ports[trace.start].excite:
                trace.start, trace.stop = trace.stop, trace.start
        for pair in self.diff_pairs:
            if not self.ports[pair.start_p].excite and self.ports[pair.start_p].excite:
                pair.start_p, pair.stop_p = pair.stop_p, pair.start_p
            if not self.ports[pair.start_n].excite and self.ports[pair.start_n].excite:
                pair.start_n, pair.stop_n = pair.stop_n, pair.start_n


class Config:
    """Config validation and parsing singleton class."""

    _instance = None
    _config: _Config

    def __new__(cls, *args: Any) -> Config:
        """Get already existing instance of Config or create new if it does not exist."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_config(cls, cfg: _Config) -> None:
        """Initialize inner config."""
        Config()._config = cfg

    @classmethod
    def load(cls, args: Namespace) -> None:
        """Load config file (default: simulation.json)."""
        global port_count
        if cls._instance is None:
            cls._instance = Config()

        logger.info("Parsing config")
        cfg_path_s = args.config
        if cfg_path_s is None:
            cfg_path_s = DEFAULT_CONFIG_PATH
        cfg_path = Path(cfg_path_s).absolute()
        json_cfg = get_cfg_json(cfg_path, args.update_config)

        version = json_cfg.get("format_version", None)
        if is_cfg_version_invalid(version):
            logger.error("Config format (%s) is not supported (supported: %s)", version, CONFIG_FORMAT_VERSION)
            sys.exit(1)

        port_count = len(json_cfg["ports"])
        cls._instance._config = from_dict(_Config, json_cfg)
        cls._instance._config.arguments = args

        if args.update_config:
            with cfg_path.open(mode="w", encoding="utf-8") as file:
                file.write(to_json(cls._instance._config, indent=4))

        for port in cls._instance._config.ports:
            port.width *= UNIT_MULTIPLIER
            port.length *= UNIT_MULTIPLIER
        cls._instance._config._apply_unit_multiplier()

    def __getattr__(self, name: str) -> Any:
        """Get value of field from internal config structure."""
        return getattr(self._config, name)

    @classmethod
    def load_stackup(cls, stackup: Dict[str, Any]) -> None:
        """Load stackup from json object."""
        if not cls._instance:
            return
        layers = []
        for layer in stackup["layers"]:
            layers.append(LayerConfig(layer))
        cls._instance._config.layers = list(
            filter(
                lambda layer: layer.kind in [LayerKind.METAL, LayerKind.SUBSTRATE],
                layers,
            )
        )

    def get_substrates(self) -> List[LayerConfig]:
        """Return substrate layers configs."""
        return list(filter(lambda layer: layer.kind == LayerKind.SUBSTRATE, self.layers))

    def get_metals(self) -> List[LayerConfig]:
        """Return metals layers configs."""
        return list(filter(lambda layer: layer.kind == LayerKind.METAL, self.layers))


def is_cfg_version_invalid(version: str | None) -> bool:
    """Check if config format version is supported."""
    return (
        version is None
        or not version.split(".")[0] == CONFIG_FORMAT_VERSION.split(".")[0]
        or version.split(".")[1] > CONFIG_FORMAT_VERSION.split(".")[1]
    )


def get_cfg_json(cfg_path: Path, update_config: bool) -> Dict[str, Any]:
    """Read config file and load it to dictionary.

    If update_config switch is enabled and there is no config file at provided path, returns config stub dictionary.
    """
    logger.info(f"Loading config from {cfg_path}")
    if not cfg_path.is_file() and update_config:
        cfg_path.touch()

    if not cfg_path.is_file():
        logger.error("Config file doesn't exist: %s", cfg_path)
        sys.exit(1)

    with cfg_path.open(mode="r", encoding="utf-8") as file:
        try:
            json_cfg = json.load(file)
        except json.JSONDecodeError as error:
            logger.error("JSON decoding failed at %d:%d: %s", error.lineno, error.colno, error.msg)
            sys.exit(1)

    if len(json_cfg) == 0:
        json_cfg = {"format_version": CONFIG_FORMAT_VERSION}

    if not json_cfg.get("ports", []):
        ports_pnp: List[Tuple[int, Tuple[float, float], float]] = []
        for filename in (Path.cwd() / "fab").glob("*pos.csv"):
            ports_pnp += get_ports_from_file(filename)
        ports = [PortConfig.with_name(str(p[0])) for p in ports_pnp]
        ports[-1].excite = True
        if len(ports) >= 4:
            ports[-3].excite = True
        json_cfg["ports"] = ports
        if len(ports) in [2, 3]:
            json_cfg["traces"] = [
                {
                    "start": ports_pnp[-1][0] - 1,
                    "stop": ports_pnp[-2][0] - 1,
                }
            ]
        if len(ports) >= 4:
            json_cfg["differential_pairs"] = [
                {
                    "start_p": ports_pnp[-1][0] - 1,
                    "stop_p": ports_pnp[-2][0] - 1,
                    "start_n": ports_pnp[-3][0] - 1,
                    "stop_n": ports_pnp[-4][0] - 1,
                }
            ]
    return json_cfg


def get_ports_from_file(filename: Path) -> List[Tuple[int, Tuple[float, float], float]]:
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
                        (
                            float(row[3]) / 1000 / BASE_UNIT * UNIT_MULTIPLIER,
                            float(row[4]) / 1000 / BASE_UNIT * UNIT_MULTIPLIER,
                        ),
                        float(row[5]),
                    )
                )
                logging.debug("Found port #%i position in pos file", number)

    return ports
