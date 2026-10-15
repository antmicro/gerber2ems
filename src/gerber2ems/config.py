"""Contains classes that describe the configuration."""

from __future__ import annotations

import csv
import json
import logging
import sys
from argparse import Namespace
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from serde import Untagged, coerce, field, from_dict, serde
from serde.json import to_json

from gerber2ems.common import PadProp, Point, mm_to_sim
from gerber2ems.constants import CONFIG_FORMAT_VERSION, FAB_DIR, GUIDELINES_FILE, UNIT_MULTIPLIER

logger = logging.getLogger(__name__)
port_count: int = 0


@serde(type_check=coerce, tagging=Untagged, skip_if_none=True)
@dataclass
class PortConfig:
    """Class representing and parsing port config."""

    name: str = field(default="Unnamed", alias=["number"])
    position: Point | None = field(default=None, skip=True)
    """[DEPRECATED: Use PortConfig.pad]"""
    direction: float | None = field(default=None, skip=True)
    width: float = 200
    """[DEPRECATED: Use PortConfig.pad]"""
    length: float = 1000
    impedance: float = 50
    layer: int = 0
    plane: int | None = None
    pad: PadProp | None = None
    reference_pad: Sequence[PadProp | str] = field(default_factory=list)
    """List of pads that are reference point for port (eg. ground pin of IC)
    Can be either `PadProp` or name of pad"""
    db_margin: float = field(default=-15, alias=["dB_margin"])
    excite: bool = False

    def __post_init__(self) -> None:
        """Resolve v2 port properties."""
        if self.is_port_v2():
            self.plane = None

    @staticmethod
    def with_name(p: str) -> PortConfig:
        """Create PortConfig with specified name."""
        pc = PortConfig()
        pc.name = p
        return pc

    def is_port_v2(self) -> bool:
        """Return true if port is V2 port (vertical, lumped port to virtual gnd patch)."""
        return bool(self.reference_pad)

    def get_ref_pads(self) -> list[PadProp]:
        """Return reference pad position & size."""
        for i in range(len(self.reference_pad)):
            pad = self.reference_pad[i]
            if isinstance(pad, str):
                f"{self.name} reference_pad {i} ({pad}) has not been resolved: run `gerber2ems -c`"
        return [pad.sim_pad() for pad in self.reference_pad if not isinstance(pad, str)]

    def get_pad(self) -> PadProp:
        """Return simulation pad position & size."""
        if self.pad:
            return self.pad.sim_pad()
        if self.position is None or self.direction is None:
            raise AssertionError("PortConfig position&direction are expected to be loaded before adding ports")
        angle = self.direction / 360 * 2 * pi
        w, h = self.width, self.length
        size = Point(w * round(cos(angle)) - h * round(sin(angle)), w * round(sin(angle)) + h * round(cos(angle)))
        pos = Point(self.position.x + (self.width / 2) * cos(angle), self.position.y + (self.width / 2) * sin(angle))
        self.pad = PadProp(pos.sim_to_mm(), size.sim_to_mm())

        return self.pad.sim_pad()

    def update_from_pad(self: PortConfig, pad: PadProp, close_gnd_pads: list[PadProp], default_name: str) -> None:
        """Update port properties using pad data."""
        if not self.pad:
            pad.size.scale(0.7)  # smaller port than pad to fit into circular pads
            self.pad = pad.mm_pad()
            if self.length == 10000 and self.width == 2000:
                self.length = 0
                self.width = 0
            w = self.width
            if w:
                self.pad.size = Point(w, w).sim_to_mm()

        self.length = self.length or 3600
        self.plane = None
        self.reference_pad = close_gnd_pads
        self.name = default_name if self.name in ("Unnamed", "", None) else self.name


@serde(type_check=coerce)
class Interface:
    """Interface config."""

    name: str | None = None
    min_frequency: float = field(default=0)
    max_frequency: float = field(default=float("nan"))
    se_impedance_nominal: float = field(default=float("nan"))
    se_impedance_tolerance: float = field(default=float("nan"))
    diff_impedance_nominal: float = field(default=float("nan"))
    diff_impedance_tolerance: float = field(default=float("nan"))
    skew_intra_pair: float = field(default=float("nan"))
    skew_inter_pair: float = field(default=float("nan"))
    skew_clock_data: float = field(default=float("nan"))
    rise_fall_time: float = field(default=float("nan"))


@serde(type_check=coerce)
class DifferentialPairConfig:
    """Class representing and parsing differential pair config."""

    start_p: int = field(default=0)
    stop_p: int = field(default=1)
    start_n: int = field(default=2)
    stop_n: int = field(default=3)
    name: str | None = field(default=None)
    nets: list[str] = field(default_factory=list)
    correct: bool = field(default=True, skip=True)
    interface: Interface = field(default_factory=Interface)

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
    name: str | None = field(default=None)
    nets: list[str] = field(default_factory=list)
    correct: bool = field(default=True, skip=True)
    interface: Interface = field(default_factory=Interface)

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
            self.thickness = mm_to_sim(config["thickness"])
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

    xy: float = field(default=1500)
    z: float = field(default=2000)
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
    ports: list[PortConfig] = field(default_factory=list)
    frequency: Frequency = field(default_factory=Frequency)
    max_steps: int | None = field(default=None)
    sim_length: float | None = field(default=None)
    pixel_size: int = field(default=5.0)
    via: Via = field(default_factory=Via)
    grid: Grid = field(default_factory=Grid)
    traces: list[SingleEndedConfig] = field(default_factory=list)
    diff_pairs: list[DifferentialPairConfig] = field(default_factory=list, rename="differential_pairs")

    pcb_width: int = field(default=0, skip=True)
    pcb_height: int = field(default=0, skip=True)
    layers: list[LayerConfig] = field(default_factory=list, skip=True)
    trace_under_test: str | None = None
    arguments: Namespace = field(
        default_factory=Namespace, skip=True, deserializer=lambda _: None, serializer=lambda _: None
    )
    unit_multiplier_applied: float = field(default=1, skip=True)

    def __post_init__(self) -> None:
        """Validate grid setting."""
        min_wavelength = 3e8 * 1e6 / sqrt(4.13) / self.frequency.stop  # min wavelength (in microns)
        self.grid.max = min(int(min_wavelength / 10), self.grid.max)
        self.grid.perpendicular = min(self.grid.perpendicular, self.grid.max)
        self.grid.diagonal = min(self.grid.diagonal, self.grid.perpendicular)
        self.grid.optimal = min(self.grid.diagonal, self.grid.optimal)
        self.format_version = CONFIG_FORMAT_VERSION
        if not self.max_steps and not self.sim_length:
            self.sim_length = 5

    def _apply_unit_multiplier(self, restore: bool = False) -> None:
        mul = 1 / self.unit_multiplier_applied if restore else UNIT_MULTIPLIER
        self.grid.max *= mul
        self.grid.diagonal *= mul
        self.grid.optimal *= mul
        self.grid.perpendicular *= mul
        self.grid.margin.xy *= mul
        self.grid.margin.z *= mul
        self.via.plating_thickness *= mul
        for port in self.ports:
            port.width *= mul
            port.length *= mul
        self.unit_multiplier_applied *= mul

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
    _path: Path | None = None

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
    def load_guidelines(cls) -> None:
        """Load interface parameters from guidelines."""
        if cls._instance is None:
            cls._instance = Config()

        with open(GUIDELINES_FILE, "r") as file:
            try:
                data = yaml.safe_load(file)
            except data.YAMLError as error:
                logger.error("Yaml parsing failed at %d:%d: %s", error.lineno, error.colno, error.msg)
                sys.exit(1)

            interfaces: list[Interface] = []
            for trace in cls._instance._config.traces:
                interfaces.append(trace.interface)

            for diff_pair in cls._instance._config.diff_pairs:
                interfaces.append(diff_pair.interface)

            for interface in interfaces:

                if interface.name is None:
                    logger.debug("Interface name is undefined")
                    continue

                interface_data = data["interfaces"].get(interface.name)
                if interface_data is None:
                    logger.warning("Interface not found in the interface list")
                    available_interfaces = "Available interface names:"
                    for available_interface in data["interfaces"]:
                        if not available_interface.startswith("."):
                            available_interfaces += "\n" + "\t" * 5 + available_interface
                    logger.debug(available_interfaces)
                    continue

                if interface_data.get("rise-fall-time") is not None:
                    if np.isnan(interface.rise_fall_time):
                        interface.rise_fall_time = interface_data.get("rise-fall-time")
                        interface.max_frequency = 0.35 / interface.rise_fall_time
                        logger.info(f"Knee frequency: {np.ceil(interface.max_frequency) / 1e6} MHz")
                if np.isnan(interface.max_frequency):
                    interface.max_frequency = interface_data.get("frequency")
                    logger.info(f"Defined frequency: {float(interface.max_frequency) / 1e6} MHz")

                impedance = interface_data.get("impedance")
                se_impedance = impedance.get("se")
                if se_impedance is not None:
                    if np.isnan(interface.se_impedance_nominal):
                        interface.se_impedance_nominal = se_impedance.get("nominal")
                    if np.isnan(interface.se_impedance_tolerance):
                        interface.se_impedance_tolerance = se_impedance.get("tolerance")
                diff_impedance = impedance.get("diff")
                if diff_impedance is not None:
                    if np.isnan(interface.diff_impedance_nominal):
                        interface.diff_impedance_nominal = diff_impedance.get("nominal")
                    if np.isnan(interface.diff_impedance_tolerance):
                        interface.diff_impedance_tolerance = diff_impedance.get("tolerance")

                skew = interface_data.get("skew")
                if skew is not None:
                    if np.isnan(interface.skew_intra_pair):
                        interface.skew_intra_pair = skew.get("intra-pair")
                    if np.isnan(interface.skew_inter_pair):
                        interface.skew_inter_pair = skew.get("inter-pair")
                    if np.isnan(interface.skew_clock_data):
                        interface.skew_clock_data = skew.get("clock-data")

        for interface in interfaces:
            interface_debug = ""

            for parameter in (
                "rise_fall_time",
                "max_frequency",
                "se_impedance_nominal",
                "se_impedance_tolerance",
                "diff_impedance_nominal",
                "diff_impedance_tolerance",
                "skew_intra_pair",
                "skew_inter_pair",
                "skew_clock_data",
            ):
                value = getattr(interface, parameter)

                if str(value) != "nan" and value:
                    interface_debug += f"\t\t {parameter + ':':<25} {value}\n"

            if interface_debug:
                logger.info(
                    "Interface: %s\n%s",
                    interface.name,
                    interface_debug,
                )

    @classmethod
    def load(cls, cfg_path: Path, args: Namespace) -> None:
        """Load config file."""
        global port_count
        if cls._instance is None:
            cls._instance = Config()

        logger.info("Parsing config")
        json_cfg = get_cfg_json(cfg_path, args.update_config)

        version = json_cfg.get("format_version", None)
        if is_cfg_version_invalid(version):
            logger.error("Config format (%s) is not supported (supported: %s)", version, CONFIG_FORMAT_VERSION)
            sys.exit(1)

        port_count = len(json_cfg["ports"])
        cls._instance._config = from_dict(_Config, json_cfg)
        cls._instance._config.arguments = args
        cls._instance._config._apply_unit_multiplier()

        cls.load_guidelines()

    def save(self) -> None:
        """Save config file."""
        if not self._path:
            raise AssertionError("Config has no filesystem path defined!")
        cfg = deepcopy(self._config)
        cfg._apply_unit_multiplier(True)
        with self._path.open(mode="w", encoding="utf-8") as file:
            file.write(to_json(cfg, indent=4))

    def __getattr__(self, name: str) -> Any:
        """Get value of field from internal config structure."""
        return getattr(self._config, name)

    @classmethod
    def load_stackup(cls, stackup: dict[str, Any]) -> None:
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

    def get_substrates(self) -> list[LayerConfig]:
        """Return substrate layers configs."""
        return list(filter(lambda layer: layer.kind == LayerKind.SUBSTRATE, self.layers))

    def get_metals(self) -> list[LayerConfig]:
        """Return metals layers configs."""
        return list(filter(lambda layer: layer.kind == LayerKind.METAL, self.layers))


def is_cfg_version_invalid(version: str | None) -> bool:
    """Check if config format version is supported."""
    return (
        version is None
        or not version.split(".")[0] == CONFIG_FORMAT_VERSION.split(".")[0]
        or version.split(".")[1] > CONFIG_FORMAT_VERSION.split(".")[1]
    )


def get_cfg_json(cfg_path: Path, update_config: bool) -> dict[str, Any]:
    """Read config file and load it to dictionary.

    If update_config switch is enabled and there is no config file at provided path, returns config stub dictionary.
    """
    logger.info(f"Loading config from {cfg_path}")
    if not cfg_path.is_file() and update_config:
        cfg_path.touch()

    if not cfg_path.is_file():
        logger.error("Config file doesn't exist: %s", cfg_path)
        sys.exit(1)

    json_cfg: dict[str, Any] = {}
    with cfg_path.open(mode="r", encoding="utf-8") as file:
        try:
            json_cfg = json.load(file)
        except json.JSONDecodeError as error:
            logger.error("JSON decoding failed at %d:%d: %s", error.lineno, error.colno, error.msg)
            sys.exit(1)

    if len(json_cfg) == 0:
        json_cfg = {"format_version": CONFIG_FORMAT_VERSION}

    if not json_cfg.get("ports", []):
        ports_numbers = [p[0] for p in get_ports_from_all_pos_files()]
        ports = [PortConfig.with_name(str(p)) for p in ports_numbers]
        ports[-1].excite = True
        if len(ports) >= 4:
            ports[-3].excite = True
        json_cfg["ports"] = ports
        if len(ports) in [2, 3]:
            json_cfg["traces"] = [{"start": ports_numbers[-1], "stop": ports_numbers[-2]}]
        if len(ports) >= 4:
            json_cfg["differential_pairs"] = [
                {
                    "start_p": ports_numbers[-1],
                    "stop_p": ports_numbers[-2],
                    "start_n": ports_numbers[-3],
                    "stop_n": ports_numbers[-4],
                }
            ]
    return json_cfg


def get_ports_from_file(filename: Path) -> list[tuple[int, Point, float]]:
    """Parse pnp CSV file and return all ports in format (number, (x, y), direction)."""
    ports = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers
        for row in reader:
            if "Simulation_Port" in row[2] or "Simulation-Port" in row[2]:
                number = int(row[0][2:])
                ports.append(
                    (
                        number - 1,
                        Point(mm_to_sim(float(row[3])), mm_to_sim(float(row[4]))),
                        float(row[5]),
                    )
                )
                logging.debug("Found port #%i position in pos file", number)

    return ports


def get_ports_from_all_pos_files(path: Path = FAB_DIR) -> list[tuple[int, Point, float]]:
    """Parse all pnp files and return all ports in format (number, (x, y), direction)."""
    ports_pnp = []
    for filename in path.glob("*pos.csv"):
        ports_pnp += get_ports_from_file(filename)
    return ports_pnp
