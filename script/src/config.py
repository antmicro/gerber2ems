"""Contains classes that describe the configuration."""
from __future__ import annotations

import sys
import logging
from typing import Any, List, Optional, Union, Tuple, Dict
from enum import Enum
from constants import CONFIG_FORMAT_VERSION, UNIT

logger = logging.getLogger(__name__)


class PortConfig:
    """Class representing and parsing port config."""

    def __init__(self, config: Any) -> None:
        """Initialize PortConfig based on passed json object."""
        self.name: str = get(config, ["name"], str, "Unnamed")
        self.position: Union[Tuple[float, float], None] = None
        self.direction: Union[float, None] = None
        self.width = get(config, ["width"], (float, int))
        self.length = get(config, ["length"], (float, int), 1000)
        self.impedance = get(config, ["impedance"], (float, int), 50)
        self.layer = get(config, ["layer"], int)
        self.plane = get(config, ["plane"], int)
        self.dB_margin = get(config, ["dB_margin"], (float, int), -15)
        self.excite = get(config, ["excite"], bool, False)


class LayerConfig:
    """Class representing and parsing layer config."""

    def __init__(self, config: Any) -> None:
        """Initialize LayerConfig based on passed json object."""
        self.kind = self.parse_kind(config["type"])
        self.thickness = 0
        if config["thickness"] is not None:
            self.thickness = config["thickness"] / 1000 / UNIT
        if self.kind == LayerKind.METAL:
            self.file = config["name"].replace(".", "_")
        elif self.kind == LayerKind.SUBSTRATE:
            self.epsilon = config["epsilon"]

    def __repr__(self):
        """Get human-readable string describing layer."""
        return f"Layer kind:{self.kind} thickness: {self.thickness}"

    @staticmethod
    def parse_kind(kind: str):
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


def get(
    config: Any,
    path: List[Union[str, int]],
    kind: Union[type, Tuple[type, ...]],
    default=None,
):
    """Gracefully look for value in object."""
    for name in path:
        if isinstance(config, Dict) and name in config:
            config = config[name]
        elif isinstance(name, int) and isinstance(config, List) and name < len(config):
            config = config[name]
        elif default is None:
            logger.error("No field %s found in config", path)
            sys.exit(1)
        else:
            logger.warning(
                "No field %s found in config. Using default: %s", path, str(default)
            )
            return default
    if isinstance(config, kind):
        return config
    if default is None:
        logger.error(
            "Field %s found in config has incorrect type %s (correct is %s)",
            path,
            type(config),
            kind,
        )
        sys.exit(1)
    else:
        logger.warning(
            "Field %s found in config has incorrect type %s (correct is %s). Using default: %s",
            path,
            type(config),
            kind,
            str(default),
        )
        return default


class Config:
    """Class representing and parsing config."""

    _instance: Optional[Config] = None

    @classmethod
    def get(cls) -> Config:
        """Return already instantiated config."""
        if cls._instance is not None:
            return cls._instance

        logger.error("Config hasn't been instantiated. Exiting")
        sys.exit(1)

    def __init__(self, json: Any, args: Any) -> None:
        """Initialize Config based on passed json object."""
        if self.__class__._instance is not None:
            logger.warning(
                "Config has already beed instatiated. Use Config.get() to get the instance. Skipping"
            )
            return

        logger.info("Parsing config")
        version = get(json, ["format_version"], str)
        if (
            version is None
            or not version.split(".")[0]
            == CONFIG_FORMAT_VERSION.split(".", maxsplit=1)[0]
            or version.split(".")[1] < CONFIG_FORMAT_VERSION.split(".", maxsplit=1)[1]
        ):
            logger.error(
                "Config format (%s) is not supported (supported: %s)",
                version,
                CONFIG_FORMAT_VERSION,
            )
            sys.exit()

        self.start_frequency = get(json, ["frequency", "start"], (float, int), 500e3)
        self.stop_frequency = get(json, ["frequency", "stop"], (float, int), 10e6)
        self.max_steps = get(json, ["max_steps"], (float, int), None)
        self.pcb_width: Union[float, None] = None
        self.pcb_height: Union[float, None] = None
        self.pcb_mesh_xy = get(json, ["mesh", "xy"], (float, int), 50)
        self.inter_copper_layers = get(json, ["mesh", "inter_layers"], int, 5)
        self.margin_xy = get(json, ["margin", "xy"], (float, int), 3000)
        self.margin_z = get(json, ["margin", "z"], (float, int), 3000)
        self.margin_mesh_xy = get(json, ["mesh", "margin", "xy"], (float, int), 200)
        self.margin_mesh_z = get(json, ["mesh", "margin", "z"], (float, int), 200)
        self.via_plating = get(json, ["via", "plating_thickness"], (int, float), 50)
        self.via_filling_epsilon = get(
            json, ["via", "filling_epsilon"], (int, float), 1
        )

        self.arguments = args

        ports = get(json, ["ports"], list)
        self.ports: List[PortConfig] = []
        for port in ports:
            self.ports.append(PortConfig(port))
        logger.debug("Found %d ports", len(self.ports))

        self.layers: List[LayerConfig] = []

        self.__class__._instance = self

    def load_stackup(self, stackup) -> None:
        """Load stackup from json object."""
        layers = []
        for layer in stackup["layers"]:
            layers.append(LayerConfig(layer))
        self.layers = list(
            filter(
                lambda layer: layer.kind in [LayerKind.METAL, LayerKind.SUBSTRATE],
                layers,
            )
        )

    def get_substrates(self) -> List[LayerConfig]:
        """Return substrate layers configs."""
        return list(
            filter(lambda layer: layer.kind == LayerKind.SUBSTRATE, self.layers)
        )

    def get_metals(self) -> List[LayerConfig]:
        """Return metals layers configs."""
        return list(filter(lambda layer: layer.kind == LayerKind.METAL, self.layers))
