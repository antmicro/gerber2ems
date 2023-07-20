import sys
import logging
from typing import Any, List, Union, Tuple, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class PortConfig:
    """Class representing and parsing port config"""

    def __init__(self, config: Any) -> None:
        self.x_pos = get(
            config,
            ["position", "x"],
            (float, int),
        )
        self.y_pos = get(config, ["position", "y"], (float, int))
        self.direction = get(config, ["direction"], str)
        self.width = get(config, ["width"], (float, int))
        self.length = get(config, ["length"], (float, int), 1000)
        self.impedance = get(config, ["impedance"], (float, int), 50)
        self.layer = get(config, ["layer"], int)
        self.plane = get(config, ["plane"], int)


class LayerConfig:
    """Class representing and parsing layer config"""

    def __init__(self, config: Any, index: int) -> None:
        self.kind = self.parse_kind(get(config, ["pcb", "layers", index, "type"], str))
        self.thickness = get(
            config, ["pcb", "layers", index, "thickness"], (float, int)
        )
        if self.kind == LayerKind.METAL:
            self.file = get(config, ["pcb", "layers", index, "file"], str)
        elif self.kind == LayerKind.SUBSTRATE:
            self.epsilon = get(
                config, ["pcb", "layers", index, "epsilon"], (float, int)
            )

    @staticmethod
    def parse_kind(kind: str):
        """Parse type name to enum"""
        if kind == "substrate":
            return LayerKind.SUBSTRATE
        elif kind == "metal":
            return LayerKind.METAL
        else:
            logger.error("Layer type is invalid: %s", kind)
            sys.exit(1)


class LayerKind(Enum):
    """Enum describing layer type"""

    SUBSTRATE = 1
    METAL = 2


def get(
    config: Any,
    path: List[Union[str, int]],
    kind: Union[type, Tuple[type, ...]],
    default=None,
):
    """Gracefully look for value in object"""
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
    elif default is None:
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
    """Class representing and parsing config"""

    def __init__(self, config: Any) -> None:
        logger.info("Parsing config")
        self.start_frequency = get(config, ["frequency", "start"], (float, int), 500e3)
        self.stop_frequency = get(config, ["frequency", "stop"], (float, int), 10e6)
        self.max_steps = get(config, ["max_steps"], (float, int), None)
        self.pcb_width = get(config, ["pcb", "dimensions", "width"], (float, int))
        self.pcb_height = get(config, ["pcb", "dimensions", "height"], (float, int))
        self.pcb_mesh_xy = get(config, ["pcb", "mesh", "xy"], (float, int), 50)
        self.pcb_mesh_z = get(config, ["pcb", "mesh", "z"], (float, int), 20)
        self.margin_xy = get(config, ["margin", "dimensions", "xy"], (float, int), 3000)
        self.margin_z = get(config, ["margin", "dimensions", "z"], (float, int), 3000)
        self.margin_mesh_xy = get(config, ["margin", "mesh", "xy"], (float, int), 200)
        self.margin_mesh_z = get(config, ["margin", "mesh", "z"], (float, int), 200)
        self.via_plating = get(config, ["via_plating"], (int, float), 50)

        ports = get(config, ["ports"], list)
        self.ports = []
        for port in ports:
            self.ports.append(PortConfig(port))
        logger.debug("Found %d ports", len(self.ports))

        layers = get(config, ["pcb", "layers"], list)
        self.layers: List[LayerConfig] = []
        for i, _ in enumerate(layers):
            self.layers.append(LayerConfig(config, i))

    def get_substrates(self) -> List[LayerConfig]:
        """Returns substrate layers configs"""
        return list(filter(lambda l: l.kind == LayerKind.SUBSTRATE, self.layers))

    def get_metals(self) -> List[LayerConfig]:
        """Returns metals layers configs"""
        return list(filter(lambda l: l.kind == LayerKind.METAL, self.layers))
