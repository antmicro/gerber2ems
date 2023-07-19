import sys
import logging
from typing import Any, List, Union, Tuple

logger = logging.getLogger(__name__)


class Config:
    """Class representing and parsing config"""

    def __init__(self, config: Any) -> None:
        logger.info("Parsing config")
        self.start_frequency = get(config, ["frequency", "start"], (float, int), 500e3)
        self.stop_frequency = get(config, ["frequency", "stop"], (float, int), 10e6)
        self.max_steps = get(config, ["max_steps"], (float, int), None)
        self.epsilon = get(config, ["epsilon"], (float, int), 4.2)
        self.pcb_width = get(config, ["pcb", "dimensions", "width"], (float, int))
        self.pcb_height = get(config, ["pcb", "dimensions", "height"], (float, int))
        self.pcb_thickness = get(
            config, ["pcb", "dimensions", "thickness"], (float, int)
        )
        self.pcb_mesh_xy = get(config, ["pcb", "mesh", "xy"], (float, int), 50)
        self.pcb_mesh_z = get(config, ["pcb", "mesh", "z"], (float, int), 20)
        self.margin_xy = get(config, ["margin", "dimensions", "xy"], (float, int), 3000)
        self.margin_z = get(config, ["margin", "dimensions", "z"], (float, int), 3000)
        self.margin_mesh_xy = get(config, ["margin", "mesh", "xy"], (float, int), 200)
        self.margin_mesh_z = get(config, ["margin", "mesh", "z"], (float, int), 200)

        ports = get(config, ["ports"], list)
        self.ports = []
        for port in ports:
            self.ports.append(PortConfig(port))
        logger.debug("Found %d ports", len(self.ports))


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


def get(
    config: Any, path: List[str], kind: Union[type, Tuple[type, ...]], default=None
):
    """Gracefully look for value in object"""
    for name in path:
        if name in config:
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
