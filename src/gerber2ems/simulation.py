"""Module containing Simulation class used for interacting with openEMS."""

import logging
import os
import re
import sys
import math
from typing import Tuple, List, Any
from multiprocessing import Pool

import CSXCAD
import openEMS
import numpy as np

from gerber2ems.config import Config, PortConfig, LayerKind
from gerber2ems.constants import (
    UNIT_MULTIPLIER,
    BASE_UNIT,
    SIMULATION_DIR,
    GEOMETRY_DIR,
    VIA_POLYGON,
)
import gerber2ems.importer as importer
from gerber2ems.grid_gen import GridGenerator
from gerber2ems.gerber_io import Pad, ApertureRect, Aperture, Position

logger = logging.getLogger(__name__)
cfg = Config()


class Simulation:
    """Class used for interacting with openEMS."""

    def __init__(self) -> None:
        """Initialize simulation object."""
        self.csx = CSXCAD.ContinuousStructure()
        self.fdtd = openEMS.openEMS(NrTS=cfg.max_steps)
        self.fdtd.SetCSX(self.csx)
        self.grid = self.csx.GetGrid()
        self.grid.SetDeltaUnit(BASE_UNIT / UNIT_MULTIPLIER)

        self.ports: List[openEMS.ports.MSLPort] = []

        # Separate metal materials for easier switching of layers
        self.gerber_materials: List[Any] = []
        self.substrate_materials: List[Any] = []
        self.plane_material = self.csx.AddMetal("Plane")
        self.via_material = self.csx.AddMetal("Via")
        self.via_filling_material = self.csx.AddMaterial("ViaFilling", epsilon=cfg.via.filling_epsilon)

    def create_materials(self) -> None:
        """Create materials required for simulation."""
        for i, _ in enumerate(cfg.get_metals()):
            self.gerber_materials.append(self.csx.AddMetal(f"Gerber_{i}"))
        for i, layer in enumerate(cfg.get_substrates()):
            self.substrate_materials.append(self.csx.AddMaterial(f"Substrate_{i}", epsilon=layer.epsilon))

    def print_grid_stats(self) -> None:
        """Print stats about grid: line count, minimal spacing, positions."""
        logger.debug("Grid x lines: %s", self.grid.GetLines("x"))
        logger.debug("Grid y lines: %s", self.grid.GetLines("y"))
        logger.debug("Grid z lines: %s", self.grid.GetLines("z"))

        def get_stat(lines: List[float]) -> List[float]:
            msize = math.inf
            mscale = -math.inf
            prev = lines[1]
            prev_size = abs((lines[0] - lines[1]))
            for line in lines:
                size = abs(prev - line)
                msize = min(size, msize)
                scale = prev_size / size
                scale = scale if scale > 1 else 1 / scale
                mscale = max(scale, mscale)
                prev = line
                prev_size = size
            return [msize, mscale]

        sx = get_stat(self.grid.GetLines("x"))
        sy = get_stat(self.grid.GetLines("y"))
        sz = get_stat(self.grid.GetLines("z"))
        xyz = [
            self.grid.GetQtyLines("x"),
            self.grid.GetQtyLines("y"),
            self.grid.GetQtyLines("z"),
        ]
        logger.info(
            "Grid line count, x: %d, y: %d z: %d. Total number of cells: ~%.2fM",
            xyz[0],
            xyz[1],
            xyz[2],
            xyz[0] * xyz[1] * xyz[2] / 1.0e6,
        )
        logger.info("Minimal cell size, x: %f, y: %f z: %f [um]", sx[0], sy[0], sz[0])
        logger.info("Max cell size ratio, x: %f, y: %f z: %f ", sx[1], sy[1], sz[1])

    def add_grid(self) -> None:
        """Add grid to simulation."""
        self.grid_gen = GridGenerator()
        self.add_port_grid()
        logger.info("Compiling grid")
        self.grid = self.grid_gen.generate(self.grid)

        self.print_grid_stats()

    def add_gerbers(self) -> None:
        """Add metal from all gerber files."""
        logger.info("Adding copper from gerber files")

        offset = 0
        index = 0
        contours = []
        with Pool(initargs=(cfg._config,), initializer=Config.set_config) as p:
            contours = p.map(
                importer.get_triangles,
                [lc.file + ".png" for lc in cfg.layers if lc.kind == LayerKind.METAL],
            )

        icontours = iter(contours)
        for layer in cfg.layers:
            if layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
            elif layer.kind == LayerKind.METAL:
                logger.info("Adding metal mesh for %s", layer.file)
                self.add_contours(next(icontours), offset, index)
                index += 1

    def add_contours(self, contours: np.ndarray, z_height: float, layer_index: int) -> None:
        """Add contours as flat polygons on specified z-height."""
        logger.debug("Adding contours on z=%f", z_height)
        for contour in contours:
            points: List[List[float]] = [[], []]
            for point in contour:
                # Half of the border thickness is subtracted as image is shifted by it
                points[0].append((point[1]))
                points[1].append(cfg.pcb_height - point[0])

            self.gerber_materials[layer_index].AddPolygon(points, "z", z_height, priority=1)

    def get_metal_layer_offset(self, index: int) -> float:
        """Get z offset of nth metal layer."""
        current_metal_index = -1
        offset = 0
        for layer in cfg.layers:
            if layer.kind == LayerKind.METAL:
                current_metal_index += 1
                if current_metal_index == index:
                    return offset
            elif layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
        logger.error("Hadn't found %dth metal layer", index)
        sys.exit(1)

    def add_port_grid(self) -> None:
        """Add grid around ports for simulation."""
        logger.info("Adding ports grid")

        for port_config in cfg.ports:
            if port_config.position is None or port_config.direction is None:
                logger.error("Port has no defined position or rotation, skipping")
                return
            angle = port_config.direction / 360 * 2 * math.pi
            ap = f"PORT{len(self.grid_gen.add_apertures) + 1}"
            (w, h) = ((port_config.width), (port_config.length))
            (width, height) = (
                w * round(math.cos(angle)) - h * round(math.sin(angle)),
                w * round(math.sin(angle)) + h * round(math.cos(angle)),
            )
            self.grid_gen.add_pads.append(
                Pad(
                    ap,
                    "PORT",
                    Position(
                        port_config.position[0] + self.grid_gen.xmin + width / 2,
                        port_config.position[1] + self.grid_gen.ymin,
                    ),
                )
            )
            self.grid_gen.add_apertures[ap] = Aperture(
                "",
                ApertureRect(width, height),
            )

    def add_msl_port(self, port_config: PortConfig, port_number: int, excite: bool = False) -> None:
        """Add microstripline port based on config."""
        logger.debug("Adding port number %d", len(self.ports))

        if port_config.position is None or port_config.direction is None:
            logger.error("Port has no defined position or rotation, skipping")
            return

        while port_config.direction < 0:
            port_config.direction += 360

        dir_map = {0: "y", 90: "x", 180: "y", 270: "x"}
        if int(port_config.direction) not in dir_map:
            logger.error("Ports rotation is not a multiple of 90 degrees which is not supported, skipping")
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)

        angle = port_config.direction / 360 * 2 * math.pi

        start = [
            round(port_config.position[0] - (port_config.width / 2) * round(math.cos(angle))),
            round(port_config.position[1] - (port_config.width / 2) * round(math.sin(angle))),
            round(start_z),
        ]
        stop = [
            round(
                port_config.position[0]
                + (port_config.width / 2) * round(math.cos(angle))
                - port_config.length * round(math.sin(angle))
            ),
            round(
                port_config.position[1]
                + (port_config.width / 2) * round(math.sin(angle))
                + port_config.length * round(math.cos(angle))
            ),
            round(stop_z),
        ]
        logger.debug("Adding port at start: %s end: %s", start, stop)

        port = self.fdtd.AddMSLPort(
            port_number,
            self.csx.AddMetal(f"Port_{port_number}"),
            start,
            stop,
            dir_map[int(port_config.direction)],
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=1 if excite else 0,
        )
        self.ports.append(port)

    def add_resistive_port(self, port_config: PortConfig, excite: bool = False) -> None:
        """Add resistive port based on config."""
        logger.debug("Adding port number %d", len(self.ports))

        if port_config.position is None or port_config.direction is None:
            logger.error("Port has no defined position or rotation, skipping")
            return

        dir_map = {0: "y", 90: "x", 180: "y", 270: "x"}
        if int(port_config.direction) not in dir_map:
            logger.error("Ports rotation is not a multiple of 90 degrees which is not supported, skipping")
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)

        angle = port_config.direction / 360 * 2 * math.pi

        start = [
            round(port_config.position[0] - (port_config.width / 2) * round(math.cos(angle))),
            round(port_config.position[1] - (port_config.width / 2) * round(math.sin(angle))),
            round(start_z),
        ]
        stop = [
            round(port_config.position[0] + (port_config.width / 2) * round(math.cos(angle))),
            round(port_config.position[1] - (port_config.width / 2) * round(math.sin(angle))),
            round(stop_z),
        ]
        logger.debug("Adding resistive port at start: %s end: %s", start, stop)

        port = self.fdtd.AddLumpedPort(
            len(self.ports),
            port_config.impedance,
            start,
            stop,
            "z",
            excite=1 if excite else 0,
            priority=100,
        )
        self.ports.append(port)

    def add_virtual_port(self, port_config: PortConfig) -> None:
        """Add virtual port for extracting sim data from files. Needed due to OpenEMS api desing."""
        logger.debug("Adding virtual port number %d", len(self.ports))
        for i in range(11):
            self.grid.AddLine("x", i)
            self.grid.AddLine("y", i)
        self.grid.AddLine("z", 0)
        self.grid.AddLine("z", 10)
        port = self.fdtd.AddMSLPort(
            len(self.ports),
            self.csx.AddMetal(f"VirtualPort_{len(self.ports)}"),
            [0, 0, 0],
            [10, 10, 10],
            "x",
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=0,
        )
        self.ports.append(port)

    def add_plane(self, z_height: float) -> None:
        """Add metal plane in whole bounding box of the PCB."""
        self.plane_material.AddBox(
            [0, 0, z_height],
            [cfg.pcb_width, cfg.pcb_height, z_height],
            priority=1,
        )

    def add_substrates(self) -> None:
        """Add substrate in whole bounding box of the PCB."""
        logger.info("Adding substrates")

        offset = 0
        for i, layer in enumerate(cfg.get_substrates()):
            self.substrate_materials[i].AddBox(
                [0, 0, offset],
                [
                    cfg.pcb_width,
                    cfg.pcb_height,
                    offset - layer.thickness,
                ],
                priority=-i - 1,
            )
            logger.debug("Added substrate from %f to %f", offset, offset - layer.thickness)
            offset -= layer.thickness

    def add_vias(self) -> None:
        """Add all vias from excellon file."""
        logger.info("Adding vias from excellon file")
        vias = importer.get_vias()
        for via in vias:
            self.add_via(via[0], via[1], via[2])

    def add_via(self, x_pos: float, y_pos: float, diameter: float) -> None:
        """Add via at specified position with specified diameter."""
        thickness = sum(layer.thickness for layer in cfg.get_substrates())

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON):
            x_coords.append(x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
            y_coords.append(y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
        self.via_filling_material.AddLinPoly([x_coords, y_coords], "z", -thickness, thickness, priority=51)

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON)[::-1]:
            x_coords.append(x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * (diameter / 2 + cfg.via.plating_thickness))
            y_coords.append(y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * (diameter / 2 + cfg.via.plating_thickness))
        self.via_material.AddLinPoly([x_coords, y_coords], "z", -thickness, thickness, priority=50)

    def add_single_dump_box(self, name: str, z: float) -> None:
        """Add electric field dump box in whole bounding box of the PCB at specified Z-position."""
        logger.debug("Adding dump box at %f", z)
        dump = self.csx.AddDump(name, sub_sampling=[1, 1, 1])
        start = [
            -cfg.grid.margin.xy,
            -cfg.grid.margin.xy,
            z,
        ]
        stop = [
            cfg.pcb_width + cfg.grid.margin.xy,
            cfg.pcb_height + cfg.grid.margin.xy,
            z,
        ]
        dump.AddBox(start, stop)

    def add_dump_boxes(self) -> None:
        """Add electric field dump box in whole bounding box of the PCB at half the thickness of each substrate."""
        if cfg.arguments.export_field is None:
            return
        logger.info("Adding field dump boxes")

        if len(cfg.arguments.export_field) == 0:
            cfg.arguments.export_field = ["outer", "cu-outer", "cu-inner", "substrate"]

        offset = 0
        metal_idx = 0
        metal_count = len(cfg.get_metals())
        for layer in cfg.layers:
            norm_name = (
                layer.name.replace(".", "_").replace("(", "_").replace(")", "_").replace(" ", "_").replace("/", "_")
            )
            if layer.kind == LayerKind.SUBSTRATE:
                if "substrate" in cfg.arguments.export_field:
                    height = offset - layer.thickness / 2
                    self.add_single_dump_box(f"e_field_{norm_name}", height)
                offset -= layer.thickness
            elif layer.kind == LayerKind.METAL:
                export_inner = "cu-inner" in cfg.arguments.export_field and metal_idx not in [0, metal_count]
                export_outer = "cu-outer" in cfg.arguments.export_field and metal_idx in [0, metal_count]
                if export_inner or export_outer:
                    self.add_single_dump_box(f"e_field_{norm_name}", offset)
                metal_idx += 1

        if "outer" in cfg.arguments.export_field:
            self.add_single_dump_box("e_field_top_over", 100)
            self.add_single_dump_box("e_field_bottom_over", offset - 100)

    def set_boundary_conditions(self, pml: bool = False) -> None:
        """Add boundary conditions. MUR for fast simulation, PML for more accurate."""
        if pml:
            logger.info("Adding perfectly matched layer boundary condition")
            self.fdtd.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"])
        else:
            logger.info("Adding MUR boundary condition")
            self.fdtd.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    def set_excitation(self) -> None:
        """Set gauss excitation according to config."""
        logger.debug(
            "Setting excitation to gaussian pulse from %f to %f",
            cfg.frequency.start,
            cfg.frequency.stop,
        )
        self.fdtd.SetGaussExcite(
            (cfg.frequency.start + cfg.frequency.stop) / 2,
            (cfg.frequency.stop - cfg.frequency.start) / 2,
        )

    def set_sinus_excitation(self, freq: float) -> None:
        """Set sinus excitation at specified frequency."""
        logger.debug("Setting excitation to sine at %f", freq)
        self.fdtd.SetSinusExcite(freq)

    def run(self, excited_port_number: int) -> None:
        """Execute simulation."""
        logger.info("Starting simulation")
        cwd = os.getcwd()
        self.fdtd.SetOverSampling(cfg.arguments.oversampling)
        self.fdtd.Run(os.path.join(os.getcwd(), SIMULATION_DIR, str(excited_port_number)))

        os.chdir(cwd)

    def save_geometry(self) -> None:
        """Save geometry to file."""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Saving geometry to %s", filename)
        self.csx.Write2XML(filename)

        # Replacing , with . for numerals in the file
        # (openEMS bug mitigation for locale that uses , as decimal separator)
        with open(filename, "r") as f:
            content = f.read()
        new_content = re.sub(r"([0-9]+),([0-9]+e)", r"\g<1>.\g<2>", content)
        with open(filename, "w") as f:
            f.write(new_content)

    def load_geometry(self) -> None:
        """Load geometry from file."""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Loading geometry from %s", filename)
        if not os.path.exists(filename):
            logger.error("Geometry file does not exist. Did you run geometry step?")
            sys.exit(1)
        self.csx.ReadFromXML(filename)
        self.grid = self.csx.GetGrid()

    def get_port_parameters(self, exindex: int, frequencies: np.ndarray) -> Tuple[List, List]:
        """Return reflected and incident power vs frequency for each port."""
        result_path = os.path.join(os.getcwd(), SIMULATION_DIR, str(exindex))

        incident: List[np.ndarray] = []
        reflected: List[np.ndarray] = []
        for index, port in enumerate(self.ports):
            try:
                port.CalcPort(result_path, frequencies)
                logger.debug("Found data for port %d", index)
            except IOError:
                logger.error("Port data files do not exist. Did you run simulation step?")
                sys.exit(1)
            incident.append(port.uf_inc)
            reflected.append(port.uf_ref)

        return (reflected, incident)

    def setup_ports(self, enabled_idx: int) -> None:
        """Set up ports excitation."""
        logger.info("Setting up ports")

        for prop in self.csx.GetAllProperties():
            if not isinstance(prop, CSXCAD.CSProperties.CSPropExcitation):
                continue
            pname = prop.GetName()
            idx = int(pname.removeprefix(pname.rstrip("0123456789")))
            if idx != enabled_idx:
                prop.SetExcitation([0, 0, 0])
            else:
                prop.SetExcitation([0, 0, 1])

    def add_ports(self) -> None:
        """Add ports for simulation."""
        logger.info("Adding ports")

        self.ports = []

        for index, port_config in enumerate(cfg.ports):
            self.add_msl_port(port_config, index, True)

    def add_virtual_ports(self) -> None:
        """Add virtual ports needed for data postprocessing due to openEMS api design."""
        logger.info("Adding virtual ports")
        for port_config in cfg.ports:
            self.add_virtual_port(port_config)
