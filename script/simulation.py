"""Module containing Simulation class used for interacting with openEMS."""
import logging
import os
import sys
import math
from typing import Tuple, List, Any

import CSXCAD
import openEMS
import numpy as np

from config import Config, PortConfig, LayerKind
from constants import (
    UNIT,
    SIMULATION_DIR,
    GEOMETRY_DIR,
    VIA_POLYGON,
)
import importer

logger = logging.getLogger(__name__)


class Simulation:
    """Class used for interacting with openEMS."""

    def __init__(self) -> None:
        """Initialize simulation object."""
        self.csx = CSXCAD.ContinuousStructure()
        self.fdtd = openEMS.openEMS(NrTS=Config.get().max_steps)
        self.fdtd.SetCSX(self.csx)
        self.mesh = self.csx.GetGrid()
        self.mesh.SetDeltaUnit(UNIT)

        self.ports: List[openEMS.ports.MSLPort] = []

        # Separate metal materials for easier switching of layers
        self.gerber_materials: List[Any] = []
        self.substrate_materials: List[Any] = []
        self.plane_material = self.csx.AddMetal("Plane")
        self.port_material = self.csx.AddMetal("Port")
        self.via_material = self.csx.AddMetal("Via")
        self.via_filling_material = self.csx.AddMaterial(
            "ViaFilling", epsilon=Config.get().via_filling_epsilon
        )

    def create_materials(self) -> None:
        """Create materials required for simulation."""
        for i, _ in enumerate(Config.get().get_metals()):
            self.gerber_materials.append(self.csx.AddMetal(f"Gerber_{i}"))
        for i, layer in enumerate(Config.get().get_substrates()):
            self.substrate_materials.append(
                self.csx.AddMaterial(f"Substrate_{i}", epsilon=layer.epsilon)
            )

    def add_mesh(self) -> None:
        """Add mesh to simulation."""
        #### X Mesh
        # Min-Max
        x_lines = [
            -Config.get().margin_xy,
            Config.get().pcb_width + Config.get().margin_xy,
        ]
        # PCB
        mesh = Config.get().pcb_mesh_xy
        x_lines = np.concatenate(
            (
                x_lines,
                np.arange(0 - mesh / 2, Config.get().pcb_width + mesh / 2, step=mesh),
            )
        )
        self.mesh.AddLine("x", x_lines)
        # Margin
        self.mesh.SmoothMeshLines("x", Config.get().margin_mesh_xy, ratio=1.2)

        #### Y Mesh
        # Min-Max
        y_lines = [
            -Config.get().margin_xy,
            Config.get().pcb_height + Config.get().margin_xy,
        ]
        # PCB
        mesh = Config.get().pcb_mesh_xy
        y_lines = np.concatenate(
            (
                y_lines,
                np.arange(0 - mesh / 2, Config.get().pcb_height + mesh / 2, step=mesh),
            )
        )
        self.mesh.AddLine("y", y_lines)
        # Margin
        self.mesh.SmoothMeshLines("y", Config.get().margin_mesh_xy, ratio=1.2)

        #### Z Mesh
        # Min-0-Max

        z_lines = np.array([0])
        offset = 0
        z_count = Config.get().inter_copper_layers
        if z_count % 2 == 1:  # Increasing by one to always have z_line at dumpbox
            z_count += 1
        for layer in Config.get().get_substrates():
            z_lines = np.concatenate(
                (
                    z_lines,
                    np.linspace(
                        offset - layer.thickness, offset, z_count, endpoint=False
                    ),
                )
            )
            offset -= layer.thickness
        z_lines = np.concatenate(
            (z_lines, [Config.get().margin_z, offset - Config.get().margin_z])
        )
        z_lines = np.round(z_lines)

        self.mesh.AddLine("z", z_lines)
        # Margin
        self.mesh.SmoothMeshLines("z", Config.get().margin_mesh_z, ratio=1.2)

        logger.debug("Mesh x lines: %s", x_lines)
        logger.debug("Mesh y lines: %s", y_lines)
        logger.debug("Mesh z lines: %s", z_lines)

        xyz = [
            self.mesh.GetQtyLines("x"),
            self.mesh.GetQtyLines("y"),
            self.mesh.GetQtyLines("z"),
        ]
        logger.info(
            "Mesh line count, x: %d, y: %d z: %d. Total number of cells: ~%.2fM",
            xyz[0],
            xyz[1],
            xyz[2],
            xyz[0] * xyz[1] * xyz[2] / 1.0e6,
        )

    def add_gerbers(self) -> None:
        """Add metal from all gerber files."""
        logger.info("Adding copper from gerber files")

        importer.process_gbr()

        offset = 0
        index = 0
        for layer in Config.get().layers:
            if layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
            elif layer.kind == LayerKind.METAL:
                logger.info("Adding contours for %s", layer.file)
                contours = importer.get_triangles(layer.file + ".png")
                self.add_contours(contours, offset, index)
                index += 1

    def add_contours(
        self, contours: np.ndarray, z_height: float, layer_index: int
    ) -> None:
        """Add contours as flat polygons on specified z-height."""
        logger.debug("Adding contours on z=%f", z_height)
        for contour in contours:
            points: List[List[float]] = [[], []]
            for point in contour:
                # Half of the border thickness is subtracted as image is shifted by it
                points[0].append((point[1]))
                points[1].append(Config.get().pcb_height - point[0])

            self.gerber_materials[layer_index].AddPolygon(
                points, "z", z_height, priority=1
            )

    def get_metal_layer_offset(self, index: int) -> float:
        """Get z offset of nth metal layer."""
        current_metal_index = -1
        offset = 0
        for layer in Config.get().layers:
            if layer.kind == LayerKind.METAL:
                current_metal_index += 1
                if current_metal_index == index:
                    return offset
            elif layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
        logger.error("Hadn't found %dth metal layer", index)
        sys.exit(1)

    def add_msl_port(
        self, port_config: PortConfig, port_number: int, excite: bool = False
    ):
        """Add microstripline port based on config."""
        logger.debug("Adding port number %d", len(self.ports))

        if port_config.position is None or port_config.direction is None:
            logger.error("Port has no defined position or rotation, skipping")
            return

        while port_config.direction < 0:
            port_config.direction += 360

        dir_map = {0: "y", 90: "x", 180: "y", 270: "x"}
        if int(port_config.direction) not in dir_map:
            logger.error(
                "Ports rotation is not a multiple of 90 degrees which is not supported, skipping"
            )
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)

        angle = port_config.direction / 360 * 2 * math.pi

        start = [
            round(
                port_config.position[0]
                - (port_config.width / 2) * round(math.cos(angle))
            ),
            round(
                port_config.position[1]
                - (port_config.width / 2) * round(math.sin(angle))
            ),
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
            self.port_material,
            start,
            stop,
            dir_map[int(port_config.direction)],
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=1 if excite else 0,
        )
        self.ports.append(port)

        self.mesh.AddLine("x", start[0])
        self.mesh.AddLine("x", stop[0])
        self.mesh.AddLine("y", start[1])
        self.mesh.AddLine("y", stop[1])

    def add_resistive_port(self, port_config: PortConfig, excite: bool = False):
        """Add resistive port based on config."""
        logger.debug("Adding port number %d", len(self.ports))

        if port_config.position is None or port_config.direction is None:
            logger.error("Port has no defined position or rotation, skipping")
            return

        dir_map = {0: "y", 90: "x", 180: "y", 270: "x"}
        if int(port_config.direction) not in dir_map:
            logger.error(
                "Ports rotation is not a multiple of 90 degrees which is not supported, skipping"
            )
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)

        angle = port_config.direction / 360 * 2 * math.pi

        start = [
            round(
                port_config.position[0]
                - (port_config.width / 2) * round(math.cos(angle))
            ),
            round(
                port_config.position[1]
                - (port_config.width / 2) * round(math.sin(angle))
            ),
            round(start_z),
        ]
        stop = [
            round(
                port_config.position[0]
                + (port_config.width / 2) * round(math.cos(angle))
            ),
            round(
                port_config.position[1]
                - (port_config.width / 2) * round(math.sin(angle))
            ),
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

        logger.debug("Port direction: %s", dir_map[int(port_config.direction)])
        if dir_map[int(port_config.direction)] == "y":
            self.mesh.AddLine("x", start[0])
            self.mesh.AddLine("x", stop[0])
            self.mesh.AddLine("y", start[1])
        else:
            self.mesh.AddLine("x", start[0])
            self.mesh.AddLine("y", start[1])
            self.mesh.AddLine("y", stop[1])

    def add_virtual_port(self, port_config: PortConfig) -> None:
        """Add virtual port for extracting sim data from files. Needed due to OpenEMS api desing."""
        logger.debug("Adding virtual port number %d", len(self.ports))
        for i in range(11):
            self.mesh.AddLine("x", i)
            self.mesh.AddLine("y", i)
        self.mesh.AddLine("z", 0)
        self.mesh.AddLine("z", 10)
        port = self.fdtd.AddMSLPort(
            len(self.ports),
            self.port_material,
            [0, 0, 0],
            [10, 10, 10],
            "x",
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=0,
        )
        self.ports.append(port)

    def add_plane(self, z_height):
        """Add metal plane in whole bounding box of the PCB."""
        self.plane_material.AddBox(
            [0, 0, z_height],
            [Config.get().pcb_width, Config.get().pcb_height, z_height],
            priority=1,
        )

    def add_substrates(self):
        """Add substrate in whole bounding box of the PCB."""
        logger.info("Adding substrates")

        offset = 0
        for i, layer in enumerate(Config.get().get_substrates()):
            self.substrate_materials[i].AddBox(
                [0, 0, offset],
                [
                    Config.get().pcb_width,
                    Config.get().pcb_height,
                    offset - layer.thickness,
                ],
                priority=-i,
            )
            logger.debug(
                "Added substrate from %f to %f", offset, offset - layer.thickness
            )
            offset -= layer.thickness

    def add_vias(self):
        """Add all vias from excellon file."""
        logger.info("Adding vias from excellon file")
        vias = importer.get_vias()
        for via in vias:
            self.add_via(via[0], via[1], via[2])

    def add_via(self, x_pos, y_pos, diameter):
        """Add via at specified position with specified diameter."""
        thickness = sum(layer.thickness for layer in Config.get().get_substrates())

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON):
            x_coords.append(x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
            y_coords.append(y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
        self.via_filling_material.AddLinPoly(
            [x_coords, y_coords], "z", -thickness, thickness, priority=51
        )

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON)[::-1]:
            x_coords.append(
                x_pos
                + np.sin(i / VIA_POLYGON * 2 * np.pi)
                * (diameter / 2 + Config.get().via_plating)
            )
            y_coords.append(
                y_pos
                + np.cos(i / VIA_POLYGON * 2 * np.pi)
                * (diameter / 2 + Config.get().via_plating)
            )
        self.via_material.AddLinPoly(
            [x_coords, y_coords], "z", -thickness, thickness, priority=50
        )

    def add_dump_boxes(self):
        """Add electric field dump box in whole bounding box of the PCB at half the thickness of each substrate."""
        logger.info("Adding dump box for each dielectic")
        offset = 0
        for i, layer in enumerate(Config.get().get_substrates()):
            height = offset - layer.thickness / 2
            logger.debug("Adding dump box at %f", height)
            dump = self.csx.AddDump(f"e_field_{i}", sub_sampling=[1, 1, 1])
            start = [
                -Config.get().margin_xy,
                -Config.get().margin_xy,
                height,
            ]
            stop = [
                Config.get().pcb_width + Config.get().margin_xy,
                Config.get().pcb_height + Config.get().margin_xy,
                height,
            ]
            dump.AddBox(start, stop)
            offset -= layer.thickness

    def set_boundary_conditions(self, pml=False):
        """Add boundary conditions. MUR for fast simulation, PML for more accurate."""
        if pml:
            logger.info("Adding perfectly matched layer boundary condition")
            self.fdtd.SetBoundaryCond(
                ["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"]
            )
        else:
            logger.info("Adding MUR boundary condition")
            self.fdtd.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    def set_excitation(self):
        """Set gauss excitation according to config."""
        logger.debug(
            "Setting excitation to gaussian pulse from %f to %f",
            Config.get().start_frequency,
            Config.get().stop_frequency,
        )
        self.fdtd.SetGaussExcite(
            (Config.get().start_frequency + Config.get().stop_frequency) / 2,
            (Config.get().stop_frequency - Config.get().start_frequency) / 2,
        )

    def set_sinus_excitation(self, freq):
        logger.debug("Setting excitation to sine at %f", freq)
        self.fdtd.SetSinusExcite(freq)

    def run(self, excited_port_number):
        """Execute simulation."""
        logger.info("Starting simulation")
        cwd = os.getcwd()
        self.fdtd.Run(
            os.path.join(os.getcwd(), SIMULATION_DIR, str(excited_port_number))
        )
        os.chdir(cwd)

    def save_geometry(self) -> None:
        """Save geometry to file."""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Saving geometry to %s", filename)
        self.csx.Write2XML(filename)

    def load_geometry(self) -> None:
        """Load geometry from file."""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Loading geometry from %s", filename)
        if not os.path.exists(filename):
            logger.error("Geometry file does not exist. Did you run geometry step?")
            sys.exit(1)
        self.csx.ReadFromXML(filename)

    def get_port_parameters(self, frequencies) -> Tuple[List, List]:
        """Return reflected and incident power vs frequency for each port."""
        result_path = os.path.join(os.getcwd(), SIMULATION_DIR)

        incident: List[np.ndarray] = []
        reflected: List[np.ndarray] = []
        for index, port in enumerate(self.ports):
            try:
                port.CalcPort(result_path, frequencies)
                logger.debug("Found data for port %d", index)
            except IOError:
                logger.error(
                    "Port data files do not exist. Did you run simulation step?"
                )
                sys.exit(1)
            incident.append(port.uf_inc)
            reflected.append(port.uf_ref)

        return (reflected, incident)
