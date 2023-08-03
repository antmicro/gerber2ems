"""Module containing Simulation class used for interacting with openEMS"""
import logging
import os
import sys
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
    """Class used for interacting with openEMS"""

    def __init__(self) -> None:
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
        """Creates materials required for simulation"""
        for i, layer in enumerate(Config.get().get_metals()):
            self.gerber_materials.append(self.csx.AddMetal(f"Gerber_{i}"))
        for i, layer in enumerate(Config.get().get_substrates()):
            self.substrate_materials.append(
                self.csx.AddMaterial(f"Substrate_{i}", epsilon=layer.epsilon)
            )

    def add_mesh(self) -> None:
        """Add mesh to simulation"""

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
        self.mesh.SmoothMeshLines("x", Config.get().margin_mesh_xy, ratio=1.2)

        #### Z Mesh
        # Min-0-Max

        thickness = sum(l.thickness for l in Config.get().get_substrates())
        z_lines = [
            -thickness - Config.get().margin_z,
            0,
            Config.get().margin_z,
        ]
        # PCB
        z_lines = np.concatenate(
            (
                z_lines,
                np.arange(-thickness, 0, step=Config.get().pcb_mesh_z),
            )
        )
        offset = 0
        for layer in Config.get().get_substrates():
            np.append(z_lines, offset - (layer.thickness / 2))
            offset -= layer.thickness

        self.mesh.AddLine("z", z_lines)
        # Margin
        self.mesh.SmoothMeshLines("z", Config.get().margin_mesh_z, ratio=1.2)

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
        """Add metal from all gerber files"""
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
        """Add contours as flat polygons on specified z-height"""
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
        """Get z offset of nth metal layer"""
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

    def add_msl_port(self, port_config: PortConfig, excite: bool = False):
        """Add microstripline port based on config"""
        logger.debug("Adding port number %d", len(self.ports))

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)

        if "y" in port_config.direction:
            start = [
                port_config.x_pos - port_config.width / 2,
                port_config.y_pos,
                start_z,
            ]
            stop = [
                port_config.x_pos + port_config.width / 2,
                port_config.y_pos + port_config.length,
                stop_z,
            ]

        elif "x" in port_config.direction:
            start = [
                port_config.x_pos,
                port_config.y_pos - port_config.width / 2,
                start_z,
            ]
            stop = [
                port_config.x_pos + port_config.length,
                port_config.y_pos + port_config.width / 2,
                stop_z,
            ]

        if "-" in port_config.direction:
            start[0:2], stop[0:2] = stop[0:2], start[0:2]

        port = self.fdtd.AddMSLPort(
            len(self.ports),
            self.port_material,
            start,
            stop,
            port_config.direction.replace("-", ""),
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=1 if excite else 0,
        )
        self.ports.append(port)

    def add_virtual_port(self, port_config: PortConfig) -> None:
        """Add virtual port for extracting sim data from files. Needed due to OpenEMS api desing"""
        logger.debug("Adding virtual port number %d", len(self.ports))
        port = self.fdtd.AddLumpedPort(
            len(self.ports), port_config.impedance, [0, 0, 0], [1, 0, 1], "z"
        )
        self.ports.append(port)

    def add_plane(self, z_height):
        """Add metal plane in whole bounding box of the PCB"""
        self.plane_material.AddBox(
            [0, 0, z_height],
            [Config.get().pcb_width, Config.get().pcb_height, z_height],
            priority=1,
        )

    def add_substrates(self):
        """Add substrate in whole bounding box of the PCB"""
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
        """Add all vias from excellon file"""
        logger.info("Adding vias from excellon file")
        vias = importer.get_vias()
        for via in vias:
            self.add_via(via[0], via[1], via[2])

    def add_via(self, x_pos, y_pos, diameter):
        """Adds via at specified position with specified diameter"""
        thickness = sum(l.thickness for l in Config.get().get_substrates())

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
        """Add electric field dump box in whole bounding box of the PCB at half the thickness of each substrate"""
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
        """Add boundary conditions. MUR for fast simulation, PML for more accurate"""
        if pml:
            logger.info("Adding perfectly matched layer boundary condition")
            self.fdtd.SetBoundaryCond(
                ["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"]
            )
        else:
            logger.info("Adding MUR boundary condition")
            self.fdtd.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    def set_excitation(self):
        """Sets gauss excitation according to config"""
        self.fdtd.SetGaussExcite(
            (Config.get().start_frequency + Config.get().stop_frequency) / 2,
            (Config.get().stop_frequency - Config.get().start_frequency) / 2,
        )

    def run(self):
        """Execute simulation"""
        logger.info("Starting simulation")
        cwd = os.getcwd()
        self.fdtd.Run(os.path.join(os.getcwd(), SIMULATION_DIR))
        os.chdir(cwd)

    def save_geometry(self) -> None:
        """Save geometry to file"""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Saving geometry to %s", filename)
        self.csx.Write2XML(filename)

    def load_geometry(self) -> None:
        """Loads geometry from file"""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Loading geometry from %s", filename)
        if not os.path.exists(filename):
            logger.error("Geometry file does not exist. Did you run geometry step?")
            sys.exit(1)
        self.csx.ReadFromXML(filename)

    def get_port_parameters(self, frequencies) -> Tuple[List, List]:
        """Returns reflected and incident power vs frequency for each port"""
        result_path = os.path.join(os.getcwd(), SIMULATION_DIR)

        incident: List[np.ndarray] = []
        reflected: List[np.ndarray] = []
        for port in self.ports:
            try:
                port.CalcPort(result_path, frequencies)
            except IOError:
                logger.error(
                    "Port data files do not exist. Did you run simulation step?"
                )
                sys.exit(1)
            incident.append(port.uf_inc)
            reflected.append(port.uf_ref)

        return (reflected, incident)
