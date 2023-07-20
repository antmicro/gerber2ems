"""Module containing Simulation class used for interacting with openEMS"""
import logging
import os
import sys
from typing import Tuple, List

import CSXCAD
import openEMS
import numpy as np

from config import Config, PortConfig
from constants import UNIT, SIMULATION_DIR, GEOMETRY_DIR, BORDER_THICKNESS, PIXEL_SIZE

logger = logging.getLogger(__name__)


class Simulation:
    """Class used for interacting with openEMS"""

    def __init__(self, config: Config) -> None:
        self.config = config

        self.csx = CSXCAD.ContinuousStructure()
        self.fdtd = openEMS.openEMS(NrTS=config.max_steps)
        self.fdtd.SetCSX(self.csx)
        self.mesh = self.csx.GetGrid()
        self.mesh.SetDeltaUnit(UNIT)

        self.ports: List[openEMS.ports.MSLPort] = []

        self.create_materials()

    def create_materials(self) -> None:
        """Creates materials required for simulation"""
        self.material_gerber = self.csx.AddMetal("Gerber")
        self.material_port = self.csx.AddMetal("Port")
        self.material_plane = self.csx.AddMetal("Plane")
        self.material_substrate = self.csx.AddMaterial(
            "Substrate", epsilon=self.config.epsilon
        )

    def add_mesh(self) -> None:
        """Add mesh to simulation"""

        #### X Mesh
        # Min-Max
        x_lines = [
            -self.config.margin_xy,
            self.config.pcb_width + self.config.margin_xy,
        ]
        # PCB
        mesh = self.config.pcb_mesh_xy
        x_lines = np.concatenate(
            (
                x_lines,
                np.arange(0 - mesh / 2, self.config.pcb_width + mesh / 2, step=mesh),
            )
        )
        self.mesh.AddLine("x", x_lines)
        # Margin
        self.mesh.SmoothMeshLines("x", self.config.margin_mesh_xy, ratio=1.2)

        #### Y Mesh
        # Min-Max
        y_lines = [
            -self.config.margin_xy,
            self.config.pcb_height + self.config.margin_xy,
        ]
        # PCB
        mesh = self.config.pcb_mesh_xy
        y_lines = np.concatenate(
            (
                y_lines,
                np.arange(0 - mesh / 2, self.config.pcb_height + mesh / 2, step=mesh),
            )
        )
        self.mesh.AddLine("y", y_lines)
        # Margin
        self.mesh.SmoothMeshLines("x", self.config.margin_mesh_xy, ratio=1.2)

        #### Z Mesh
        # Min-0-Max
        z_lines = [
            -self.config.pcb_thickness - self.config.margin_z,
            0,
            self.config.margin_z,
        ]
        # PCB
        z_lines = np.concatenate(
            (
                z_lines,
                np.arange(-self.config.pcb_thickness, 0, step=self.config.pcb_mesh_z),
            )
        )
        z_lines = np.concatenate((z_lines, [-self.config.pcb_thickness / 2]))
        self.mesh.AddLine("z", z_lines)
        # Margin
        self.mesh.SmoothMeshLines("z", self.config.margin_mesh_z, ratio=1.2)

    def add_contours(self, contours: Tuple[np.ndarray, ...], z_height: float) -> None:
        """Add contours as flat polygons on specified z-height"""
        logger.debug("Adding contours on z=%f", z_height)
        for contour in contours:
            points: List[List[float]] = [[], []]
            for point in contour:
                # Half of the border thickness is subtracted as image is shifted by it
                points[0].append((point[0][0] * PIXEL_SIZE) - BORDER_THICKNESS / 2)
                points[1].append((point[0][1] * PIXEL_SIZE) - BORDER_THICKNESS / 2)

            self.material_gerber.AddPolygon(points, "z", z_height, priority=1)

    def add_port(self, port_config: PortConfig, excite: bool = False):
        """Add microstripline port based on config"""
        logger.debug("Adding port number %d", len(self.ports))
        # TODO: Add handling different layers
        if "y" in port_config.direction:
            start = [port_config.x_pos - port_config.width / 2, port_config.y_pos, 0]
            stop = [
                port_config.x_pos + port_config.width / 2,
                port_config.y_pos + port_config.length,
                -self.config.pcb_thickness,
            ]

        elif "x" in port_config.direction:
            start = [port_config.x_pos, port_config.y_pos - port_config.width / 2, 0]
            stop = [
                port_config.x_pos + port_config.length,
                port_config.y_pos + port_config.width / 2,
                -self.config.pcb_thickness,
            ]

        if "-" in port_config.direction:
            start[0:2], stop[0:2] = stop[0:2], start[0:2]

        port = self.fdtd.AddMSLPort(
            len(self.ports),
            self.material_port,
            start,
            stop,
            "y",
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=1 if excite else 0,
        )
        self.ports.append(port)

    def add_plane(self, z_height):
        """Add metal plane in whole bounding box of the PCB"""
        self.material_plane.AddBox(
            [0, 0, z_height],
            [self.config.pcb_width, self.config.pcb_height, z_height],
            priority=1,
        )

    def add_substrate(self, start, stop):
        """Add substrate in whole bounding box of the PCB"""
        self.material_substrate.AddBox(
            [0, 0, start],
            [self.config.pcb_width, self.config.pcb_height, stop],
            priority=-1,
        )

    def add_dump_box(self):
        """Add electric field dump box in whole bounding box of the PCB at half the thickness"""
        Et = self.csx.AddDump("Et", sub_sampling=[1, 1, 1])
        start = [
            -self.config.margin_xy,
            -self.config.margin_xy,
            -self.config.pcb_thickness / 2,
        ]
        stop = [
            self.config.pcb_width + self.config.margin_xy,
            self.config.pcb_height + self.config.margin_xy,
            -self.config.pcb_thickness / 2,
        ]
        Et.AddBox(start, stop)

    def set_boundary_conditions(self):
        """Add MUR boundary conditions"""
        self.fdtd.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    def set_excitation(self):
        """Sets gauss excitation according to config"""
        self.fdtd.SetGaussExcite(
            (self.config.start_frequency + self.config.stop_frequency) / 2,
            (self.config.stop_frequency - self.config.start_frequency) / 2,
        )

    def run(self):
        """Execute simulation"""
        logger.info("Starting simulation")
        self.fdtd.Run(os.path.join(os.getcwd(), SIMULATION_DIR))

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
