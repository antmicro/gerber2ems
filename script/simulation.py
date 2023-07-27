"""Module containing Simulation class used for interacting with openEMS"""
import logging
import os
import sys
from typing import Tuple, List

import CSXCAD
import openEMS
import numpy as np

from config import Config, PortConfig, LayerKind
from constants import (
    UNIT,
    SIMULATION_DIR,
    GEOMETRY_DIR,
    BORDER_THICKNESS,
    PIXEL_SIZE,
    VIA_POLYGON,
)
import process_gbr

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
        self.materials_substrate = []
        for i, layer in enumerate(self.config.get_substrates()):
            self.materials_substrate.append(
                self.csx.AddMaterial(f"Substrate_{i}", epsilon=layer.epsilon)
            )
        self.material_via = self.csx.AddMetal("Via")
        self.material_filling = self.csx.AddMaterial("ViaFilling", epsilon=1)

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

        thickness = sum(l.thickness for l in self.config.get_substrates())
        z_lines = [
            -thickness - self.config.margin_z,
            0,
            self.config.margin_z,
        ]
        # PCB
        z_lines = np.concatenate(
            (
                z_lines,
                np.arange(-thickness, 0, step=self.config.pcb_mesh_z),
            )
        )
        offset = 0
        for layer in self.config.get_substrates():
            np.append(z_lines, offset - (layer.thickness / 2))
            offset -= layer.thickness

        self.mesh.AddLine("z", z_lines)
        # Margin
        self.mesh.SmoothMeshLines("z", self.config.margin_mesh_z, ratio=1.2)

    def add_gerbers(self) -> None:
        """Add metal from all gerber files"""
        process_gbr.process()

        offset = 0
        for layer in self.config.layers:
            if layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
            elif layer.kind == LayerKind.METAL:
                logger.info("Adding contours for %s", layer.file)
                contours = process_gbr.get_triangles(layer.file + ".png")
                self.add_contours(contours, offset)

    def add_contours(self, contours: np.ndarray, z_height: float) -> None:
        """Add contours as flat polygons on specified z-height"""
        logger.debug("Adding contours on z=%f", z_height)
        for contour in contours:
            points: List[List[float]] = [[], []]
            for point in contour:
                # Half of the border thickness is subtracted as image is shifted by it
                points[0].append((point[1] * PIXEL_SIZE) - BORDER_THICKNESS / 2)
                points[1].append(
                    self.config.pcb_height
                    - (point[0] * PIXEL_SIZE)
                    + BORDER_THICKNESS / 2
                )

            self.material_gerber.AddPolygon(points, "z", z_height, priority=1)

    def get_metal_layer_offset(self, index: int) -> float:
        """Get z offset of nth metal layer"""
        current_metal_index = -1
        offset = 0
        for layer in self.config.layers:
            if layer.kind == LayerKind.METAL:
                current_metal_index += 1
                if current_metal_index == index:
                    return offset
            elif layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
        logger.error("Hadn't found %dth metal layer", index)
        sys.exit(1)

    def add_port(self, port_config: PortConfig, excite: bool = False):
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
            self.material_port,
            start,
            stop,
            port_config.direction.replace("-", ""),
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

    def add_substrates(self):
        """Add substrate in whole bounding box of the PCB"""
        offset = 0
        for i, layer in enumerate(self.config.get_substrates()):
            self.materials_substrate[i].AddBox(
                [0, 0, offset],
                [
                    self.config.pcb_width,
                    self.config.pcb_height,
                    offset - layer.thickness,
                ],
                priority=-i,
            )
            offset -= layer.thickness

    def add_vias(self):
        """Add all vias from excellon file"""
        vias = process_gbr.get_vias()
        for via in vias:
            self.add_via(via[0], via[1], via[2])

    def add_via(self, x_pos, y_pos, diameter):
        """Adds via at specified position with specified diameter"""
        thickness = sum(l.thickness for l in self.config.get_substrates())

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON):
            x_coords.append(x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
            y_coords.append(y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
        self.material_filling.AddLinPoly(
            [x_coords, y_coords], "z", -thickness, thickness, priority=51
        )

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON)[::-1]:
            x_coords.append(
                x_pos
                + np.sin(i / VIA_POLYGON * 2 * np.pi)
                * (diameter / 2 + self.config.via_plating)
            )
            y_coords.append(
                y_pos
                + np.cos(i / VIA_POLYGON * 2 * np.pi)
                * (diameter / 2 + self.config.via_plating)
            )
        self.material_via.AddLinPoly(
            [x_coords, y_coords], "z", -thickness, thickness, priority=50
        )

    def add_dump_boxes(self):
        """Add electric field dump box in whole bounding box of the PCB at half the thickness of each substrate"""
        offset = 0
        for i, layer in enumerate(self.config.get_substrates()):
            dump = self.csx.AddDump(f"e_field_{i}", sub_sampling=[1, 1, 1])
            start = [
                -self.config.margin_xy,
                -self.config.margin_xy,
                offset - layer.thickness / 2,
            ]
            stop = [
                self.config.pcb_width + self.config.margin_xy,
                self.config.pcb_height + self.config.margin_xy,
                offset - layer.thickness / 2,
            ]
            dump.AddBox(start, stop)
            offset -= layer.thickness

    def set_boundary_conditions(self):
        """Add MUR boundary conditions"""
        self.fdtd.SetBoundaryCond(
            ["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"]
        )

    def set_excitation(self):
        """Sets gauss excitation according to config"""
        self.fdtd.SetGaussExcite(
            (self.config.start_frequency + self.config.stop_frequency) / 2,
            (self.config.stop_frequency - self.config.start_frequency) / 2,
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
