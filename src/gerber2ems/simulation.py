"""Module containing Simulation class used for interacting with openEMS."""

import logging
import math
import os
import re
import sys
import threading
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from types import TracebackType
from typing import Any, List, Self, Tuple

import CSXCAD
import numpy as np
import openEMS

import gerber2ems.importer as importer
from gerber2ems.common import BBox, Point
from gerber2ems.config import Config, LayerKind, PortConfig, PortType
from gerber2ems.constants import BASE_UNIT, GEOMETRY_FILE, SIMULATION_DIR, UNIT_MULTIPLIER, VIA_POLYGON, ObjectPriority
from gerber2ems.gerber_io import Aperture, ApertureRect, Pad
from gerber2ems.grid_gen import GridGenerator

logger = logging.getLogger(__name__)
cfg = Config()


class LogFilter:
    """Class used for filtering openEMS output."""

    def __init__(self) -> None:
        """Create pipes and backup original system file descriptors."""
        self.pipe_r, self.pipe_w = os.pipe()
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()

        self.old_stdout = os.dup(self.stdout_fd)
        self.old_stderr = os.dup(self.stderr_fd)

    def __enter__(self) -> Self:
        """Force both C++ stdout and stderr to stream into the pipe."""
        os.dup2(self.pipe_w, self.stdout_fd)
        os.dup2(self.pipe_w, self.stderr_fd)
        self.filter_thread = threading.Thread(target=self.filter_gerber_primitive_warnings, daemon=True)
        self.filter_thread.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Clean up open OS pipes and restore the original streams."""
        os.dup2(self.old_stdout, self.stdout_fd)
        os.dup2(self.old_stderr, self.stderr_fd)
        os.close(self.pipe_w)
        if self.filter_thread is not None:
            self.filter_thread.join()
        os.close(self.old_stdout)
        os.close(self.old_stderr)

    def filter_gerber_primitive_warnings(self) -> None:
        """Filter unwanted warnings and write logs back."""
        with os.fdopen(self.pipe_r, "r", errors="ignore") as f:
            for line in f:
                if "Unused primitive" in line and "Gerber" in line:
                    continue
                os.write(self.old_stdout, line.encode("utf-8"))


class Simulation:
    """Class used for interacting with openEMS."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialize simulation object."""
        self.csx = CSXCAD.ContinuousStructure()
        if path:
            self.load_geometry(path)
            self.grid = self.csx.GetGrid()
            sim_steps = (cfg.sim_length * self.calculate_excitation_length()) if cfg.sim_length else cfg.max_steps
        else:
            self.grid = self.csx.GetGrid()
            sim_steps = 1e5
        self.fdtd = openEMS.openEMS(NrTS=sim_steps)
        self.fdtd.SetCSX(self.csx)
        self.grid.SetDeltaUnit(BASE_UNIT / UNIT_MULTIPLIER)

        self.ports: List[openEMS.ports.MSLPort | openEMS.ports.LumpedPort] = []

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
            return [msize / UNIT_MULTIPLIER, mscale]

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
        with Pool(processes=4, initargs=(cfg._config,), initializer=Config.set_config) as p:
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

    def add_contours(self, contours: list[importer.Polygon], z_height: float, layer_index: int) -> None:
        """Add contours as flat polygons on specified z-height."""
        logger.debug("Adding contours on z=%f", z_height)
        for contour in contours:
            points: List[List[float]] = [[], []]
            for point in contour:
                # Half of the border thickness is subtracted as image is shifted by it
                points[0].append((point.y))
                points[1].append(cfg.pcb_height - point.x)

            self.gerber_materials[layer_index].AddPolygon(points, "z", z_height, priority=ObjectPriority.COOPER)

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

    def get_virt_ref_plane_offset(self, port_config: PortConfig) -> float:
        """Get z offset of virtual ref plane linked to port layer."""
        if port_config.layer == 0:
            return port_config.vertical_height()
        return self.get_metal_layer_offset(len(cfg.get_metals()) - 1) - port_config.vertical_height()

    def get_lumped_port_z_span(self, port_config: PortConfig) -> tuple[float, float]:
        """Get Z coordinates that limit port."""
        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = (
            self.get_metal_layer_offset(port_config.plane)
            if port_config.plane
            else self.get_virt_ref_plane_offset(port_config)
        )
        return start_z, stop_z

    def add_port_grid(self) -> None:
        """Add grid around ports for simulation."""
        logger.info("Adding ports grid")

        main_ports = cfg.ports
        for port_config in cfg.all_ports:
            if port_config.type == PortType.VERTICAL:
                _port_start, port_end = self.get_lumped_port_z_span(port_config)
                self.grid_gen.add_z_lines.append(round(port_end))
            for idx, pp in enumerate((port_config.get_pad(), *port_config.get_ref_pads())):
                ap = f"PORT{len(self.grid_gen.add_apertures) + 1}" + f"_{idx}"

                pp.position.move(self.grid_gen.bbox.min)
                self.grid_gen.add_pads.append(Pad(ap, "PORT", pp.position))
                aper_size = pp.size if port_config in main_ports else Point(0, 0)
                self.grid_gen.add_apertures[ap] = Aperture("", ApertureRect(abs(aper_size.x), abs(aper_size.y)))

    def add_msl_port(self, port_config: PortConfig, port_number: int, excite: bool = False) -> None:
        """Add MicroStripLine port based on config."""
        logger.debug(f"Adding port number {port_number}")

        if port_config.direction is None:
            raise AssertionError("PortConfig direction is expected to be loaded before adding ports")

        pp = port_config.get_pad()

        while port_config.direction < 0:
            port_config.direction += 360

        dir_map = {0: "y", 90: "x", 180: "y", 270: "x"}
        if int(port_config.direction) not in dir_map:
            logger.error("Ports rotation is not a multiple of 90 degrees which is not supported, skipping")
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane or port_config.default_plane_idx())

        start = [pp.position.x - pp.size.x / 2, pp.position.y - pp.size.y / 2, round(start_z)]
        stop = [pp.position.x + pp.size.x / 2, pp.position.y + pp.size.y / 2, round(stop_z)]
        logger.debug("Adding port at start: %s end: %s", start, stop)

        port = self.fdtd.AddMSLPort(
            port_number,
            self.csx.AddMetal(f"Port_{port_number}"),
            start,
            stop,
            dir_map[int(port_config.direction)],
            "z",
            Feed_R=port_config.impedance,
            priority=ObjectPriority.PORT,
            excite=1 if excite else 0,
        )
        self.ports.append(port)

    def add_lumped_port(
        self, port_config: PortConfig, port_number: int, start_z: float, stop_z: float, excite: bool = False
    ) -> None:
        """Add lumped port based on config."""
        logger.debug(f"Adding port number {port_number}")

        pp = port_config.get_pad()
        pos = pp.position
        w2 = deepcopy(pp.size)
        w2.scale(0.5)
        start = [pos.x - w2.x, pos.y - w2.y, start_z]
        stop = [pos.x + w2.x, pos.y + w2.y, stop_z]
        logger.debug("Adding lumped port at start: %s end: %s", start, stop)

        port = self.fdtd.AddLumpedPort(
            port_number,
            port_config.impedance,
            start,
            stop,
            "z",
            excite=1 if excite else 0,
            priority=ObjectPriority.PORT,
        )
        self.ports.append(port)

    def add_vertical_port(self, port_config: PortConfig, port_number: int, excite: bool = False) -> None:
        """Add Vertical port (vertical, lumped port to virtual gnd patch)."""
        start_z, stop_z = self.get_lumped_port_z_span(port_config)
        self.add_lumped_port(port_config, port_number, start_z, stop_z, excite)

        pp = port_config.get_pad()
        bbox = BBox()
        bbox.add_pad(pp)

        for ref in port_config.get_ref_pads():
            ref_size = min(ref.size.x, ref.size.y)
            ref.size.x, ref.size.y = ref_size, ref_size
            self.add_cylinder(
                self.via_material,
                ref.position,
                ref_size / 2,
                min(start_z, stop_z),
                abs(start_z - stop_z),
                ObjectPriority.PORT_VERTICAL_REF_STITCH,
            )
            bbox.add_pad(ref)

        self.plane_material.AddBox(
            [bbox.min.x, bbox.min.y, stop_z], [bbox.max.x, bbox.max.y, stop_z], priority=ObjectPriority.COOPER
        )

    def add_plane(self, z_height: float) -> None:
        """Add metal plane in whole bounding box of the PCB."""
        self.plane_material.AddBox(
            [0, 0, z_height],
            [cfg.pcb_width, cfg.pcb_height, z_height],
            priority=ObjectPriority.COOPER,
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
                priority=ObjectPriority.substrate(i),
            )
            logger.debug("Added substrate from %f to %f", offset, offset - layer.thickness)
            offset -= layer.thickness

    def add_vias(self) -> None:
        """Add all vias from excellon file."""
        logger.info("Adding vias from excellon file")
        vias = importer.get_vias()
        for via in vias:
            self.add_via(via[0], via[1], via[2])

    def add_cylinder(
        self,
        material: CSXCAD.CSProperties.CSProperties,
        pos: Point,
        radius: float,
        z_start: float,
        height: float,
        prio: int,
        aprox_vert: int = VIA_POLYGON,
    ) -> None:
        """Add vertical cylinder at specified position with specified diameter."""
        x_coords = []
        y_coords = []
        for i in range(aprox_vert):
            x_coords.append(pos.x + np.sin(i / aprox_vert * 2 * np.pi) * radius)
            y_coords.append(pos.y + np.cos(i / aprox_vert * 2 * np.pi) * radius)
        material.AddLinPoly([x_coords, y_coords], "z", z_start, height, priority=prio)

    def add_via(self, x_pos: float, y_pos: float, diameter: float) -> None:
        """Add via at specified position with specified diameter."""
        h = sum(layer.thickness for layer in cfg.get_substrates())

        pos = Point(x_pos, y_pos)
        self.add_cylinder(self.via_filling_material, pos, diameter / 2, -h, h, ObjectPriority.VIA_FILLING)
        self.add_cylinder(self.via_material, pos, diameter / 2 + cfg.via.plating_thickness, -h, h, ObjectPriority.VIA)

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
        cwd = Path.cwd()
        self.fdtd.SetOverSampling(cfg.arguments.oversampling)
        with LogFilter():
            self.fdtd.Run(str(SIMULATION_DIR / str(excited_port_number)))
        os.chdir(cwd)  # OpenEMS changes cwd, restore original

    def save_geometry(self) -> None:
        """Save geometry to file."""
        filename = GEOMETRY_FILE
        logger.info("Saving geometry to %s", filename)
        self.csx.Write2XML(str(filename))

        # Replacing , with . for numerals in the file
        # (openEMS bug mitigation for locale that uses , as decimal separator)
        with open(filename, "r") as f:
            content = f.read()
        new_content = re.sub(r"([0-9]+),([0-9]+e)", r"\g<1>.\g<2>", content)
        with open(filename, "w") as f:
            f.write(new_content)

    def load_geometry(self, path: Path = GEOMETRY_FILE) -> None:
        """Load geometry from file."""
        logger.info("Loading geometry from %s", path)
        if not path.exists():
            logger.error("Geometry file does not exist. Did you run geometry step?")
            sys.exit(1)
        self.csx.ReadFromXML(str(path))
        self.grid = self.csx.GetGrid()

    def get_port_parameters(self, exindex: int, frequencies: np.ndarray) -> Tuple[List, List]:
        """Return reflected and incident power vs frequency for each port."""
        result_path = SIMULATION_DIR / str(exindex)

        incident: List[np.ndarray] = []
        reflected: List[np.ndarray] = []
        for idx, port in enumerate(self.ports):
            cfg_port = cfg.all_ports[idx]
            try:
                port.CalcPort(str(result_path), frequencies, ref_impedance=cfg_port.impedance)
                logger.debug("Found data for port %d", idx)
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

    def clear_ports(self) -> None:
        """Remove all ports & port leftovers from geometry.xml."""
        self.ports = []

        # delete everything port x created: probes port_ut_xA..C, port_it_xA..B, port_excite_x
        pat = re.compile(r"^port_[a-z]+_[0-9a-zA-Z]")
        for prop in self.csx.GetAllProperties():
            if pat.match(prop.GetName()):
                self.csx.DeleteProperty(prop)

    def add_ports(self) -> None:
        """Add ports for simulation."""
        logger.info("Adding ports")

        self.clear_ports()

        for index, port_config in enumerate(cfg.all_ports):
            match port_config.type:
                case PortType.VERTICAL:
                    self.add_vertical_port(port_config, index, True)
                case PortType.MSL:
                    self.add_msl_port(port_config, index, True)
                case PortType.LUMPED:
                    start_z, stop_z = self.get_lumped_port_z_span(port_config)
                    self.add_lumped_port(port_config, index, start_z, stop_z, True)

    def calculate_excitation_length(self) -> int:
        """Calculate how many steps excitation signal will have."""
        cfl = 0.9  # Courant–Friedrichs–Lewy convergence condition
        c = 299792458  # speed of light
        unit = self.grid.GetDeltaUnit()
        delta_coeff = 0
        for axis in range(3):
            lines = self.grid.GetLines(axis, do_sort=True)
            min_delta = min(b - a for a, b in zip(lines, lines[1:], strict=False)) * unit
            delta_coeff += 1 / (min_delta**2)
        step_time = cfl / (c * math.sqrt(delta_coeff))
        center_freq = (cfg.frequency.start + cfg.frequency.stop) * 0.5
        effective_pulse_len = math.sqrt(math.log(10)) * 6 / math.pi / center_freq
        return effective_pulse_len / step_time
