#!/usr/bin/env python3

"""Main module of EM-Simulator."""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import coloredlogs
import numpy as np

import gerber2ems.importer as importer
from gerber2ems.config import Config, PortType
from gerber2ems.constants import BASE_DIR, DEFAULT_CONFIG_PATH, GEOMETRY_DIR, GEOMETRY_FILE, RESULTS_DIR, SIMULATION_DIR
from gerber2ems.postprocess import Postprocesor
from gerber2ems.preprocess_vertical import preprocess_vertical

logger = logging.getLogger(__name__)
cfg = Config()


def main() -> None:
    """Run the script."""
    args = parse_arguments()

    setup_logging(args)

    cfg_path = Path(args.config or DEFAULT_CONFIG_PATH).absolute()
    Config.load(cfg_path, args)
    if args.update_config:
        cfg.save()
        sys.exit(0)

    if not any([args.geometry, args.simulate, args.postprocess, args.vertical, args.all]):
        logger.info('No steps selected. Exiting. To select steps use "-g", "-s", "-p", "-a" flags')
        sys.exit(0)

    create_dir(BASE_DIR)

    if args.vertical or args.all:
        preprocess_vertical()

    for port in cfg.all_ports:
        port.type = port.type or PortType.MSL

    if args.geometry or args.all:
        logger.info("Creating geometry")
        create_dir(GEOMETRY_DIR, cleanup=True)
        geometry()
    if args.simulate or args.all:
        logger.info("Running simulation")
        create_dir(SIMULATION_DIR, cleanup=True)
        simulate()
    if args.postprocess or args.all:
        logger.info("Postprocessing")
        create_dir(RESULTS_DIR, cleanup=True)
        postprocess()


def geometry() -> None:
    """Create a geometry for the simulation."""
    from gerber2ems.simulation import Simulation

    sim = Simulation()
    importer.import_stackup()
    if any(p.position is None and p.pad is None for p in cfg.ports):
        # skip loading ports if they've been loaded in `--vertical`
        importer.import_port_positions()
    importer.process_gbrs_to_pngs()

    top_layer_name = cfg.get_metals()[0].file
    (width, height) = importer.get_dimensions(top_layer_name + ".png")
    cfg.pcb_height = height
    cfg.pcb_width = width

    sim.create_materials()
    sim.add_gerbers()
    sim.add_grid()
    sim.add_substrates()
    if cfg.arguments.export_field is not None:
        sim.add_dump_boxes()
    sim.set_boundary_conditions(pml=False)
    sim.add_vias()
    sim.add_ports()
    sim.save_geometry()


def simulate() -> None:
    """Run the simulation."""
    from gerber2ems.simulation import Simulation

    for index, port in enumerate(cfg.ports):
        if port.excite:
            sim = Simulation(GEOMETRY_FILE)
            logging.info("Simulating with excitation on port #%i", index)
            sim.set_excitation()
            sim.setup_ports(index)
            sim.run(index)

    if len(sim.ports) == 0:
        importer.import_stackup()
        if any(p.position is None and p.pad is None for p in cfg.ports):
            # skip loading ports if they've been loaded in `--vertical`
            importer.import_port_positions()
        sim.add_ports()

    frequencies = np.linspace(cfg.frequency.start, cfg.frequency.stop, 1001)
    post = Postprocesor(frequencies, len(cfg.ports))

    for index, port in enumerate(cfg.ports):
        if port.excite:
            reflected, incident = sim.get_port_parameters(index, frequencies)
            for i, _ in enumerate(cfg.ports):
                post.add_port_data(i, index, incident[i], reflected[i])

    post.calculate_sparams()
    post.sparam_to_file()


def postprocess() -> None:
    """Postprocess data from the simulation."""
    cfg.arguments.output.mkdir(exist_ok=True, parents=True)
    frequencies = np.linspace(cfg.frequency.start, cfg.frequency.stop, 1001)
    post = Postprocesor(frequencies, len(cfg.ports))
    post.load_sparams()
    post.process_data()
    post.save_to_file()
    post.render_s_params()
    post.render_impedance()
    post.render_smith()
    post.render_diff_pair_s_params()
    post.render_diff_impedance()
    post.render_trace_delays()


def parse_arguments() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        prog="EM-Simulator",
        description="This application allows to perform EM simulations based on standard PCB production files (gerber)",
    )
    parser.add_argument(
        "-c", "--config", metavar="CONFIG_FILE", help="Path to config file [default: `./simulation.json`]"
    )
    parser.add_argument("--update-config", action="store_true", help="Add missing fields to config file")
    parser.add_argument("-g", "--geometry", action="store_true", help="Create geometry")
    parser.add_argument("-s", "--simulate", action="store_true", help="Run simulation")
    parser.add_argument("-p", "--postprocess", action="store_true", help="Postprocess the data")
    parser.add_argument("--vertical", action="store_true", help="Extract vertical ports data from gerber files")
    parser.add_argument(
        "-a", "--all", action="store_true", help="Execute all steps (geometry, simulation, postprocessing)"
    )
    parser.add_argument(
        "--export-field",
        "--ef",
        choices=["outer", "cu-outer", "cu-inner", "substrate"],
        nargs="*",
        default=None,
        help="[s] Export electric field data from the simulation",
    )
    parser.add_argument("--oversampling", type=int, default=4, help="[s] Field dump time-oversampling")
    parser.add_argument(
        "-t", "--transparent", action="store_true", help="[p] Export graphs with transparent background"
    )
    parser.add_argument("--plot-phase", action="store_true", help="[p] Plot phase on S-param graphs")
    parser.add_argument(
        "-b", "--boundaries", action="store_true", help="[p] Export impedance graphs with AOI boundaries"
    )
    parser.add_argument(
        "-i", "--input", default=Path.cwd() / SIMULATION_DIR, type=Path, help="[p] Directory with input S-param files"
    )
    parser.add_argument(
        "-o", "--output", default=Path.cwd() / RESULTS_DIR, type=Path, help="[p] Directory where results will be placed"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--debug", action="store_true", dest="debug")
    group.add_argument("-l", "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level")

    return parser.parse_args()


def setup_logging(args: Any) -> None:
    """Set up logging based on command line arguments."""
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    if args.log_level is not None:
        level = logging.getLevelName(args.log_level)

    if level == logging.DEBUG:
        coloredlogs.install(
            fmt="[%(asctime)s][%(name)s:%(lineno)d][%(levelname).4s] %(message)s",
            datefmt="%H:%M:%S",
            level=level,
            logger=logger,
        )
    else:
        coloredlogs.install(
            fmt="[%(asctime)s][%(levelname).4s] %(message)s",
            datefmt="%H:%M:%S",
            level=level,
            logger=logger,
        )

    to_disable = ["PIL", "matplotlib"]
    for name in to_disable:
        disabled_logger = logging.getLogger(name)
        disabled_logger.setLevel(logging.ERROR)


def create_dir(path: Path, cleanup: bool = False) -> None:
    """Create a directory if doesn't exist."""
    if cleanup and path.exists():
        shutil.rmtree(path)
    if not path.exists():
        path.mkdir(parents=True)


if __name__ == "__main__":
    main()
