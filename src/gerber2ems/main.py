#!/usr/bin/env python3

"""Main module of EM-Simulator."""
import os
import sys
import argparse
import logging
from typing import Any
import shutil

import coloredlogs
import numpy as np

from gerber2ems.constants import BASE_DIR, SIMULATION_DIR, GEOMETRY_DIR, RESULTS_DIR
from gerber2ems.simulation import Simulation
from gerber2ems.postprocess import Postprocesor
from gerber2ems.config import Config
import gerber2ems.importer as importer

logger = logging.getLogger(__name__)
cfg = Config()


def main() -> None:
    """Run the script."""
    args = parse_arguments()
    Config.load(args)
    setup_logging(args)
    if args.update_config:
        exit(0)

    if not any(
        [
            args.geometry,
            args.simulate,
            args.postprocess,
            args.all,
        ]
    ):
        logger.info('No steps selected. Exiting. To select steps use "-g", "-s", "-p", "-a" flags')
        sys.exit(0)

    create_dir(BASE_DIR)

    if args.geometry or args.all:
        logger.info("Creating geometry")
        create_dir(GEOMETRY_DIR, cleanup=True)
        sim = Simulation()
        geometry(sim)
    if args.simulate or args.all:
        logger.info("Running simulation")
        create_dir(SIMULATION_DIR, cleanup=True)
        simulate()
    if args.postprocess or args.all:
        logger.info("Postprocessing")
        create_dir(RESULTS_DIR, cleanup=True)
        sim = Simulation()
        postprocess(sim)


def geometry(sim: Simulation) -> None:
    """Create a geometry for the simulation."""
    importer.import_stackup()
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
    if cfg.arguments.export_field:
        sim.add_dump_boxes()
    sim.set_boundary_conditions(pml=False)
    sim.add_vias()
    sim.add_ports()
    sim.save_geometry()


def simulate() -> None:
    """Run the simulation."""
    for index, port in enumerate(cfg.ports):
        if port.excite:
            sim = Simulation()
            logging.info("Simulating with excitation on port #%i", index)
            sim.load_geometry()
            sim.set_excitation()
            sim.setup_ports(index)
            sim.run(index)


def postprocess(sim: Simulation) -> None:
    """Postprocess data from the simulation."""
    if len(sim.ports) == 0:
        sim.add_virtual_ports()

    frequencies = np.linspace(cfg.frequency.start, cfg.frequency.stop, 1001)
    post = Postprocesor(frequencies, len(cfg.ports))
    impedances = np.array([p.impedance for p in cfg.ports])
    post.add_impedances(impedances)

    for index, port in enumerate(cfg.ports):
        if port.excite:
            reflected, incident = sim.get_port_parameters(index, frequencies)
            for i, _ in enumerate(cfg.ports):
                post.add_port_data(i, index, incident[i], reflected[i])

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
    parser.add_argument(
        "-a", "--all", action="store_true", help="Execute all steps (geometry, simulation, postprocessing)"
    )
    parser.add_argument(
        "--export-field",
        "--ef",
        choices=["outer", "cu-outer", "cu-inner", "substrate"],
        nargs="*",
        default=None,
        help="Export electric field data from the simulation",
    )
    parser.add_argument("--oversampling", type=int, default=4, help="Field dump time-oversampling")
    parser.add_argument("-t", "--transparent", action="store_true", help="Export graphs with transparent background")
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


def create_dir(path: str, cleanup: bool = False) -> None:
    """Create a directory if doesn't exist."""
    directory_path = os.path.join(os.getcwd(), path)
    if cleanup and os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)


if __name__ == "__main__":
    main()
