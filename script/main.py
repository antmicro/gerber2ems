#!/usr/bin/env python3

"""Main module of EM-Simulator."""
import os
import sys
import json
import argparse
import logging
from typing import Any, Optional
import shutil

import coloredlogs
import numpy as np

from constants import BASE_DIR, SIMULATION_DIR, GEOMETRY_DIR, RESULTS_DIR
from simulation import Simulation
from postprocess import Postprocesor
from config import Config
import importer
import kmake_interface

logger = logging.getLogger(__name__)


def main():
    """Run the script."""
    args = parse_arguments()
    setup_logging(args)

    if not any(
        [
            args.geometry,
            args.simulate,
            args.postprocess,
            args.kmake,
            args.report,
            args.all,
        ]
    ):
        logger.info(
            'No steps selected. Exiting. To select steps use "-k", "-g", "-s", "-p", "-r", "-a" flags'
        )
        sys.exit(0)

    config = open_config(args)
    config = Config(config, args)
    create_dir(BASE_DIR)

    sim = Simulation()

    if args.kmake or args.all:
        logger.info("Creating prerequisites")
        try:
            os.remove(os.path.join(os.getcwd(), BASE_DIR, "kmake.log"))
        except OSError:
            pass
        kmake_interface.generate_prerequisites()
    if args.geometry or args.all:
        logger.info("Creating geometry")
        create_dir(GEOMETRY_DIR, cleanup=True)
        geometry(sim)
    if args.simulate or args.all:
        logger.info("Running simulation")
        create_dir(SIMULATION_DIR, cleanup=True)
        simulate(sim)
    if args.postprocess or args.all:
        logger.info("Postprocessing")
        create_dir(RESULTS_DIR, cleanup=True)
        postprocess(sim)


def add_ports(sim: Simulation, excited_port_number: Optional[int] = None) -> None:
    """Add ports for simulation."""
    logger.info("Adding ports")

    sim.ports = []
    importer.import_port_positions()

    for index, port_config in enumerate(Config.get().ports):
        sim.add_msl_port(port_config, index, index == excited_port_number)


def add_virtual_ports(sim: Simulation) -> None:
    """Add virtual ports needed for data postprocessing due to openEMS api design."""
    logger.info("Adding virtual ports")
    for port_config in Config.get().ports:
        sim.add_virtual_port(port_config)


def geometry(sim: Simulation) -> None:
    """Create a geometry for the simulation."""
    importer.import_stackup()
    importer.process_gbr()
    (width, height) = importer.get_dimensions("F_Cu.png")
    Config.get().pcb_height = height
    Config.get().pcb_width = width
    sim.create_materials()
    sim.add_gerbers()
    sim.add_mesh()
    sim.add_substrates()
    sim.add_dump_boxes()
    sim.set_boundary_conditions(pml=False)
    sim.add_vias()
    add_ports(sim)
    sim.save_geometry()


def simulate(sim: Simulation) -> None:
    """Run the simulation."""
    importer.import_stackup()
    sim.create_materials()
    sim.set_excitation()
    for index, port in enumerate(Config.get().ports):
        if port.excite:
            logging.info("Simulating with excitation on port #%i", index)
            sim.load_geometry()
            add_ports(sim, index)
            sim.run(index)


def postprocess(sim: Simulation) -> None:
    """Postprocess data from the simulation."""
    if len(sim.ports) == 0:
        add_virtual_ports(sim)

    frequencies = np.linspace(
        Config.get().start_frequency, Config.get().stop_frequency, 1001
    )
    post = Postprocesor(frequencies, len(Config.get().ports))
    impedances = np.array([p.impedance for p in Config.get().ports])
    post.add_impedances(impedances)

    for index, port in enumerate(Config.get().ports):
        if port.excite:
            reflected, incident = sim.get_port_parameters(index, frequencies)
            for i, _ in enumerate(Config.get().ports):
                post.add_port_data(i, index, incident[i], reflected[i])

    post.process_data()
    post.save_to_file()
    post.render_s_params()
    post.render_impedance()
    post.render_smith()


def parse_arguments() -> Any:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        prog="EM-Simulator",
        description="This application allows to perform EM simulations for PCB's created with KiCAD",
    )
    parser.add_argument("-c", "--config", dest="config", metavar="CONFIG_FILE")
    parser.add_argument(
        "-g",
        "--geometry",
        dest="geometry",
        action="store_true",
        help="Pass to create geometry",
    )
    parser.add_argument(
        "-s",
        "--simulate",
        dest="simulate",
        action="store_true",
        help="Pass to run simulation",
    )
    parser.add_argument(
        "-p",
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="Pass to postprocess the data",
    )
    parser.add_argument(
        "-r",
        "--report",
        dest="report",
        action="store_true",
        help="Pass to generate report",
    )
    parser.add_argument(
        "-k",
        "--kmake",
        dest="kmake",
        action="store_true",
        help="Pass to generate prerequisites using kmake",
    )
    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        action="store_true",
        help="Pass to execute all steps (geometry, simulation, postprocessing)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--debug", action="store_true", dest="debug")
    group.add_argument(
        "-l", "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level"
    )

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

    # Temporary fix to disable logging from other libraries
    to_disable = ["PIL", "matplotlib"]
    for name in to_disable:
        disabled_logger = logging.getLogger(name)
        disabled_logger.setLevel(logging.ERROR)


def open_config(args: Any) -> None:
    """Try to open and parse config as json."""
    file_name = args.config
    if file_name is None:  # If filename is not supplied fallback to default
        file_name = "./simulation.json"
    file_name = os.path.abspath(file_name)
    if not os.path.isfile(file_name):
        logger.error("Config file doesn't exist: %s", file_name)
        sys.exit(1)

    with open(file_name, "r", encoding="utf-8") as file:
        try:
            config = json.load(file)
        except json.JSONDecodeError as error:
            logger.error(
                "JSON decoding failed at %d:%d: %s",
                error.lineno,
                error.colno,
                error.msg,
            )
            sys.exit(1)

    return config


def create_dir(path: str, cleanup: bool = False) -> None:
    """Create a directory if doesn't exist."""
    directory_path = os.path.join(os.getcwd(), path)
    if cleanup and os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)


if __name__ == "__main__":
    main()
