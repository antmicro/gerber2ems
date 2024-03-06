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

from gerber2ems.constants import BASE_DIR, SIMULATION_DIR, GEOMETRY_DIR, RESULTS_DIR
from gerber2ems.simulation import Simulation
from gerber2ems.postprocess import Postprocesor
from gerber2ems.config import Config
import gerber2ems.importer as importer

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
            args.all,
        ]
    ):
        logger.info('No steps selected. Exiting. To select steps use "-g", "-s", "-p", "-a" flags')
        sys.exit(0)

    config = open_config(args)
    config = Config(config, args)
    create_dir(BASE_DIR)

    if args.geometry or args.all:
        logger.info("Creating geometry")
        create_dir(GEOMETRY_DIR, cleanup=True)
        sim = Simulation()
        geometry(sim)
    if args.simulate or args.all:
        logger.info("Running simulation")
        create_dir(SIMULATION_DIR, cleanup=True)
        simulate(threads=args.threads)
    if args.postprocess or args.all:
        logger.info("Postprocessing")
        create_dir(RESULTS_DIR, cleanup=True)
        sim = Simulation()
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
    importer.process_gbrs_to_pngs()

    top_layer_name = Config.get().get_metals()[0].file
    (width, height) = importer.get_dimensions(top_layer_name + ".png")
    Config.get().pcb_height = height
    Config.get().pcb_width = width

    sim.create_materials()
    sim.add_gerbers()
    sim.add_mesh()
    sim.add_substrates()
    if Config.get().arguments.export_field:
        sim.add_dump_boxes()
    sim.set_boundary_conditions(pml=False)
    sim.add_vias()
    add_ports(sim)
    sim.save_geometry()


def simulate(threads: None | int = None) -> None:
    """Run the simulation."""
    for index, port in enumerate(Config.get().ports):
        if port.excite:
            sim = Simulation()
            importer.import_stackup()
            sim.create_materials()
            sim.set_excitation()
            logging.info("Simulating with excitation on port #%i", index)
            sim.load_geometry()
            add_ports(sim, index)
            sim.run(index, threads=threads)


def postprocess(sim: Simulation) -> None:
    """Postprocess data from the simulation."""
    if len(sim.ports) == 0:
        add_virtual_ports(sim)

    frequencies = np.linspace(Config.get().start_frequency, Config.get().stop_frequency, 1001)
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
    post.render_diff_pair_s_params()
    post.render_diff_impedance()
    post.render_trace_delays()


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
        help="Create geometry",
    )
    parser.add_argument(
        "-s",
        "--simulate",
        dest="simulate",
        action="store_true",
        help="Run simulation",
    )
    parser.add_argument(
        "-p",
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="Postprocess the data",
    )
    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        action="store_true",
        help="Execute all steps (geometry, simulation, postprocessing)",
    )

    parser.add_argument(
        "--export-field",
        "--ef",
        dest="export_field",
        action="store_true",
        help="Export electric field data from the simulation",
    )

    parser.add_argument("--threads", dest="threads", help="Number of threads to run the simulation on")

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
