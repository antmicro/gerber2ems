#!/usr/bin/env python3

"""Main module of EM-Simulator"""
import os
import sys
import json
import argparse
import logging
from typing import Any
import coloredlogs

from constants import SIM_DIR, TMP_DIR
import process_gbr
from simulation import Simulation
from config import Config

logger = logging.getLogger(__name__)


def main():
    """Main function of the program"""
    args = parse_arguments()
    setup_logging(args)

    config = open_config(args)
    config = Config(config)
    create_dirs()
    process_gbr.process()

    sim = Simulation(config)
    sim.add_mesh()
    contours = process_gbr.get_contours("F_Cu.png")

    logger.info("Adding contours")
    sim.add_contours(contours, 0)

    logger.info("Adding ports")
    excited = True
    for port_config in config.ports:
        sim.add_port(port_config, excited)
        excited = False

    logger.info("Adding planes and substrates")
    sim.add_plane(-config.pcb_thickness)
    sim.add_substrate(0, -config.pcb_thickness)

    sim.add_dump_box()
    sim.set_boundary_conditions()

    sim.set_excitation()
    sim.run()

    sim.save_geometry()


def parse_arguments() -> Any:
    """Parses commandline arguments"""
    parser = argparse.ArgumentParser(
        prog="EM-Simulator",
        description="This application allows to perform EM simulations for PCB's created with KiCAD",
    )
    parser.add_argument("-c", "--config", dest="config", metavar="CONFIG_FILE")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--debug", action="store_true", dest="debug")
    group.add_argument(
        "-l", "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], dest="log_level"
    )

    args = parser.parse_args()
    return args


def setup_logging(args: Any) -> None:
    """Set up logging based on command line arguments"""
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
        )
    else:
        coloredlogs.install(
            fmt="[%(asctime)s][%(levelname).4s] %(message)s",
            datefmt="%H:%M:%S",
            level=level,
        )


def open_config(args: Any) -> None:
    """Try to open and parse config as json"""
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


def create_dirs() -> None:
    """Creates directories for output and temporary files"""
    sim_dir = os.path.join(os.getcwd(), SIM_DIR)
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)

    tmp_dir = os.path.join(os.getcwd(), TMP_DIR)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)


if __name__ == "__main__":
    main()
