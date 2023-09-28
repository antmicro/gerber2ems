"""Helper functions for generating prerequisites using kmake."""

import subprocess
import logging
import sys
import os

from constants import BASE_DIR

logger = logging.getLogger(__name__)


def generate_prerequisites():
    """Generate all needed prerequisites."""
    check_kmake()
    run_pnp()
    run_stackup()
    run_gerber()


def run_pnp():
    """Run kmake pnp command."""
    log = open(os.path.join(os.getcwd(), BASE_DIR, "kmake.log"), "a+")
    subprocess.Popen(
        "kmake pnp -e -v -o",
        shell=True,
        stdout=log,
        stderr=log,
    ).wait()


def run_stackup():
    """Run kmake stackup-export command."""
    log = open(os.path.join(os.getcwd(), BASE_DIR, "kmake.log"), "a+")
    subprocess.Popen(
        "kmake stackup-export",
        shell=True,
        stdout=log,
        stderr=log,
    ).wait()


def run_gerber():
    """Run kmake gerber command."""
    log = open(os.path.join(os.getcwd(), BASE_DIR, "kmake.log"), "a+")
    subprocess.Popen(
        "kmake gerber -e -x",
        shell=True,
        stdout=log,
        stderr=log,
    ).wait()


def check_kmake():
    """Check whether kmake is installed."""
    try:
        subprocess.Popen(
            "kmake", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).wait()
    except FileNotFoundError:
        logger.error("Kmake is not installed. Exiting...")
        sys.exit(1)
