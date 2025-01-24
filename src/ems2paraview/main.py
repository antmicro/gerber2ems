#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(prog="gerber2ems-preview")
parser.add_argument("port", nargs="?", help="Select port number")
parser.add_argument("-l", "--list-layers", help="list all simulated layers", action="store_true")


def get_ports() -> list[str]:
    """Get all the simulated ports"""
    path = os.getcwd() + "/ems/simulation/"
    ports = os.listdir(path)
    return ports


def run_paraview(port: str) -> None:
    """Run paraview"""
    path = Path(__file__)
    path = path.parent / "paraview_preview.py"
    subprocess.run(["paraview", "--script", path], env=dict(os.environ, GERBER2EMS_PORT=port))


def main():
    """Run the script"""
    args = parser.parse_args()
    ports = get_ports()
    if args.list_layers:
        print("Simulated ports:")
        for port in ports:
            print(port)
        exit()

    if args.port in ports:
        run_paraview(str(args.port))
    elif args.port is None:
        print("Port not specified")
        exit()
    else:
        print(f"Port {args.port} not found")
        exit()
