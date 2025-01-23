#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(prog="gerber2ems-preview")
parser.add_argument("port", nargs="?", help="Select port number")
parser.add_argument("-l", "--list-layers", help="list all simulated layers", action="store_true")


def get_ports() -> list[str]:
    path = os.getcwd() + "/ems/simulation/"
    layers = os.listdir(path)
    return layers


def run(layer: str) -> None:
    path = Path(__file__)
    path = path.parent / "paraview_preview.py"
    os.system(f'GERBER2EMS_PORT="{layer}" paraview --script={path}')


if __name__ == "__main__":
    args = parser.parse_args()
    ports = get_ports()
    if args.list_layers:
        print("Simulated ports:")
        for port in ports:
            print(port)
        exit()

    if args.port in ports:
        run(str(args.port))
    elif args.port is None:
        print("Port not specified")
        exit()
    else:
        print(f"Port {args.port} not found")
        exit()
