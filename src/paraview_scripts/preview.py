#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(prog="gerber2ems-preview")
parser.add_argument("-l", "--layer", help="select PCB layer")
parser.add_argument("-L", "--list-layers", help="list all simulated layers", action="store_true")


def get_layers() -> list[str]:
    path = os.getcwd() + "/ems/simulation/"
    layers = os.listdir(path)
    return layers


def run(layer: str) -> None:
    path = Path(__file__)
    path = path.parent / "paraview_preview.py"
    os.system(f'GERBER2EMS_PREVIEW_LAYER="{layer}" paraview --script={path}')


if __name__ == "__main__":
    args = parser.parse_args()
    layers = get_layers()
    if args.list_layers:
        print("Simulated layers:")
        for layer in layers:
            print(layer)
        exit()
    if args.layer in layers:
        run(str(args.layer))
    elif args.layer is None:
        print("Layer is not specified")
        exit()
    else:
        print(f"Layer {args.layer} not found")
        exit()
