#!/usr/bin/env python3
"""Export pngs used for blender visualization purposes."""
import typer
from typing import Annotated
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy import interpolate
from PIL import Image
import re
import logging
import coloredlogs
import sys
from multiprocessing import Pool, Value, Lock
from functools import partial
from math import inf, sqrt
from typing import Tuple, List, Any

app = typer.Typer()
progress_counter = Value("i", 0)
counter_lock = Lock()


def get_min_max(fnm: List[Tuple[int, str, int, Path]]) -> Tuple[float, float]:
    """Get min and max value across all data."""
    min_val, max_val = +inf, -inf
    reader = vtk.vtkXMLRectilinearGridReader()
    for idx, f in enumerate(fnm):
        if idx % 100 == 0:
            logging.info(f"Acquire dynamic range: {idx}/{len(fnm)}")
        _, _, _, filename = f
        reader.SetFileName(filename)
        reader.Update()

        # Get data from vtk object to numpy arrays
        data_linear = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0))

        # Calculate magnitude of the vector
        data_linear = np.square(data_linear[:, 0]) + np.square(data_linear[:, 1]) + np.square(data_linear[:, 2])
        min_val, max_val = min(min_val, min(data_linear)), max(max_val, max(data_linear))
    if min_val == max_val:
        max_val = min_val + 0.1

    return sqrt(min_val), sqrt(max_val)


def export_single(res_n: int, filecount: int, min_max: Tuple[float, float], fnm: Tuple[int, str, int, Path]) -> None:
    """Export single VTK file to PNG."""
    port_num, layer_id, idx, filename = fnm
    min_val, max_val = min_max
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get data from vtk object to numpy arrays
    x_coords = vtk_to_numpy(reader.GetOutput().GetXCoordinates())
    y_coords = vtk_to_numpy(reader.GetOutput().GetYCoordinates())
    data_linear = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0))
    org_shape = x_coords.shape[0], y_coords.shape[0]

    # Calculate magnitude of the vector
    data_linear = np.sqrt(np.square(data_linear[:, 0]) + np.square(data_linear[:, 1]) + np.square(data_linear[:, 2]))

    # Transform data from linear representation to matrix
    data = np.array(data_linear).reshape(org_shape, order="F")

    # Create an interpolation function to interpolate it into image
    interp = interpolate.RegularGridInterpolator((x_coords, y_coords), data, method="linear")

    # Calculate image shape
    shape = org_shape[0] * res_n, org_shape[1] * res_n
    xmin, xmax, ymin, ymax = min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    # Fill in every pixel with interpolated values
    imax = np.iinfo(np.uint16).max
    vmul = imax / sqrt(max_val - min_val)
    new_grid = np.mgrid[
        xmin : xmax : (1j * shape[0]),
        ymin : ymax : (1j * shape[1]),
    ]
    new_grid = new_grid.reshape(2, -1).T

    data_interp = interp(new_grid).reshape(shape)
    img_data = (np.sqrt(data_interp.T - min_val) * vmul).astype(np.uint16)

    # Save as image
    image = Image.fromarray(img_data, "I;16")
    path = Path.cwd() / "simulation_images" / f"{port_num}" / f"{layer_id}"

    path.mkdir(parents=True, exist_ok=True)
    path = path / f"e_field_{layer_id}_{idx}.png"
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image.save(path)
    with counter_lock:
        global progress_counter
        progress_counter.value += 1
        pc = progress_counter.value
    if pc % 100 == 0:
        logging.info(f"Processing files: {pc}/{filecount}")


def initializer(*args: Any) -> None:
    """Initialize progress counter inside thread pool."""
    global progress_counter, counter_lock
    progress_counter, counter_lock = args


@app.command()
def export_pngs(
    res_n: Annotated[int, typer.Option("--res_n", "-n", help="Resolution multiplier")] = 4,
) -> None:
    """Export pngs from e_field files."""
    coloredlogs.install(
        fmt="[%(asctime)s][%(levelname).4s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Load data with vtk
    filenames = list(Path.cwd().glob("**/e_field_*.vtr"))

    if len(filenames) == 0:
        raise Exception("No field data (**.vtr files) found!")

    idx = 0
    fnm = []
    old_box_id = ("", "")
    for filename in sorted(filenames):
        port_num, layer_id, timestep = re.findall(r"/(\d+)/e_field_(.+)_(\d+)", str(filename))[0]
        if old_box_id != (port_num, layer_id):
            idx = 0
            old_box_id = (port_num, layer_id)
        fnm.append((port_num, layer_id, idx, filename))
        idx = idx + 1

    avg = get_min_max(fnm)
    logging.info("Export data to png.")
    with Pool(initializer=initializer, initargs=(progress_counter, counter_lock)) as p:
        p.map(partial(export_single, res_n, len(fnm), avg), fnm)


if __name__ == "__main__":
    app()
