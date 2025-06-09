#!/usr/bin/env python3
"""Script running within Paraview."""

import os
from pathlib import Path
from paraview.simple import *  # type: ignore #noqa: F403
from typing import List, Dict


def get_sim_results(port: str) -> list[str]:
    """Get path to simulation result files, from specified port."""
    path = Path.cwd() / "ems" / "simulation" / port
    return [str(p) for p in path.glob("*.vtr")]


def run_preview(files: list[str]) -> None:
    """Set up and run preview."""
    paraview.simple._DisableFirstRenderCameraReset()  # type: ignore #noqa: F405

    render_view = GetActiveViewOrCreate("RenderView")  # type: ignore #noqa: F405
    grouped_files: Dict[str, List[str]] = {}
    for file in files:
        dump_name = Path(file).stem.rstrip("0123456789").rstrip("_")
        gf = grouped_files.setdefault(dump_name, [])
        gf.append(file)
        grouped_files[dump_name] = gf

    if len(grouped_files) == 0:
        raise Exception("No data files found!")

    for dump_name, f in sorted(grouped_files.items()):
        e_field = XMLRectilinearGridReader(registrationName=dump_name, FileName=sorted(f))  # type: ignore #noqa: F405
        e_field.PointArrayStatus = ["E-Field"]

        animation_scene = GetAnimationScene()  # type: ignore #noqa: F405

        animation_scene.UpdateAnimationUsingDataTimeSteps()

        e_field_vtr_display = Show(e_field, render_view, "UniformGridRepresentation")  # type: ignore #noqa: F405

        ColorBy(e_field_vtr_display, ("POINTS", "E-Field", "Magnitude"))  # type: ignore #noqa: F405

        Hide()  # type: ignore #noqa: F405

    Show()  # type: ignore #noqa: F405

    e_field_vtr_display.RescaleTransferFunctionToDataRangeOverTime()

    e_field_vtr_display.SetScalarBarVisibility(render_view, True)

    render_view.Update()

    render_view.ResetCamera(False)

    animation_scene.Play()


if __name__ == "__main__":
    port = os.environ["GERBER2EMS_PORT"]

    files = get_sim_results(port)
    run_preview(files)
