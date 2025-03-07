#!/usr/bin/env python3
"""Script running within Paraview."""

import os
from pathlib import Path
from paraview.simple import *  # type: ignore #noqa: F403


def get_sim_results(port: str) -> list[str]:
    """Get path to simulation result files, from specified port."""
    path = Path.cwd() / "ems" / "simulation" / port
    return [str(p) for p in path.glob("*.vtr")]


def run_preview(files: list[str]) -> None:
    """Set up and run preview."""
    paraview.simple._DisableFirstRenderCameraReset()  # type: ignore #noqa: F405

    e_field = XMLRectilinearGridReader(registrationName="e_field", FileName=files)  # type: ignore #noqa: F405
    e_field.PointArrayStatus = ["E-Field"]

    animation_scene = GetAnimationScene()  # type: ignore #noqa: F405

    animation_scene.UpdateAnimationUsingDataTimeSteps()

    render_view = GetActiveViewOrCreate("RenderView")  # type: ignore #noqa: F405

    e_field_vtr_display = Show(e_field, render_view, "UniformGridRepresentation")  # type: ignore #noqa: F405

    ColorBy(e_field_vtr_display, ("POINTS", "E-Field", "Magnitude"))  # type: ignore #noqa: F405

    e_field_vtr_display.RescaleTransferFunctionToDataRange(True, False)

    e_field_vtr_display.SetScalarBarVisibility(render_view, True)

    render_view.Update()

    render_view.ResetCamera(False)

    animation_scene.Play()


if __name__ == "__main__":
    port = os.environ["GERBER2EMS_PORT"]

    files = get_sim_results(port)
    run_preview(files)
