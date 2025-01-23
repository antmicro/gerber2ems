#!/usr/bin/env python3
import os
from pathlib import Path

from paraview.simple import *


def get_sim_results(port: str) -> list[str]:
    """Get path to simulation result files, from specified port"""
    # path = Path.cwd() / "ems" / "simulation" / port
    # return list(path.glob("*.vtr"))

    path = os.getcwd() + "/ems/simulation/" + port
    files = os.listdir(path)

    files = [path + "/" + file for file in files if file[-4:] == ".vtr"]

    return files


def run_preview(files: list[str]) -> None:
    """Setup and run preview"""
    paraview.simple._DisableFirstRenderCameraReset()

    e_field = XMLRectilinearGridReader(registrationName="e_field", FileName=files)
    e_field.PointArrayStatus = ["E-Field"]

    animationScene = GetAnimationScene()

    animationScene.UpdateAnimationUsingDataTimeSteps()

    renderView = GetActiveViewOrCreate("RenderView")

    e_field_vtrDisplay = Show(e_field, renderView, "UniformGridRepresentation")

    ColorBy(e_field_vtrDisplay, ("POINTS", "E-Field", "Magnitude"))

    e_field_vtrDisplay.RescaleTransferFunctionToDataRange(True, False)

    e_field_vtrDisplay.SetScalarBarVisibility(renderView, True)

    renderView.Update()

    renderView.ResetCamera(False)

    animationScene.Play()


if __name__ == "__main__":
    port = os.environ["GERBER2EMS_PORT"]

    files = get_sim_results(port)
    run_preview(files)
