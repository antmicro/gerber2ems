#!/usr/bin/env python3
import os
import sys

from paraview.simple import *


def get_sim_results(layer: str) -> list[str]:
    path = os.getcwd() + "/ems/simulation/" + layer
    files = os.listdir(path)

    files = [path + "/" + file for file in files if file[-4:] == ".vtr"]
    return files


def run_preview(files: list[str]) -> None:
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


def run() -> None:
    layer = os.environ["GERBER2EMS_PREVIEW_LAYER"]

    files = get_sim_results(layer)
    run_preview(files)


run()
