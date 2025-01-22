import os
from paraview.simple import *

path = os.getcwd() + "/ems/simulation/0"
files = os.listdir(path)

files = [path + "/" + file for file in files if file[-4:] == ".vtr"]

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
