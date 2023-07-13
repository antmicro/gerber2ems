unit = 1e-6

pcb_width = 10000 # Width of the PCB (in x direction)
pcb_height = 15000 # Height of the PCB (in y direction)
substrate_thickness = 200 # Thickness of the substrate (in z direction)

substrate_epsilon = 4.2 # Dielectric constant of the substrate material

border_thickness = 100


port1_position = [5000, 800] # Center position of port 1
port1_direction = 'y' # Di
port1_width = 400
port1_length = 1000
port1_impedance = 50
port2_position = [5000, 13200] # Center position of port 2
port2_direction = '-y'
port2_width = 400
port2_length = 1000
port2_impedance = 50

margin = 4000
z_margin = 4000

f_start = 2e9
f_stop = 20e9

mesh_max = 50
mesh_margin = 200
mesh_z_max = 200
mesh_z_substrate = 10

## Initialization
import CSXCAD
import openEMS
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
result_path = os.path.join(os.getcwd(), 'result')
CSX = CSXCAD.ContinuousStructure()
FDTD = openEMS.openEMS(NrTS=2e4)
FDTD.SetCSX(CSX)
FDTD.SetGaussExcite((f_start + f_stop)/2, (f_stop - f_start)/2)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

## Adding ports
def addPort(number, position, direction, width, length, impedance, excite=False):
    if 'y' in direction:
        start = [position[0]-width/2, position[1], 0]
        stop = [position[0]+width/2, position[1]+length, -substrate_thickness]

    elif direction == 'x':
        print("Not handled yet")
        sys.exit()

    if '-' in direction:
        start[0:2], stop[0:2] = stop[0:2], start[0:2]
    pec = CSX.AddMetal( 'PEC' )
    if excite:
        return FDTD.AddMSLPort(number, pec, start, stop, 'y', 'z', Feed_R=impedance, priority=100, excite=1)
    else:
        return FDTD.AddMSLPort(number, pec, start, stop, 'y', 'z', Feed_R=impedance, priority=100)


def addMesh():
    x_lines = [-margin, pcb_width+margin]
    x_lines = np.concatenate((x_lines, np.arange(0-mesh_max/2, pcb_width+mesh_max/2, step=mesh_max)))
    mesh.AddLine('x', x_lines)

    y_lines = [-margin, pcb_height+margin]
    y_lines = np.concatenate((y_lines, np.arange(0-mesh_max/2, pcb_height+mesh_max/2, step=mesh_max)))
    mesh.AddLine('y', y_lines)

    z_lines = [-substrate_thickness-z_margin, 0, z_margin]
    z_lines = np.concatenate((z_lines, np.arange(-substrate_thickness, 0, step=mesh_z_substrate)))
    z_lines = np.concatenate((z_lines, [-substrate_thickness/2]))
    mesh.AddLine('z', z_lines)

    mesh.SmoothMeshLines('x', mesh_margin, ratio=1.5)
    mesh.SmoothMeshLines('y', mesh_margin, ratio=1.5)
    mesh.SmoothMeshLines('z', mesh_z_max, ratio=1.2)

def addPlane(z):
    plane = CSX.AddMetal( 'Plane' )
    plane.AddBox([0, 0, z], [pcb_width, pcb_height, z], priority=1)

def addSubstrate(start, stop, epsilon):
    substrate = CSX.AddMaterial('Substrate', epsilon=epsilon)
    substrate.AddBox([0, 0, start], [pcb_width, pcb_height, stop], priority=-1)

def addGerber():
    import process_gbr
    process_gbr.gbr_to_png("gerber_import-F_Cu.gbr", "top.png")
    contours = process_gbr.get_outline("top.png")[0]

    points = [[],[]]
    for point in contours:
        points[0].append((point[0][0] * 10)-border_thickness/2)
        points[1].append((point[0][1] * 10)-border_thickness/2)

    gerber = CSX.AddMetal( 'Gerber' )
    gerber.AddPolygon(points, "z", 0, priority=1)

def addDumpBox():
    Et = CSX.AddDump('Et', sub_sampling=[1,1,1])
    start = [-margin, -margin, -substrate_thickness/2]
    stop  = [pcb_width+margin, pcb_height+margin, -substrate_thickness/2]
    Et.AddBox(start, stop)

def setBoundaryConditions():
    FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])

## Write geometry to file
def saveGeometry():
    CSX_file = os.path.join(result_path, 'geometry.xml')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    CSX.Write2XML(CSX_file)

def simulate():
    FDTD.Run(result_path, cleanup=True)

def getSParameters(port1, port2, freq):
    port1.CalcPort(result_path, freq)
    port2.CalcPort(result_path, freq)

    s11 = port1.uf_ref / port1.uf_inc
    s21 = port2.uf_ref / port1.uf_inc

    return (s11, s21)


addMesh()
port1 = addPort(1, port1_position, port1_direction, port1_width, port1_length, port1_impedance, excite=True)
port2 = addPort(2, port2_position, port2_direction, port2_width, port2_length, port2_impedance)
addPlane(-substrate_thickness)
addSubstrate(-substrate_thickness, 0, substrate_epsilon)
addGerber()
addDumpBox()
setBoundaryConditions()

simulate()

freq = np.linspace(f_start, f_stop, 1001)
s11, s21 = getSParameters(port1, port2, freq)

plt.figure()
plt.plot(freq, 20*np.log10(abs(s11)))
plt.plot(freq, 20*np.log10(abs(s21)))

impedance = 50*(1-s11)/(1+s11)

plt.figure()
plt.plot(freq, impedance)

plt.figure()
from smithplot import SmithAxes
ax = plt.subplot(1, 1, 1, projection='smith')
plt.plot(impedance, datatype=SmithAxes.Z_PARAMETER, marker=" ")


plt.show()

saveGeometry()
