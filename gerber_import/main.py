import os
from pylab import *

from CSXCAD  import ContinuousStructure, CSPrimitives, CSProperties
from openEMS import openEMS
from openEMS.physical_constants import *

Sim_Path = os.path.join(os.getcwd(), 'tmp')
if not os.path.exists(Sim_Path):
    os.mkdir(Sim_Path)
post_proc_only = False
show_geometry = True

unit = 1e-6
f_start = 1e9
f_stop = 5e9
mesh_res = 50

FDTD = openEMS(NrTS=3e4)
FDTD.SetGaussExcite((f_start + f_stop)/2, (f_stop + f_start)/2)
FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)


# Getting points from gerber
import process_gbr
process_gbr.gbr_to_png("gerber_import-F_Cu.gbr", "top.png")
contours = process_gbr.get_outline("top.png")[0]

points = [[],[]]
for point in contours:
    points[0].append(point[0][0] * 10)
    points[1].append(point[0][1] * 10)

max_x = 10000
max_y = 22000
thickness = 75

print("Max:", max_x, max_y)

# Adding substrate
fr4 = CSX.AddMaterial('FR4', epsilon=4.2)
fr4.AddBox([0, 0, 0], [max(points[0]),max(points[1]), -thickness], priority=-1)

# Adding all metal
gnd = CSX.AddMetal('PEC')
gnd.AddBox([0, 0, -thickness], [max(points[0]),max(points[1]), -thickness], priority=1)

pec = CSX.AddMetal( 'PEC' )
pec.AddPolygon(points, "z", 0, priority=1)

# Adding ports
start_port1 = [350, 20800, -thickness]
stop_port1  = [350, 20700, 0]
port = FDTD.AddLumpedPort(1, 50, start_port1, stop_port1, 'z', excite=1, priority=100)

start_port2 = [8400, 630, -thickness]
stop_port2  = [8550, 630, 0]
port2 = FDTD.AddLumpedPort(2, 50, start_port2, stop_port2, 'z', excite=0, priority=101)

# Meshing
mesh.AddLine('x', [-1000, start_port1[0], stop_port1[0], start_port2[0], stop_port2[0], max_x+1000])
mesh.AddLine('y', [-1000, start_port1[1], stop_port1[1], start_port2[1], stop_port2[1], max_y+1000])
z_lines = [-z*1/6*thickness for z in range(7)] + [-1000, 1000]
mesh.AddLine('z', z_lines)
mesh.SmoothMeshLines('all', mesh_res, ratio=1.5)

# Dump box
Et = CSX.AddDump('Et')
start = [0, 0, -thickness/2]
stop  = [max(points[0])+1000, max(points[1])+1000, -thickness/2]
Et.AddBox(start, stop)

if show_geometry:  # debugging only
    CSX_file = os.path.join(Sim_Path, 'rect_wg.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)

if not post_proc_only:
    FDTD.Run(Sim_Path, cleanup=True)

freq = linspace(f_start,f_stop,201)
for port in [port, port2]:
    port.CalcPort(Sim_Path, freq)

s11 = port.uf_ref / port.uf_inc
s21 = port2.uf_ref / port.uf_inc
impedance = 50*(1-s11)/(1+s11)



figure()
plot(freq*1e-6,20*log10(abs(s11)),'k-',linewidth=2, label='$S_{11}$')
grid()
plot(freq*1e-6,20*log10(abs(s21)),'r--',linewidth=2, label='$S_{21}$')
legend();
ylabel('S-Parameter (dB)')
xlabel(r'frequency (MHz) $\rightarrow$')

figure()
plot(freq*1e-6,abs(impedance),'r--',linewidth=2, label='$Z$')
xlabel(r'frequency (MHz) $\rightarrow$')

show()