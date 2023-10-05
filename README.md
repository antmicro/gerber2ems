# OpenEMS simulation with KiCAD
## Installation
Needed tools are:
* `KiCAD` - editing the PCB
* `OpenEMS` - simulation
* `Paraview` - viewing simulation results (creating images etc.)
* Script from this repository

To install, first install `KiCad`, then download, compile and install `OpenEMS`: https://docs.openems.de/install.html#linux. After that install `Paraview`: https://www.paraview.org/. Finally instal the script using `install.sh`

## Flow
Simulation flow consists of multiple independent steps:
#### PCB Preparation
Simulating the whole PCB is extremaly resource intensive, therefore separating the region of interest is very important. Size should be as small as possible. Uneeded traces, pours etc. should be removed. If whole layers are unneeded they can be removed in later steps.

* Ports of interest should be marked using special simulation port footprint. Rotation and position should be appropriate. (It's reference number tells the simulator which port to assign to it).

* Origin point for drill files should be placed in bottom-left corner.

* Every trace, pour that is somehow terminated in reality and will exist in the simulation should also be terminated using simulation port or connected to ground.

* For now, capacitors are not simulated and for high frequency simulation they can be aproximated by shorting them using a trace.

#### `simulation.json` preparation
`simulation.json` file configures the whole simulation. Example files can be found in `example_gerbers` folder. All dimensions in this file are specified in **micrometers**. There are a few sections to this config file:
##### Miscellaneous
* `format_version` - specifies with which format the config file was written. When writing a new config this should be the newest supported version (this number can be found in `constants.py` file).
* `frequency` - `start` specifies the lowest frequency of interest and `stop` the highest.
* `max_steps` - max number of simulation steps after which the simulation will stop, no matter what.
* `via/plating_thickness` - thickness of via plating.
* `via/filling_epsilon` - dielectric constant of the material the vias are filled in with. If they are not filled in, it should be 1.
* `margin/xy` - margin (area added outside the board) size in x and y directions.
* `margin/xy` - margin size in z direction.
##### Mesh
* `xy` - mesh grid size in x and y direction (micrometers).
* `inter_layers` - number of mesh lines in z direction between neighbouring pcb layers.
* `margin/xy` - mesh grid size in x and y direction (micrometers) outside of the board area.
* `margin/z` - mesh grid size in z direction (micrometers) outside of the board area.
##### Ports
`ports` is a port-number to port-parameters map. Each port has multiple parameters:
* `width` - width of the port.
* `length` - length of the port (right now port are fragments of microstriplines. Their length should be at least 8x mesh cell size).
* `impedance` - terminating impedance of the port (impedance of driver or receiver).
* `layer` - layer number on which the port is.
* `plane` - layer number on which reference plane of the microstrip is.
* `excite` - whether simulator should use this port as an input port. (if there are multiple excited ports, they will be excited in separate simulations)

### KiCAD data export
This step is done fully automatically when `-k` flag is passed to `ems-kicad` script.
Script calls `kmake gerber -x` to generate gerbers and drill files, `kmake pnp -v` to get positions of the ports and `kmake stackup-export` to get layer information.
If you want to remove unneeded layers from the simulation, you should do it now, by removing them from stackup file.

### Geometry creation
This is an automatic step done with `-g` flag.
Script locates all the files needed for creating the geometry. (gerbers, drill files, pnp files, stackup file, simulation config file). After that it converts gerbers to png's using gerbv. Then the PNG's are processed into triangles and input into the geometry. It also adds via geometries as well as port geometries. Everything is placed on correct Z heights using the stackup file.

You can view the generated geometry which is saved to `ems/geometry/geometry.xml` file using AppCSXCAD app which was installed during OpenEMS installation.

### Simulation
This is an automatic step done with `-s` flag.
Script loads the geometry and config files. It inputs all the information into the engine and starts the simulations, iterating over every "excited" port.

User should verify if selected number of timesteps is enough. The engine recommends that it should be at least 3x as long as the pulse:
```
Excitation signal length is: 3211 timesteps (3.18343e-10s)
Max. number of timesteps: 10000 ( --> 3.11429 * Excitation signal length)
```

Simulator converts the geometry into voxels then starts solving the Maxwell equations for each edge in the mesh. It does that for a number of timesteps (maximum number specified in config) and then exits. For each timestep electric field date from planes between copper planes is saved to files in `ems/simulation` folder. Port voltage and current data is also saved.

During the simulation one of the ports is exited using a gaussian pulse (wideband frequency content). This pulse traverses the network and exits using ports (it can also get emitted outside of the board). 

You can monitor the simulation looking at the engine output:
```
[@ 20s] Timestep: 4620 || Speed:  294.4 MC/s (3.372e-03 s/TS) || Energy: ~4.16e-16 (- 7.15dB)
```
You can see what timestep are you on. How many mesh voxels per second the simulator processes. How much energy is left in the system. The energy should drop during the simulation as it exits through the ports (after the excitation pulse ends). However due to inaccuracies and energy radiated it won't drop to 0.

After the simulation finishes, user can verify using `Paraview` if it makes sense. How to do this is described later.

### Postprocessing
This is an automatic step done with `-p` flag.
Scripts loads in the simulator data for each excited port. Then it computes an FFT to get data in frequency domain. It then converts the incident and reflected wave data to impedance and S parameters. Those are saved in csv format in `ems/results/S-parameters` folder. This data is also automatically ploted and those plots are saved to `ems/results`


## Paraview
To view simulation data in Paraview follow these steps:
* Open Paraview
* Load the data you are interested in (File->Open...). The files are located in `ems/simulation/<port_number>/e_field_<layer_number>_..vtr`
* Show loaded in data. Eye symbol in `Pipeline Browser` on the left side of the window.
![](./docs/images/eye.png)
* Change `Coloring` to `E-Field`. Properties window on the left side.
![](./docs/images/color.png)
* Rescale to data range. Second icon bar from the top.
![](./docs/images/range.png)
* Press Play to play the animation. First icon bar from the top.
![](./docs/images/play.png)