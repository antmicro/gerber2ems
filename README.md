# OpenEMS simulation with KiCAD
## Installation
Needed tools are:
* `KiCAD` - editing the PCB
* `Octave` - interfacing with openEMS, setting up simulation, postprocessing and ploting some results
* `OpenEMS` - simulation
* `Paraview` - viewing simulation results (creating images etc.)
* `pcbmodelgen` - importing PCB's from kicad to openems

To install, first install `KiCad` and `Octave`. Then download, compile and install `OpenEMS`: https://docs.openems.de/install.html#linux. After that install `Paraview`. Finally download, compile and install `pcbmodelgen`: https://github.com/jcyrax/pcbmodelgen

## Setup
* Prepare a kicad PCB file that has the feature of interest isolated to save on computation time. (unfortunately, more than two layers are not supported, as well as curved tracks). 
* Select places to place ports an save their coordinates.
* Edit pcbmodels `config.json`. At the minimum set PCB thickness and simulation box.
* Run pcbmodelgen: ```pcbmodelgen -p pcb_file.kicad_pcb -c config.json -m kicad_pcb_model.m -g kicad_pcb_mesh.m```
* Setup ports in `simulation_script.m`. They are modeled as rectangular resistors, you need to supply two corners' coordinates.
* Setup data dump box'es in `simulation_script.m`.
* Setup other simulation parameters as well as postprocessing that may be needed.
* Run simulation ```octave --silent --persist simulation_script.m```
* To view dumped data use `paraview`. (e.g. load dumped data, set color to be proportional to magnitude)