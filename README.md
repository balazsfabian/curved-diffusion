# CurD : diffusion on curved surfaces from molecular dynamics

## Usage example

### Prerequisites:
The following python modules are required:

### Step 0: pre-processing of the trajectory
The first configuration of the trajectory should be preprocessed to have all
particles correctly wrapped inside the simulation box, and the entire trajectory
should be `nojump`. Because in most cases, the topological feature is not subject
to thermostatting, there is a divide between the actual box size and the periodicity
of the topology. The user can either save the periodicity of the topology upon setting
up the simulation and use it as input in Step 1, or let the code average over the first
couple of frames to get and estimate. Additionally, for the sake of speed, users are
encouraged to use trajectories containing only the molecular centers of mass.

### Step 1: creating a mesh surface
Generation of surface mesh from the trajectory. For possible options, run the script with
the `--help` flag. The current version is meant for 2D PBC surfaces, although extensions
to nonperiodic (closed) surfaces should be trivial. An example is:

```bash
python 01-generate-mesh.py conf.gro traj.xtc --leaflet upper --lg "name PC" --average X --output mesh.vtk --lxy 343.98 343.98
```
**What it does:** Based on the selection `--lg "name PC"`, and using the default `--cutoff 12` identifies `--leaflet upper`. As the
xy-periodic lattice spacing is known, it is used as input `--lxy 343.98 343.98` instead
of trying to figure it out. Because our example trajectory is assumed to have
only mean curvature along the y axis, the molecular positions are averaged over the x axis
in order to smoothen the resulting surface. This can also be done along the y axis. After
some smooting of the resulting surface, the PBC image is repeated twice (default) along both
x and y axes as required by the MSD calculation. Finally, the mean and Gaussian curvatures
are evaluated, the metadata is added to the mesh, and its suitability for the VTP algorithm
(see below) is checked.

---
**NOTES**

* The code always uses the CoM of the selected residues. To avoid this, one can either modify the code, or just create a subset of the trajectory
only containing a single bead per molecule.
* To end up with more-or-less uniform mesh spacing on the actual surface, the external `pyacvd` algorithm is used. However, it sometimes goes wrong
and creates "redundant half-edges", essentially dangling triangles. I haven't figures out a nice way of removing it... The current solution is to
tweak the desired number of meshpoints (`--pts`) or the number of triangular subdivision in `pyacvd` (`--nsub`)

---


### Step 2: assigning molecules to surface mesh vertices
As VTP can measure distances between vertices on the mesh, all molecular displacements (pairs of initial and final positions) must be mapped onto
the mesh surface from Step 1. The resulting v<sub>i</sub> and v<sub>j</sub> vertex pairs are written to binary file along with the associated lagtime. An example:
``` bash
python 02-spread-on-mesh.py mesh.vtk --trestart 10
```

This way, the given square displacements can be binned both spatially by assigning it to both vertices, and based on lagtime.
The latter is an essential requirement in the calculation of MSD curves. The turnover of (initial position, final position) pairs in  MSD calculations is **huge**, on the order of terabytes. During the conventional MSD calculations only a tiny fraction needs to be kept in memory. Unfortunately, for the sake of efficiency in the VTP algorithm, we do need to store this information in the from of (starting vertex, end vertex, lagtime) as the suitable sorting of these triplets is an essential step of the method. To limit the burden of storing terabyte-sized files, we chose to evalute the MSD at only reduced number of lagtimes. Currently, this means uniform logarithmic spacing of lagtimes. However, as the dynamics of interest are already somewaht constrained to occur on length scales commensurate with the lateral dimensions of the simulation box, evaluating all lagtimes is indeed a possibility, although without much added information.

The name of the binary 'indexfile' is stored in the vtk mesh file. Its name (along with other useful properties of the mesh) can be checked using
``` bash
python mesh-info-py mesh.vtk indexfile
```

### Step 3:
The final step of the computation orders the indexfile. For this, the `bsort` git-submodule needs to be installed and the path in `03-parallel.py` must be updated. Similarly, the `VTP` source code must be compiled, and its location should be updated in `geodesic.py`'s `vtp_path` variable.
