import argparse

import trimesh
import numpy as np
import pyvista as pv
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from tqdm import tqdm


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def CheckLeafType(leaf):
    """ Check leaflet in 'upper'/'lower'/'both'.
    """
    if leaf not in ['upper', 'lower', 'both']:
        raise ValueError()
    return leaf


desc = """Calculate density on surface mesh.
"""

parser = argparse.ArgumentParser(description=desc, formatter_class=CustomFormatter)
parser.add_argument('mesh')
parser.add_argument('trajectory')
parser.add_argument('--start'    , default=None    , type=int            , help="First frame to analyze.")
parser.add_argument('--step'     , default=None    , type=int            , help="Skip every n frames.")
parser.add_argument('--stop'     , default=None    , type=int            , help="Last frame to analyze.")
parser.add_argument('--lg'       , default=None    , type=str            , help="Use this group for the density calculation. If not specified, then use to one in the mesh.")
# parser.add_argument('--leaflet', default='both', type=CheckLeafType, help="Leaflet to mesh. The default value creates a single mesh of both layers.")
# parser.add_argument('--lg', default='name PO4', type=str, help="Atomtype for leaflet-identification. Preferably the headgroup.")

args = parser.parse_args()


##########################################
# set up the mesh
##########################################
mesh = pv.read(args.mesh)

topology = mesh.field_arrays['topology'][0]
trajectory = args.trajectory
if args.lg:
    select = args.lg
else:
    select = mesh.field_arrays['select'][0]
leaflet = mesh.field_arrays['leaflet'][0]
cutoff = mesh.field_arrays['cutoff'][0]
lx = mesh.field_arrays['lx'][0]
ly = mesh.field_arrays['ly'][0]

# assuming triangular mesh
faces = mesh.faces.reshape((-1,4))[:,1:]
# VTK Cell Locator does not work
tmesh = trimesh.Trimesh(vertices = mesh.points, faces=faces)


##########################################
# set up the universe
##########################################
universe = mda.Universe(topology, trajectory)


if leaflet != 'both':
    lf = LeafletFinder(universe, select, pbc=True, cutoff=cutoff)
    assert len(lf.groups()) == 2, f"Could not separate {trajectory} into upper and lower leaflets..."
    if leaflet == "upper":
        indices = lf.groups(0).indices
    if leaflet == "lower":
        indices = lf.groups(1).indices
    selection = universe.atoms[indices]
else:
    selection = universe.select_atoms(select)

print ()
print (f"* Calculating density on mesh {args.mesh}")
print (f"   topology: {topology}")
print (f"   trajectory: {args.trajectory}")
print (f"   leaflet: \'{leaflet}\' based on group {select}")
print (f"   Selected {selection.__repr__()}")


cntFrame = 0
hist = np.zeros(len(faces))
try:
    # setting up a slice
    sl = slice(args.start,args.stop,args.step)

    for ts in tqdm(universe.trajectory[sl]):
        cntFrame += 1

#       pos = [res.atoms.center_of_mass() for res in selection.residues]
        pos = selection.positions
    
        _, _, triangle_id = tmesh.nearest.on_surface(pos)

        bbins = np.bincount(triangle_id)
    
        hist[:len(bbins)] += bbins

finally:
    
    area = mesh.compute_cell_sizes(length=False,area=True,volume=False).cell_arrays['Area']
    
    density = np.divide(hist,area) / cntFrame
    density[density==0] = np.nan
    mesh.cell_arrays['Density'] = density
    mesh.save(args.mesh)
