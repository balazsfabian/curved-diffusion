import sys
import argparse
import numpy as np
import MDAnalysis as mda

from itertools import product
from MDAnalysis.analysis.leaflet import LeafletFinder

import scipy
import scipy.stats as st
from scipy.interpolate import griddata
from scipy.signal import convolve2d as convolve
from scipy.spatial import Delaunay

import pyvista as pv
import pyacvd


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel.
    """

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def border(x,y, box, xrepeat=[-1,0,1], yrepeat=[-1,0,1]):
    """ Repeat the periodic box by offseting the coordinates
    """
    out1 = []
    out2 = []
    for elem in product(xrepeat,yrepeat):
        px = x[:,0]+elem[0]*box[0]
        py = x[:,1]+elem[1]*box[1]
        out1.append(np.array((px,py)).T)
        out2.append(y)

    return np.concatenate(np.array(out1),axis=0),  np.concatenate(np.array(out2),axis=0)


def make_surface_function(func,lx,ly):
    """ This is a factory function: it returns a PBC version of the interpolated surface
    """
    def interp(x,y):
        """ Returns the PBC wrapped values of the x,y points. The arrays x and y can be arrays: in this case, the result is also an array """
        x = np.mod(x,lx)
        y = np.mod(y,ly)
        result = func(x,y, grid=False).ravel()
        if len(result) == 1:
            result = result.item()
        return result

    return interp


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def CheckLeafType(leaf):
    """ Check leaflet in 'upper'/'lower'/'both'.
    """
    if leaf not in ['upper', 'lower', 'both']:
        raise ValueError()
    return leaf


def CheckAverageAxis(ax):
    """ Check axis in 'x'/'y'.
    """
    ax = ax.lower()
    if ax not in ['x','y']:
        raise ValueError()
    return ax


desc = """Create a VTK triangular mesh from a membrane bilayer.

The program accepts gromacs gro & tpr files as topology, and xtc & trr as trajectory.
Meshes can be generated for the upper and lower leaflets, separately, or both at the
same time. As reference points, the molecular CoM-s of the residues containing the 'lg'
sites are used. Currently, only the first frame is used in meshing.

The macroscopic plane of the membrane is always assumed to be the xy plane.The code
can \"flatten\" the mesh by ignoring the z coordinate. It can also eliminate any variation
along the x or y axes, thus preserving the symmetry and smoothness of developable surfaces.

As curved simulations are performed **without** refcoord-scaling but on NPT, there is
difference between using the box length at the first frame and the periodicity of the
topological features. Here, either the average box dimensions are used, or the proper
dimensions can be provided by using "--lxy" followed by the x and y dimensions (angstroms).
"""

parser = argparse.ArgumentParser(description=desc, formatter_class=CustomFormatter)
parser.add_argument('topology')
parser.add_argument('trajectory')
parser.add_argument('--leaflet', default='both'    , type=CheckLeafType    , help="Leaflet to mesh. The default value creates a single mesh of both layers.")
parser.add_argument('--lg'     , default='name PO4', type=str              , help="Atomtype for leaflet-identification. Preferably the headgroup.")
parser.add_argument('--pts'    , default=100**2    , type=int              , help="Approximate number of meshpoints to use per periodic image.")
parser.add_argument('--repeat' , default=2         , type=int              , help="Repeate the unit box this many times along the x & y axes.")
parser.add_argument('--nsub'   , default=3         , type=int              , help="Number of subdivisions in pyacvd before uniformizing the surface")
parser.add_argument('--output' , default='mesh.vtk', type=str              , help="Name of the output VTK file.")
parser.add_argument('--average', default=False     , type=CheckAverageAxis , help="Take the average value of the surface along the specified direction ('x'/'y'). Remove artifical faces at the edges.")
parser.add_argument('--flat'   , default=False     , action='store_true'   , help="Create dummy mesh by setting the z direction to 0.")
parser.add_argument('--ndx'    , default=False     , action='store_true'   , help="Write Gromacs-style ndx files with leaflets")
parser.add_argument('--lxy'    , action='append'   , nargs=2, type=float   , help="Specify the lx and ly dimensions of the topological feature")

args = parser.parse_args()


##########################################
# set up the universe
##########################################

universe = mda.Universe(args.topology, args.trajectory)

print ()

# Identify leaflets (if necessary)
if args.leaflet != 'both':

    print (f"* Identifying {args.leaflet} leaflet:")

    lf = LeafletFinder(universe, args.lg, pbc=True)
    assert len(lf.groups()) == 2, f"Could not separate {args.trajectory} into upper and lower leaflets..."
    if args.leaflet == 'upper':
        indices = lf.groups(0).indices
    if args.leaflet == 'lower':
        indices = lf.groups(1).indices
    selection = universe.atoms[indices]

    print (f"    Selected {selection.__repr__()}")

    if args.ndx:

        print (f"    Writing selection to leaflets.ndx")

        with mda.selections.gromacs.SelectionWriter('leaflets.ndx', mode='w') as ndx:
            ndx.write(lf.groups(0), name='upper')
            ndx.write(lf.groups(1), name='lower')

else:
    selection = universe.select_atoms(args.lg)

    print (f"    Selected {selection.__repr__()}")


# Average box parameters
if args.lxy is not None:
    lxy = args.lxy[0]
    lx = lxy[0]
    ly = lxy[1]
else:
    box = np.zeros(3)
    for ts in universe.trajectory:
        box += ts.dimensions[:3]

    box /= universe.trajectory.n_frames
    lx = box[0]
    ly = box[1]

print ()
print (f"* Box dimensions:")
print (f"    Lx = {lx}")
print (f"    Ly = {ly}")


##########################################
# gridding the data
##########################################

print ()
print ("* Gridding the positions")

# approximate number of points along x
n = int(np.floor(np.sqrt(args.pts)))
# single-box grid
grid_x, grid_y = np.mgrid[0:lx:(n+1)*1j, 0:ly:(n+1)*1j]
grid_x = grid_x[:-1,:-1]
grid_y = grid_y[:-1,:-1]

print (f"    Using the CoM-s of {selection.residues.__repr__()}")

pos = []
for ts in universe.trajectory[:10]:
    p = np.array([res.atoms.center_of_mass() for res in selection.residues])
    pos.append(p)

pos = np.vstack(pos)

print (f"    Collected {len(pos)} points for gridding")

# xy-coords
pts = pos[:,0:2]

# z-coord
if args.flat:
    print (f"    Projecting onto the Z plane")
    z = np.zeros(pos[:,0].shape)
else:
    z = pos[:,2]

pts,z = border(pts,z,universe.dimensions[0:3])
zi = griddata(pts, z, (grid_x, grid_y), method = 'linear')

if args.average:
    print (f"    Averaging the positions along {args.average}")
    if args.average == 'x':
        zi = np.tile(zi.mean(axis=0), (n,1) )
    if args.average == 'y':
        zi = np.tile(zi.mean(axis=1), (n,1) )

##########################################
# Smooth the positions (with PBC)
##########################################

print ()

# Smoothing kernel parameters
n_gauss = 7
n_sig = 3
gauss_mat = np.ones((n_gauss,n_gauss))
gauss_mat /= np.sum(gauss_mat)
gauss_mat = gkern(kernlen=n_gauss, nsig=n_sig)

print (f"* Smoothing the surface using a gaussian kernel of size={n_gauss} and sigma={n_sig}")

zi = convolve(zi, gauss_mat, mode="same", boundary='wrap')


##########################################
# Create an PBC interpolating function
##########################################

print ()
print (f"* Creating the PBC iterpolating function")
interFunction = scipy.interpolate.RectBivariateSpline(grid_x[:,0], grid_y[0,:], zi)
pbcFunction = make_surface_function(interFunction, lx, ly) 


##########################################
# PBC repetition
##########################################

print ()
print (f"* Repeating the simulation box {args.repeat} times along x and y")
largeGridScale = args.repeat
n_large = n*largeGridScale + 1
large_grid_x, large_grid_y = np.mgrid[0:lx*largeGridScale:n_large*1j, 0:ly*largeGridScale:n_large*1j]
large_grid_x = large_grid_x[:-1,:-1]
large_grid_y = large_grid_y[:-1,:-1]

large_zi = pbcFunction(large_grid_x.flatten(),large_grid_y.flatten())

##########################################
# Remeshing to get a more uniform triangle size
##########################################

print ()
print (f"* Re-meshing for more uniform triangle sizes")
print (f"    Subdividing triangles by a factor of {args.nsub}")
# Mesh points and triangular faces
vertices = np.array((large_grid_x.flatten(),large_grid_y.flatten(),large_zi.flatten())).T
faces = Delaunay(vertices[:,:2]).vertices
mesh = pv.PolyData(vertices, np.insert(faces, 0, 3, axis=1))

# create uniform mesh in 3D
clus = pyacvd.Clustering(mesh)
clus.subdivide(args.nsub)
clus.cluster((n*largeGridScale)**2)
mesh = clus.create_mesh()

##########################################
# Evaluating curvature
##########################################

print ()
print ("* Evaluating the curvature")
n_iter = 1000
print (f"    Creating a smoothed copy using {n_iter} Laplacian iterations")
mesh.smooth(n_iter=n_iter, boundary_smoothing = False, inplace = True)

curvatures = ['Mean', 'Gaussian']
for c in curvatures:
    print (f"    Computing {c} curvature")
    values = mesh.curvature(curv_type=c)
    mesh.point_arrays[c] = values

area_ratio = (mesh.compute_cell_sizes().cell_arrays['Area'].sum()/largeGridScale**2) / (lx*ly)
print (f"    The ratio of surface- and projected areas: {area_ratio}")

##########################################
# Verification with VTP
##########################################
print ()
print ("* Verifying the mesh for use with VTP")

from geodesic import ExactGeodesicMixin

pts = mesh.points
faces = mesh.faces.reshape((-1,mesh.faces[0]+1))[:,1:]
vtp = ExactGeodesicMixin(pts,faces)
distances = vtp.exact_geodesic_distance(0)

##########################################
# Adding metadata
##########################################

print ()
print (f"* Loading metadata into {args.output}")
mesh.field_arrays['topology'] = (args.topology,)
mesh.field_arrays['trajectory'] = (args.trajectory,)
mesh.field_arrays['select'] = (args.lg,)
mesh.field_arrays['lx'] = lx
mesh.field_arrays['ly'] = ly
mesh.field_arrays['leaflet'] = (args.leaflet,)
mesh.save(args.output)

print ("DONE!")

#
# NOTE: this section seems unnecessary. If redundant half-edges appear, try using a different mesh size first!
#
# # remove redundant half-edges. Not an elegant solution to a problem I refuse to understand now
# faces = Delaunay(mesh.points[:,:2]).vertices
# mesh = pv.PolyData(mesh.points, np.insert(faces, 0, 3, axis=1))
# mesh.save('tmp2-'+args.output)
# 
# # remove vertical faces along the edges
# if args.average:
#     print ()
#     print (f"* Removing vertical faces along {args.average}")
#     mesh = mesh.compute_normals()
#     normals = mesh.cell_arrays['Normals']
#     if args.average == 'x':
#         norm = np.abs(normals[:,0])
#     if args.average == 'y':
#         norm = np.abs(normals[:,1])
#     faces = faces[norm < 0.5]
#     mesh = pv.PolyData(mesh.points, np.insert(faces, 0, 3, axis=1))
