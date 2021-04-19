#!/home/balazs/.conda/envs/mdanalysis/bin/python

import argparse
import numpy as np
import pyvista as pv


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def CheckAverageDirection(ax):
    """ Check axis in 'x'/'y'.
    """
    ax = ax.lower()
    if ax not in ['x','y','radial']:
        raise ValueError()
    return ax


desc = """Calculate averages over the specified directions

The possible option are:
    * 'x': Average along the x axis
    * 'y': Average along the y axis
    * 'radial': Radial average
"""

parser = argparse.ArgumentParser(description=desc, formatter_class=CustomFormatter)
parser.add_argument('mesh')
parser.add_argument('--direction', default='x',    type=CheckAverageDirection, help="Direction for averaging")
parser.add_argument('--bins',      default=50,     type=int,                   help="Number of bins")
parser.add_argument('--name',      default='mesh', type=str,                   help="Stem of the output names")

args = parser.parse_args()


mesh = pv.read(args.mesh)

lagtimes = mesh.field_arrays['lagtimes']
lx = mesh.field_arrays['lx'][0]
ly = mesh.field_arrays['ly'][0]
    
# project onto the xy plane
mesh.project_points_to_plane(origin=(0,0,0), normal=(0,0,1),inplace=True)

# calculate the relevant positions
if args.direction == 'radial':
    mesh.points[:,0] -= lx/2
    mesh.points[:,1] -= ly/2
    positions = np.linalg.norm(mesh.points,axis=1)
    hist_range = [0,min(lx,ly)/2]

if args.direction == 'x':
    positions = mesh.points[:,0]
    hist_range = [0, lx]

if args.direction == 'y':
    positions = mesh.points[:,1]
    hist_range = [0, ly]

# Take the non-normalized values, and bin them
for dt in lagtimes:
    dsum = mesh.point_arrays[f'dsum={dt}']
    csum = mesh.point_arrays[f'csum={dt}']

    # filter out NaN-s
    mask =np.isnan(dsum)
    dsum = dsum[~mask]
    csum = csum[~mask]
    p = positions[~mask]

    # histogram
    hist, edges = np.histogram(p, bins=args.bins, weights=dsum, range=hist_range) 
    norm, _     = np.histogram(p, bins=args.bins, weights=csum, range=hist_range) 

    values = np.divide(hist,norm)
    centers = (edges[:-1] + edges[1:])/2
    data = np.vstack((centers,values)).T

    np.savetxt(f"{args.name}-{args.direction}-{dt}.dat", data)

