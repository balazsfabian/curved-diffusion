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
parser.add_argument('--property',  default='msd',  type=str,                   help="Property to be processed")

args = parser.parse_args()


mesh = pv.read(args.mesh)

lagtimes = mesh.field_arrays['lagtimes']
lx = mesh.field_arrays['lx'][0]
ly = mesh.field_arrays['ly'][0]
    
# project onto the xy plane
mesh.project_points_to_plane(origin=(0,0,0), normal=(0,0,1),inplace=True)

# Calculate MSD for all lagtimes ------------------------------
if args.property == 'msd':
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
        mask = np.isnan(dsum)
        dsum = dsum[~mask]
        csum = csum[~mask]
        p = positions[~mask]

        # histogram
        hist, edges = np.histogram(p, bins=args.bins, weights=dsum, range=hist_range)
        norm, _     = np.histogram(p, bins=args.bins, weights=csum, range=hist_range)

        values = np.divide(hist,norm)
        centers = (edges[:-1] + edges[1:])/2
        data = np.vstack((centers,values)).T

        np.savetxt(f"{args.name}-{args.property}-{args.direction}-{dt}.dat", data)


# Calculate Mean or Gaussian curvature for all lagtimes ----------
elif args.property in ['Mean','Gaussian']:
    # calculate the relevant positions
    # * use centers of the faces
    points = mesh.cell_centers().points
    if args.direction == 'radial':
        points[:,0] -= lx/2
        points[:,1] -= ly/2
        positions = np.linalg.norm(points,axis=1)
        hist_range = [0,min(lx,ly)/2]

    if args.direction == 'x':
        positions = points[:,0]
        hist_range = [0, lx]

    if args.direction == 'y':
        positions = points[:,1]
        hist_range = [0, ly]

    # covert point property to cell property
    prop = mesh.ptc().cell_arrays[args.property]
    area = mesh.compute_cell_sizes(length=False,area=True,volume=False).cell_arrays['Area']
    prop = np.multiply(prop,area)

    # area-weighted curvature values
    hist, edges = np.histogram(positions, bins=args.bins, weights=prop, range=hist_range)
    norm, _     = np.histogram(positions, bins=args.bins, weights=area, range=hist_range)

    values = np.divide(hist,norm)
    centers = (edges[:-1] + edges[1:])/2
    data = np.vstack((centers,values)).T

    np.savetxt(f"{args.name}-{args.property}-{args.direction}.dat", data)

