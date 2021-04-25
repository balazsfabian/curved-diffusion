#!/home/balazs/.conda/envs/mdanalysis/bin/python
import sys
from itertools import cycle

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors


def write_xyz(fout, coords, title="", atomtypes=("A",)):
    """ write a xyz file from file handle
    Writes coordinates in xyz format. It uses atomtypes as names. The list is
    cycled if it contains less entries than there are coordinates,
    One can also directly write xyz data which was generated with read_xyz.
    >>> xx = read_xyz("in.xyz")
    >>> write_xyz(open("out.xyz", "w"), *xx)
    Parameters
    ----------
    fout : an open file
    coords : np.array
        array of coordinates
    title : title section, optional
        title for xyz file
    atomtypes : iteratable
        list of atomtypes.
    See Also
    --------
    read_xyz
    """
    fout.write("%d\n%s\n" % (coords.size / 3, title))
    for x, atomtype in zip(coords.reshape(-1, 3), cycle(atomtypes)):
        fout.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))


def wrap_msd_arrays(mesh):
    """ wrap point-array properties into the unit cell of the mesh
    """
    lagtimes = mesh.field_arrays['lagtimes']
    lx = mesh.field_arrays['lx'][0]
    ly = mesh.field_arrays['ly'][0]
    
    #Set up NBsearch
    pts = mesh.points
    neigh = NearestNeighbors(n_neighbors=1,n_jobs=-1)
    neigh.fit(pts)
    
    #Wrap points into [0,lx] x [0,ly] x [.,.]
    periodic_pts = np.mod(pts,[lx,ly,np.inf])
    # negative points misbehave when mod is calculated
    periodic_pts[pts <0] = pts[pts <0]
    
    mapping = neigh.kneighbors(periodic_pts)[1].flatten()

    for dt in lagtimes:
        dsum = mesh.point_arrays[f'dsum={dt}']
        csum = mesh.point_arrays[f'csum={dt}']
    
        for i,ndx in enumerate(mapping):
            if i == ndx:
                continue
            # add to central image...
            dsum[ndx] += dsum[i]
            csum[ndx] += csum[i]
            # ... and clear the non-central
            dsum[i] = np.nan
            csum[i] = np.nan
    
        csum[csum==0] = np.nan
    
        mesh.point_arrays[f'dsum={dt}'] = dsum
        mesh.point_arrays[f'csum={dt}'] = csum
    
        mesh.point_arrays[f"lagtime={dt}"] = np.divide(dsum,csum)


fn = sys.argv[1]
prop = sys.argv[2]

mesh = pv.read(fn)


# Wrap properties into the unit cell
if prop == 'wrap-msd':
    wrap_msd_arrays(mesh)
    mesh.save(fn)
    exit()


# Write an obj file
if prop == 'obj':
    pv.save_meshio(fn[:-3]+'obj',mesh)
    exit()


# Write an xyz file
if prop == 'xyz':
    write_xyz(open(fn[:-3]+'xyz','w'), mesh.points, title='meshed surface')
    exit()


# Print average MSD curve
if prop == 'msd':
    try:
        lagtimes = mesh.field_arrays['lagtimes']
        for dt in lagtimes:
            dsum = mesh.point_arrays[f'dsum={dt}']
            csum = mesh.point_arrays[f'csum={dt}']
            print (dt, np.nansum(dsum)/np.nansum(csum))
    except KeyError:
        print ("This mesh does not contain MSD data!")
        exit(1)


# Access generic data from any source
data = mesh[prop]
if data is None:
    print (f"No attribute named \'{prop}\' in mesh!")
    exit(1)
else:
    for elem in data:
        print (elem)
