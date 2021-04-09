#!/home/balazs/.conda/envs/mdanalysis/bin/python
import sys
from itertools import cycle
import pyvista as pv

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


fn = sys.argv[1]
prop = sys.argv[2]

mesh = pv.read(fn)

if prop == 'obj':
    pv.save_meshio(fn[:-3]+'obj',mesh)
    exit()

if prop == 'xyz':
    write_xyz(open(fn[:-3]+'xyz','w'), mesh.points, title='meshed surface')
    exit()

try:
    data = mesh.cell_arrays[prop]
except KeyError:
    pass

try:
    data = mesh.point_arrays[prop]
except KeyError:
    pass

try:
    data = mesh.field_arrays[prop]
except KeyError:
    pass

for elem in data:
    print (elem)
