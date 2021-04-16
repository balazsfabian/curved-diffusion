import os
import sys
import argparse
import itertools
import multiprocessing as mp

import numpy as np
import pyvista as pv
import MDAnalysis as mda

from geodesic import ExactGeodesicMixin


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


desc = """ Process the pre-calculated distances on the mesh.
"""
parser = argparse.ArgumentParser(description=desc, formatter_class=CustomFormatter)
parser.add_argument('mesh')

args = parser.parse_args()

mesh = pv.read(args.mesh)

print (f"* Using mesh {args.mesh}")
print (" ", mesh)

# Sort the indexfile
fname = mesh.field_arrays['indexfile'][0]
print (f"* Sorting the index file {fname}")

bsort_path = '/home/balazs/Documents/Programs/curved-diffusion/bsort/src/bsort'
os.system(f'{bsort_path} -r 12 -k 4 {fname}')

# Create mapping between lagtimes and array indices
lagtimes = np.array(mesh.field_arrays['lagtimes'])
l_inds = np.array(range(lagtimes.size))
ltab = dict(zip(lagtimes,l_inds))

# ( points, lagtimes )
diffusionMap = np.zeros((mesh.n_points,l_inds.size))
counterMap = np.zeros((mesh.n_points,l_inds.size))

# Any distance larger than this is considered as an error in vtp
vtpError = 10 * max(mesh.bounds)
pts = mesh.points
faces = mesh.faces.reshape((-1,mesh.faces[0]+1))[:,1:]
vtp = ExactGeodesicMixin(pts,faces)

# Load all sorted values in a single go
data = np.fromfile(fname,dtype=np.int32).reshape((-1,3))

# IMPORTANT !!! In this function, we use global scoping
def f(source):

    print ("next...")
    chunk = data.view()[data[:,0]==source]
    # would be better as sparse arrays
    dmap = np.zeros((mesh.n_points,l_inds.size))
    cmap = np.zeros((mesh.n_points,l_inds.size))
    distances = vtp.call_vtp_geodesic(chunk[0,0])

    d = distances[chunk[:,1]]

    # ONLY SELECT THE X DISTANCE
#   d = (mesh.points[chunk[:,0]] - mesh.points[chunk[:,1]])[:,0]

    d2 = np.power(d,2) / 100  # convert to nanometers
    
    for dataLine in zip(chunk,d2):
    
        triplet = dataLine[0]
        i = ltab[triplet[2]]
        value = dataLine[1]
        if value > vtpError:
            continue
        # These could be sparse arrays
        dmap[triplet[0],i] += value
        dmap[triplet[1],i] += value
        cmap[triplet[0],i] += 1
        cmap[triplet[1],i] += 1

    return dmap,cmap
  

sources = list(set(data[:,0]))
print (f"Found {len(sources)} different sources!")
g0 = (n for n in sources)
try:
    pool = mp.Pool(processes=12)              # start 4 worker processes
    N = 120
    while True:
        res = pool.map(f, itertools.islice(g0, N))
        if len(res) == 0:
            break
        res = np.array(res).sum(axis=0)
        diffusionMap += res[0]
        counterMap += res[1]

finally:
    # NOTE: diffusionMap.sum() / counterMap.sum() != np.divide(diffusionMap,counterMap).mean()

    # load raw data into vtk
    for dt in lagtimes:
        mesh.point_arrays["dsum=" + str(dt)] = diffusionMap[:,ltab[dt]]
        mesh.point_arrays["csum=" + str(dt)] = counterMap[:,ltab[dt]]

    counterMap[counterMap == 0] += 1
    diffusionMap = np.divide(diffusionMap, counterMap)
    
    # load maps into vtk
    for dt in lagtimes:
        mesh.point_arrays["lagtime=" + str(dt)] = diffusionMap[:,ltab[dt]]
    
    mesh.save(args.mesh)
