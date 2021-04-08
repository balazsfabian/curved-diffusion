import os
import subprocess
import tempfile
import atexit
from pathlib import Path

from shutil import rmtree
import numpy as np

import uuid



class ExactGeodesicException(Exception):
    """Raised when exact_geodesic_distance() is unavailable or used improperly
    - to create a fallback to geodesic_distance()
    """
    pass


class ExactGeodesicMixin(object):
    """Mixin for computing exact geodesic distance along surface"""

    def __init__(self, pts, polys,dir='.'):
    
        self.pts = pts
        self.polys = polys
        self.dir = Path(dir)
        self.vtp_path = '/home/balazs/Documents/Programs/curved-diffusion/VTP_source_code/vtp'

        if not os.path.exists(self.vtp_path):
            raise ExactGeodesicException('vtp_path does not exist: ' + str(self.vtp_path))

        #create tempdir in current working directory
        self.tmp_dir_path = tempfile.mkdtemp(dir=self.dir)

        # create object file containing the mesh
        self.f_obj, self.tmp_obj_path = tempfile.mkstemp(dir=self.tmp_dir_path)
        self.write_obj(self.tmp_obj_path)

        # register cleaning up
        atexit.register(self.__del__)


    def exact_geodesic_distance(self, vertex):
        """Compute exact geodesic distance along surface
        - uses VTP geodesic algorithm
        Parameters
        ----------
        - vertex : int or list of int
            index of vertex or vertices to compute geodesic distance from
        """
        if isinstance(vertex, list):
            return np.vstack(self.exact_geodesic_distance(v) for v in vertex).min(0)
        else:
            return self.call_vtp_geodesic(vertex)


    def call_vtp_geodesic(self, vertex):
        """Compute geodesic distance using VTP method
        VTP Code
        --------
        - uses external authors' implementation of [Qin el al 2016]
        - https://github.com/YipengQin/VTP_source_code
        - vtp code must be compiled separately to produce VTP executable
        - once compiled, place path to VTP executable in pycortex config
        - i.e. in config put:
            [geodesic]
            vtp_path = /path/to/compiled/VTP
        Parameters
        ----------
        - vertex : int
            index of vertex to compute geodesic distance from
        """

        # initialize temporary result
        self.tmp_output_path = self.tmp_dir_path+'/'+str(uuid.uuid4())

        # run algorithm
        cmd = [self.vtp_path, '-m', self.tmp_obj_path, '-s', str(vertex), '-o', self.tmp_output_path]
        subprocess.call(cmd)

        # read output
        with open(self.tmp_output_path) as f:
            output = f.read()
            distances = np.array(output.split('\n')[:-2], dtype=float)

        if distances.shape[0] == 0:
            raise ExactGeodesicException('VTP error')

#       os.remove(self.tmp_output_path)
        Path(self.tmp_output_path).unlink()

        return distances


    def __del__(self):
        Path(self.tmp_obj_path).unlink()
        rmtree(self.tmp_dir_path)


    def write_obj(self, filename):
        with open(filename, 'w') as fp:
            fp.write("o Object\n")
            for pt in self.pts:
                fp.write("v %0.6f %0.6f %0.6f\n"%tuple(pt))
            fp.write("s off\n")
            for f in self.polys:
                fp.write("f %d %d %d\n"%tuple((f+1)))
