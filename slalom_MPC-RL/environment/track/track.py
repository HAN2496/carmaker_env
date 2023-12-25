from os.path import join
from pathlib import Path
from numpy import array, loadtxt, argmin, sin, cos, arctan2
from numpy.linalg import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from numpy import genfromtxt
from scipy.integrate import quad
import numpy as np

class Track:

    def __init__(self, filename="trackdata.npz"):
        track_file = join(str(Path(__file__).parent), "data/"+filename)
        trackdata = np.load(track_file)

        # X: x position of centertline. shape:(81,) 
        # border_left: y position of left(top) border. shape:(81,)
        # border_right: y position of right(bottom) border. shape:(81,)
        # cone_position : x position of cones. shape:(10,)

        self.cone_position = trackdata["cone_position"]
        self.X = trackdata["X"]
        self.border_left = trackdata["border_left"]
        self.border_right = trackdata["border_right"]


    def get_nearlest_cone(self, X, return_cone_index=False):
        idx = np.argmin((self.cone_position - X)**2)
        cone_position = np.array([self.cone_position[idx], 0.0])
        if return_cone_index:
            return cone_position, idx
        else:
            return cone_position
