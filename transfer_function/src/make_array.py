import numpy as np
from netCDF4 import Dataset

def make_array(place, tsu_list, coarse=False):
    N = len(tsu_list)
    bathy = Dataset(place, "r")
    var = list(bathy.variables.keys())
    if coarse:
        x = bathy.variables[var[2]]
        x = np.asarray(x)
        y = bathy.variables[var[1]]
        y = np.asarray(y)
        z = bathy.variables[var[0]]
    else:
        x = bathy.variables[var[0]]
        x = np.asarray(x)
        y = bathy.variables[var[1]]
        y = np.asarray(y)
        z = bathy.variables[var[2]]

    Bath = np.asarray(z)
    (east, north) = np.shape(Bath)
    SimRes = np.zeros((east, north, N))
    for i in range(N):
       tsu = Dataset(tsu_list[i], "r")
       var = list(tsu.variables.keys())
       if coarse:
          sim = tsu.variables[var[0]]
       else:
          sim = tsu.variables[var[2]]
          lon = tsu.variables[var[0]]
          lat = tsu.variables[var[1]]
       SimRes[:, :, i] = np.asarray(sim)
    return x,  y,  Bath,  SimRes
