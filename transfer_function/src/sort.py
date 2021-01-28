import numpy as np

def sort(bathy):
    i = (-bathy).argsort(axis=None)[::-1]
    j = np.unravel_index(i, bathy.shape)
    j = np.array(j)
    return j
