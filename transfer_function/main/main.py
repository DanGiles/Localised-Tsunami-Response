#!/usr/bin/env python
#-*-coding:utf-8-*-
from netCDF4 import Dataset
import numpy as np
import os
import scipy.interpolate as scp
import matplotlib.pyplot as plt
import scipy
from time import time
import sys
sys.path.append('../src/')
## Import own functions
from fine_forecast import *
from coarse_forecast import *
from write2nc import *
from make_array import *
from sort import *
from params import *

# # Load in the global bathymetry and predicted eta_max simulations (.nc)
global_bathy = ""
global_sim = [""]
print("Loading coarse data into arrays")
lon, lat, Bath, SimRes = make_array(global_bathy, global_sim)
# # Load in the local bathymetry and predicted eta_max simulations (.nc)
local_bathy = ""
local_eta = [""]
print("Loading fine data into arrays")
finelon, finelat, FineBath, FineSimRes = make_array(local_bathy, local_eta, coarse=False)
write2nc('Fine_bathy.nc',finelon, finelat, FineBath, 'bathy')

t1 = time()
# Compute list of indices corresponding to the positions in the coarse grid of each point ranked by bathymetry value
idx_coarse = sort(Bath)
# Compute coarse forecast over the coarse grid
coarse_forecast = coarse_forecast(SimRes, Bath, idx_coarse, coarse_lowlim, coarse_highlim)
#compute list of indices corresponding to the positions in the grid of each point ranked by bathymetry value
idx_fine = sort(FineBath)
# Compute the fine forecast and optimise for the amplification parameter (beta)
betas, fine_forecast = beta_calculate(coarse_forecast, Bath, FineSimRes, FineBath, lat, lon, finelat, finelon, idx_fine,lowlim,highlim, hE)
t2 = time()
print('Runtime for coarse forecast, fine forecast and beta optimisation = %f \n'%(t2-t1))

# Write beta output to a netCDF files
write2nc('Betas.nc',finelon, finelat, betas,'beta')
# Output of the forecasted maximum wave heights .npy file
np.save('Forecasted.npy',fine_forecast)
