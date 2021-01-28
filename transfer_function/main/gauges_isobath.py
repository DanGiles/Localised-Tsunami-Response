#!/usr/bin/env python
#-*-coding:utf-8-*-
"""
Plotting the forecasted wave heights and calculating the relevant error
along an isobath
"""
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob
from netCDF4 import MFDataset, Dataset
from scipy import stats
import sys
sys.path.append('../src/')
from make_array import *
from matplotlib.ticker import PercentFormatter
def indices(lat,lon,bathy):
    indx = []
    indy = []
    print(len(lon),len(lat),np.shape(bathy))
    for i in range(len(lat)):
        for j in range(len(lon)):
            if -5.1< bathy[i,j] <=-5.0:
                indx.append(i)
                indy.append(j)

    indx = np.array(indx)
    indy = np.array(indy)
    print(len(indx),len(indy))
    return indx,indy


#### Read Bathy and predicted wave heights #####
bathy = ""
local_eta = [""]
lon, lat, Bath, SimRes = make_array(bathy, local_eta, coarse=False)
# ___________________Load in forecasted_____________________________
eta_forecast = np.load("forecasted.npy")
print(np.shape(eta_forecast))
# __________________Isobath Locations_____________________________
xloc, yloc = indices(lat,lon,Bath)
np.save("xlocs.npy",xloc)
np.save("ylocs.npy",yloc)
# _________________________________Plotting_________________________________
stats_sim = SimRes[xloc,yloc,:]
stats_forecast = eta_forecast[xloc,yloc,:]
print(np.shape(stats_sim), np.shape(stats_forecast))
# # Calculate error and std
L1 = np.sum(abs(stats_forecast - stats_sim))/len(stats_forecast.flatten())
std = np.sqrt(np.sum((stats_forecast - stats_sim)**2)/len(stats_forecast.flatten()))
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(stats_sim.flatten()),np.log10(stats_forecast.flatten()))
print(slope, intercept, r_value, p_value, std_err)
y_fit = 10**(slope*np.log10(stats_sim.flatten()) + intercept)
upper = 10**(slope*np.log10(stats_sim.flatten()) + intercept + std)
lower =  10**(slope*np.log10(stats_sim.flatten()) + intercept - std)

print("L1 error = %f"%L1)
print("Std = %f"%std)

textstr = '\n'.join((r'Mean Err. = %.2f' % (L1, ),r'Std dev. = +/- %.2f ' % (std, )))

fig, ax = plt.subplots(figsize=(8,8))
plt.loglog((stats_sim[:,6]),(stats_forecast[:,6]), 'o', fillstyle='none', color ='blue', label = 'S_7.1')
plt.loglog((stats_sim[:,7]),(stats_forecast[:,7]), 'o', fillstyle='none', color ='k', label = 'Y_7.5')
plt.loglog((stats_sim[:,4]),(stats_forecast[:,4]), 'o', fillstyle='none', color ='lawngreen', label = 'J_7.1')
plt.loglog((stats_sim[:,5]),(stats_forecast[:,5]), 'o', fillstyle='none', color ='purple', label = 'J_7.8')
plt.loglog((stats_sim[:,3]),(stats_forecast[:,3]), '+', fillstyle='none', color ='orange', label = '413_6.5')
plt.loglog((stats_sim[:,2]),(stats_forecast[:,2]), '+', fillstyle='none', color ='brown', label = '413_7.0')
plt.loglog((stats_sim[:,1]),(stats_forecast[:,1]), '+', fillstyle='none', color ='pink', label = '413_7.5')
plt.loglog((stats_sim[:,0]),(stats_forecast[:,0]), '+', fillstyle='none', color ='green', label = '413_7.5s')
plt.plot(stats_sim.flatten(), y_fit, 'k')
plt.plot(stats_sim.flatten(), upper, 'grey')
plt.plot(stats_sim.flatten(), lower, 'grey')
plt.xlabel('$\eta_{s}$ (m)', fontsize = 20)
plt.ylabel(r'$\eta_{f} (\beta)$ (m)', fontsize = 20)
ax.text(0.5, 0.1, textstr, verticalalignment='top', fontsize=12)
plt.legend()
plt.savefig('Gauges.jpg', format='jpg',bbox_inches='tight')
plt.show()

#### Percentage Relative difference #####
rel_diff_mean = 100*np.mean((stats_forecast.flatten()-stats_sim.flatten())/stats_sim.flatten())
rel_diff_std = 100*np.std((stats_forecast.flatten()-stats_sim.flatten())/stats_sim.flatten())
print(rel_diff_mean,rel_diff_std)

fig, ax = plt.subplots(figsize=(8,8))
plt.plot((stats_sim[:,6]),100*(stats_forecast[:,6]-stats_sim[:,6])/(stats_sim[:,6]), 'o', fillstyle='none', color ='blue', label = 'S_7.1')
plt.plot((stats_sim[:,7]),100*(stats_forecast[:,7]-stats_sim[:,7])/(stats_sim[:,7]), 'o', fillstyle='none', color ='k', label = 'Y_7.5')
plt.plot((stats_sim[:,4]),100*(stats_forecast[:,4]-stats_sim[:,4])/(stats_sim[:,4]), 'o', fillstyle='none', color ='lawngreen', label = 'J_7.1')
plt.plot((stats_sim[:,5]),100*(stats_forecast[:,5]-stats_sim[:,5])/(stats_sim[:,5]), 'o', fillstyle='none', color ='purple', label = 'J_7.8')
plt.plot((stats_sim[:,3]),100*(stats_forecast[:,3]-stats_sim[:,3])/(stats_sim[:,3]), '+', fillstyle='none', color ='orange', label = '413_6.5')
plt.plot((stats_sim[:,2]),100*(stats_forecast[:,2]-stats_sim[:,2])/(stats_sim[:,2]), '+', fillstyle='none', color ='brown', label = '413_7.0')
plt.plot((stats_sim[:,1]),100*(stats_forecast[:,1]-stats_sim[:,1])/(stats_sim[:,1]), '+', fillstyle='none', color ='pink', label = '413_7.5')
plt.plot((stats_sim[:,0]),100*(stats_forecast[:,0]-stats_sim[:,0])/(stats_sim[:,0]), '+', fillstyle='none', color ='green', label = '413_7.5s')
plt.plot(stats_sim.flatten(),np.ones(len(stats_sim.flatten()))*rel_diff_mean,'k')
plt.plot(stats_sim.flatten(),np.ones(len(stats_sim.flatten()))*rel_diff_mean + rel_diff_std, 'grey')
plt.plot(stats_sim.flatten(),np.ones(len(stats_sim.flatten()))*rel_diff_mean - rel_diff_std, 'grey')
plt.xscale('log')
plt.gca().get_yaxis().set_major_formatter(PercentFormatter())
plt.xlabel(r'$\eta_{s}$ (m)', fontsize = 20)
plt.ylabel(r'Relative Difference', fontsize = 20)
plt.legend()
plt.savefig('Rel_Difference.jpg', format='jpg',bbox_inches='tight')
plt.show()
