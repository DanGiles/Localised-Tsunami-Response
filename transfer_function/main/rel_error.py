#!/usr/bin/env python
#-*-coding:utf-8-*-
"""
Plotting the forecasted wave heights and calculating the relevant error

"""
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob
import sys
from netCDF4 import MFDataset, Dataset
from scipy import stats
import cmocean
import cmocean.cm as cmo

sys.path.append('../src/')
## Import own functions
from make_array import *

def plot_error(Lon, Lat,error,FineBath,ax):
	topomap = cmo.topo
	topomap.set_under('white')
	topolevels = np.arange(0,500,50)
	newcmap = cmo.tools.crop(topomap, 0, 500, 0)
	# Wave height colorbar
	colormap = cmo.solar
	error = ax[i].contourf(Lon, Lat, error,levels=np.arange(0,100.1,10.0), cmap = colormap)
	topo = ax[i].contourf(Lon, Lat, FineBath,topolevels,cmap = newcmap)
	# ax[i].tick_params(labelsize=15)
	#
	# if num ==1:
	# 	ax[i].set_title('Warning $\mathcal{W}^{1}$ - $\sigma$', fontsize =25)
	# elif num ==2:
	# 	ax[i].set_title('Warning $\mathcal{W}^{2}$ - $\sigma$', fontsize =25)
	# else:
	# if i == 8:
	p0 = ax[0].get_position().get_points().flatten()
	p1 = ax[0].get_position().get_points().flatten()
	cbar_ax = fig.add_axes([p0[0], 0.075, p1[2]-p0[0], 0.01])
	cbar = plt.colorbar(error, ticks=np.arange(0,100.1,10.0), orientation='horizontal',extend='neither', cax=cbar_ax)
	cbar.ax.set_xlabel('Relative error %', fontsize =20)
	cbar.ax.get_yaxis().labelpad = 50
	cbar.ax.tick_params(labelsize=20)
	# 	ax[i].set_title('Warning $\mathcal{W}^{3}$ - $\sigma$', fontsize =25)
	# 	p0 = ax[i].get_position().get_points().flatten()
	# 	cbar_ax = fig.add_axes([(p0[0]+0.01), 0.07, (p0[2]-0.01)-(p0[0]+0.01), 0.01])
	# 	# cbar = plt.colorbar(eta,ticks=np.arange(0,0.121,0.02), orientation='horizontal',extend='neither', cax=cbar_ax)
	# 	cbar = plt.colorbar(eta,ticks=np.arange(0,0.076,0.025), orientation='horizontal',extend='neither', cax=cbar_ax)
	# 	# cbar = plt.colorbar(eta,ticks=np.arange(0,0.61,0.1), orientation='horizontal',extend='neither', cax=cbar_ax)
	# 	cbar.ax.set_xlabel('Standard Deviation $\sigma$ (m)', fontsize =30)
	# 	cbar.ax.get_yaxis().labelpad = 50
	# 	cbar.ax.tick_params(labelsize=20)
	# 	ax[i].set_xlabel('X [m]', fontsize = 20)
	plt.show(block=False)
	return

def plot_sim(Lon, Lat,error,FineBath,i,ax):
	topomap = cmo.topo
	topomap.set_under('white')
	topolevels = np.arange(0,300,20)
	newcmap = cmo.tools.crop(topomap, 0, 300, 0)
	# Wave height colorbar
	colormap = cmo.amp
	print(i)
	error = ax[i].contourf(Lon, Lat, error,levels=np.arange(0,3.1,0.5))#, cmap = colormap)
	topo = ax[i].contourf(Lon, Lat, FineBath,topolevels,cmap = newcmap)
	# ax[i].tick_params(labelsize=15)
	#
	# if num ==1:
	# 	ax[i].set_title('Warning $\mathcal{W}^{1}$ - $\sigma$', fontsize =25)
	# elif num ==2:
	# 	ax[i].set_title('Warning $\mathcal{W}^{2}$ - $\sigma$', fontsize =25)
	# else:
	if i == 8:
		p0 = ax[0].get_position().get_points().flatten()
		p1 = ax[8].get_position().get_points().flatten()
		cbar_ax = fig.add_axes([p0[0], 0.075, p1[2]-p0[0], 0.01])
		cbar = plt.colorbar(error, ticks=np.arange(0,3.1,0.5), orientation='horizontal',extend='neither', cax=cbar_ax)
		cbar.ax.set_xlabel('Maximum Wave Height (m)', fontsize =20)
		cbar.ax.get_yaxis().labelpad = 50
		cbar.ax.tick_params(labelsize=20)
	plt.show(block=False)
	return



#### Read in Bathy Grid and Simulation Results ######
local_bathy = "../../bathy/bathy10m/niceArea-cut_10m.grd"
local_eta = ["../../Resultat_Calc_multigrd/Casst413_Mw7.5s/hmax413-75s.gr03.3h_NiceArea.grd", \
"../../Resultat_Calc_multigrd/Casst413_Mw7.5/hmax413-75.gr03.3h_NiceArea.grd",\
"../../Resultat_Calc_multigrd/Casst413_Mw7.0/hmax413-70.gr03.3h_NiceArea.grd", \
"../../Resultat_Calc_multigrd/Casst413_Mw6.5/hmax413-65.gr03.3h_NiceArea.grd", \
"../../Resultat_Calc_multigrd/Jijel7.1/hmaxJ7.1.gr03.3h_NiceArea.grd", \
"../../Resultat_Calc_multigrd/Jijel7.8/hmaxJ7.8.gr03.3h_NiceArea.grd", \
# "../../Resultat_Calc_multigrd/Ligure1887/hmaxL.gr03.1h_NiceArea.grd", \
"../../Resultat_Calc_multigrd/Boum2003_SemmaneMw7.1/hmaxS.gr03.3h_NiceArea.grd", \
"../../Resultat_Calc_multigrd/Boum2003_YellesMw7.5/hmaxY.gr03.3h_NiceArea.grd"]

print("Loading fine data into arrays")
finelon, finelat, FineBath, FineSimRes = make_array(local_bathy, local_eta, coarse=False)

print(np.shape(FineSimRes))
N = len(local_eta)
(east, north) = np.shape(FineBath)

forecast = np.zeros((east, north, N))
print(np.shape(forecast))
#### Read in Forecasted Results #######
forecast = np.load("../../Results/Nice/Forecast_no_coarse.npy")

Lon, Lat = np.meshgrid(finelon, finelat)
# # #### Calculate error and std #######
error = np.zeros((east, north, N))
print(np.shape(error))
for i in range(N):
	for x in range(len(error[:,0,0])):
		for y in range(len(error[0,:,0])):
			if (FineBath[x,y] < 0.0) and (FineSimRes[x,y,i] > 0.01):
				error[x,y,i] = abs(forecast[x,y,i] - FineSimRes[x,y,i])/FineSimRes[x,y,i]

mean_error = np.mean(error, axis = 2)
# np.save('../../Results/Nice/mean_rel_error_full.npy',mean_error)
# mean_error = np.load('../../Results/Nice/mean_rel_error_full.npy')
print(np.shape(mean_error))
mean_error = mean_error * 100.0

# xloc = np.load('../../Results/Nice/xloc.npy')
# yloc = np.load('../../Results/Nice/yloc.npy')
print(np.max(mean_error))
#### Plotting Error and Std #######
fig, ax = plt.subplots(figsize=(8,8))
topomap = 'Greys_r'
topolevels = np.arange(0,np.amax(FineBath),10)
# topomap.set_under('white')
# topolevels = np.arange(0,500,50)
# newcmap = cmo.tools.crop(topomap, 0, 500, 0)
# Wave height colorbar
colormap = cmo.turbid_r
colormap = 'Blues_r'
error = ax.contourf(Lon, Lat, mean_error,levels=np.arange(0,100.1,10.0), cmap = colormap)
topo = ax.contourf(Lon, Lat, FineBath,topolevels,cmap = topomap)
ax.contour(Lon, Lat, FineBath,[-5], color = 'red')
# plt.plot(xloc,yloc,'o', color='r', fillstyle = 'none')
# ax.contour(Lon, Lat, FineBath,[-50], color = 'red')

p0 = ax.get_position().get_points().flatten()
cbar_ax = fig.add_axes([p0[0], 0.05, p0[2]-p0[0], 0.01])
cbar = plt.colorbar(error, ticks=np.arange(0,100.1,10.0), orientation='horizontal',extend='neither', cax=cbar_ax)
cbar.ax.set_xlabel('Relative error %', fontsize =20)
cbar.ax.get_yaxis().labelpad = 50
cbar.ax.tick_params(labelsize=20)
plt.savefig('../../Results/Nice/Mean_rel_error_no_coarse.png', format='png',bbox_inches='tight')
plt.show()
