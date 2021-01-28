#!/usr/bin/env python
#-*-coding:utf-8-*-
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


def plot_beta(Lon, Lat,beta,FineBath,i,ax,cmaps,arr):
	topomap = 'Greys_r'
	topolevels = np.arange(0,np.amax(FineBath)/3,2)
	# Wave height colorbar
	colormap = cmaps
	beta = ax[i].contourf(Lon, Lat, beta,levels=arr, cmap = colormap,extend='both')
	topo = ax[i].contourf(Lon, Lat, FineBath,topolevels,cmap = topomap,extend='max')
	cs = ax[i].contour(Lon, Lat, FineBath,[-1])
	if i == 1:
		ax[i].set_xlabel('Lon ($^\circ$)', fontsize =15)
		p0 = ax[1].get_position().get_points().flatten()
		cbar_ax = fig.add_axes([p0[0], 0.03, p0[2]-p0[0], 0.01])
		cbar = plt.colorbar(beta, orientation='horizontal',extend='both', ticks = [-1.0,-0.5,0,0.5,1.0,1.5,2.0,2.5,3.0],cax=cbar_ax)
		cbar.ax.set_xlabel(r'$\beta$', fontsize =20)
		cbar.ax.get_yaxis().labelpad = 50
		cbar.ax.tick_params(labelsize=20)

	plt.show(block=False)
	return

def plot_bathy(Lon, Lat,FineBath,i,ax,cmaps,array):
	topomap = 'Greys_r'
	topolevels = np.arange(0,np.amax(FineBath)/3,2)
	newcmap = cmo.tools.crop(cmaps, np.amin(FineBath), 0, 0)
	# # Wave height colorbar
	error = ax[i].contourf(Lon, Lat, FineBath,levels=array, cmap = newcmap)
	topo = ax[i].contourf(Lon, Lat, FineBath,topolevels,cmap = topomap, extend='max')
	ax[i].contour(Lon, Lat, FineBath,[-1])
	xind = np.load("xlocs.npy")
	yind = np.load("ylocs.npy")
	ax[i].plot(Lon[xind,yind], Lat[xind,yind], '*', color='r')
	dx = 250
	if i == 0:
		ax[i].set_ylabel('Lat ($^\circ$)', fontsize =15)
		ax[i].set_xlabel('Lon ($^\circ$)', fontsize =15)
		p0 = ax[0].get_position().get_points().flatten()
		cbar_ax = fig.add_axes([p0[0], 0.03, p0[2]-p0[0], 0.01])
		cbar = plt.colorbar(error, orientation='horizontal',extend='neither',ticks = [-1000,-1000+dx,-1000+2*dx,-1000+3*dx,0],cax=cbar_ax)
		cbar.ax.set_xlabel('Bathymetry (m)', fontsize =20)
		cbar.ax.get_yaxis().labelpad = 50
		cbar.ax.tick_params(labelsize=20)

	plt.show(block=False)
	return



fig, ax = plt.subplots(1,2,figsize=(21,9),sharex=True, sharey=True,gridspec_kw={'wspace': 0.1,'hspace': 0.1})
ax = ax.ravel()
data = Dataset('Fine_bathy.nc')
print(data.variables.keys())
x = data.variables["lon"]
y = data.variables["lat"]
bathy = data.variables["bathy"]
x = np.array(x)
y = np.array(y)
bathy = np.array(bathy)
print(np.amax(bathy),np.amin(bathy))
X,Y = np.meshgrid(x,y)
colormap = cmo.topo
array = np.arange(np.amin(bathy),11,10)
plot_bathy(X,Y,bathy,0,ax,colormap,array)
################################################################################
data = Dataset('Betas.nc')
print(data.variables.keys())
res = data.variables["beta"]
res = np.array(res)
colormap = cmo.balance
print(np.amax(res),np.amin(res))
## Masking Array
arg = np.argwhere((bathy<-50) | (bathy >-1))
print(arg[:,0])
res[arg[:,0],arg[:,1]] = np.nan
array = np.arange(-1.0,3.1,0.25)
plot_beta(X,Y,res,bathy,1,ax,colormap,array)
################################################################################

plt.savefig('Bathy_beta.jpg', format='jpg',bbox_inches='tight')
plt.show()
