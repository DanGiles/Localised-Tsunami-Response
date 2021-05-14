#-*-coding:utf-8-*-
from netCDF4 import Dataset
import numpy as np
import os
import scipy.interpolate as scp
import matplotlib.pyplot as plt
import scipy
from time import time
from ctypes import *
from numpy.ctypeslib import ndpointer
from params import *
from write2nc import *

#
def interpolate(lat,lon,Z,B,Bath_fine):
	'''
	Interpolates coarse grid onto fine grid at one point by selecting nearest deeper points in the Coarse grid,
	setting up a mesh on these points and then interpolating on the deepest point in the fine grid.
	lat , lon: Coordinates of the points in the coarse grid
	lat_fine, lon_fine : Coordinates of the fine grid
	Z : Gridded eta values from the coarse grid at the selected (lat, lon)
	B : Gridded bathymetry values from the coarse grid at the selected (lat, lon)
	imin , jmin : Index of the deepest point in the fine grid (imin , jmin )

	'''
	Z = np.nan_to_num(Z)
	# Intepolate eta and bathy on fine grid
	interp_Z = scp.RectBivariateSpline(lat,lon, Z,kx=3,ky=3)
	interp_B = scp.RectBivariateSpline(lat,lon, B,kx=3,ky=3)
	# lat_int = np.linspace(lat_fine[imin]-2,lat_fine[imin]+2,100)
	# lon_int = np.linspace(lon_fine[jmin]-2,lon_fine[jmin]+2,100)
	lat_int = np.linspace(lat[0],lat[-1],100)
	lon_int = np.linspace(lon[0],lon[-1],100)
	# X2, Y2 = np.meshgrid(lat_fine,lon_fine)
	# Z2 = interp_Z(lat_fine,lon_fine)
	# B2 = interp_B(lat_fine,lon_fine)
	Z2 = interp_Z.__call__(lat_int,lon_int)
	B2 = interp_B.__call__(lat_int,lon_int)


	# find closest point in the real fine grid
	# umin = np.argmin(abs(lat-lat_fine[imin]))
	# vmin = np.argmin(abs(lon-lon_fine[jmin]))
	# print(umin,vmin,Z2[umin,vmin], B2[umin,vmin],B2, np.shape(B2), np.amax(B2), np.amin(B2), lat_fine )
	# print(B, np.shape(B))#,B[imin,jmin])
	# # find bathy point
	temp_array = abs(Bath_fine-B2)
	umin,vmin = np.unravel_index(np.argmin(temp_array, axis=None), temp_array.shape)
	# umin,vmin = np.argmin(abs(Bath_fine[imin,jmin]-B2))
	# print(umin,vmin,Z2[umin,vmin], B2[umin,vmin])
	return Z2[umin,vmin], B2[umin,vmin]


def coarse_to_fine(SimRes, Bath, FineBath, Lat, Lon, FineLat, FineLon, alpha, imin,jmin):
	'''
	Interpolates the maximum wave heights from the coarse grid to the fine grid.
	'''
	(east, north)=np.shape(Bath)
	# (fine_east, fine_north)=np.shape(FineBath)
	N = len(SimRes[0,0,:])
	# Fine_computed_heights2=np.zeros((fine_east,fine_north,N))
	'''
	get closest big cell to small cell (imin, jmin)
	'''
	u,v= np.argmin(np.abs(Lat-FineLat)),  np.argmin(np.abs(Lon-FineLon))

	num = 0
	radius = 0
	while Bath[u,v] > FineBath:
	   # print(u,v,Bath[u,v], FineBath[imin,jmin])
	   ind = np.unravel_index(np.argmin(Bath[u-1-radius:u+2+radius,v-1-radius:v+2+radius], axis=None), Bath.shape)
	   ind_val = ind[1]
	   count = 0
	   for i in range(len(Lat[u-1-radius:u+2+radius])):
		   for j in range(len(Lon[v-1-radius:v+2+radius])):
			   if (count) == ind_val:
				   indx = i - 1 - radius
				   indy = j - 1 - radius
			   count += 1
	   u = u + indx
	   v = v + indy
	   if (indx == 0) and (indy ==0):
		   radius += 1
	radius = max(radius,3)
	# print(Bath[u,v],radius)

	Lon_grid=Lon[v-1-radius:v+2+radius]
	Lat_grid=Lat[u-1-radius:u+2+radius]

	Z = np.zeros(N)
	Fine_computed_heights = np.zeros(N)
	for n in range(N):
		Z_grid = SimRes[u-1-radius:u+2+radius,v-1-radius:v+2+radius,n]
		B_grid = Bath[u-1-radius:u+2+radius,v-1-radius:v+2+radius]
		Z[n],B=interpolate(Lat_grid,Lon_grid,Z_grid, B_grid,FineBath)

	Fine_computed_heights = Z*alpha[imin,jmin]*(B/FineBath)**(1/4)
	return Fine_computed_heights


def forecasting(SimRes, Bath, FineBath, alpha, Lat, Lon, FineLat, FineLon, idx_list, idx_neighbour, lowlim, highlim):
	'''
	Use precomputed beta values to forecast for unseen event.

	SimRes , FineSimRes : simulation results for coarse and fine grids
	Bath , FineBath : bathymetry tables for coarse and fine grids
	Lat , Lon , FineLat , FineLon : latitude and longitude tables
	idx_list : list of of indices corresponding to the positions in the grid of each
	point ranked by bathymetry value from idx_table ( FineBath )
	lowlim , highlim : range for Extended Green ’s Law
	hE : typical value
	'''
	(east, north) = np.shape(Bath)
	(fine_east, fine_north) = np.shape(FineBath)
	N = len(SimRes[0,0,:])
	Fine_computed_heights = np.zeros((fine_east,fine_north,N))
	x,y = int(idx_list[0,0]),int(idx_list[1,0])
	# Interpolate wave heights from the coarse grid to the fine one.
	print("Interpolation from coarse to fine")
	Fine_computed_heights[x,y,:]= coarse_to_fine(SimRes, Bath, FineBath[x,y], Lat, Lon, FineLat[x], FineLon[y], alpha, x, y)
	print('Interpolation to fine grid and simulation')
	print(Fine_computed_heights[x,y,:], alpha[x,y])
	print('\n')
	maxint = int(len(alpha.flatten())/10)
	print("Fine Forecast")

	for i in range(1, fine_east*fine_north):
		'''
		Iterate through the fine grid by increasing bathymetry values
		'''
		# x,y = u,v
		x,y = int(idx_neighbour[0,i]),int(idx_neighbour[1,i])
		u,v= int(idx_list[0,i]),int(idx_list[1,i])

		if FineBath [u,v] < highlim:
			'''
			Use extended Green ’s law
			'''
			Fine_computed_heights[u,v,:]= Fine_computed_heights[x,y,:]*alpha[u,v]*(FineBath[x,y]/FineBath[u,v])**(1/4)

		elif highlim <= FineBath[u,v] < 0.0:
			'''
			Extrapolate to the shoreline
			'''
			Fine_computed_heights[u,v,:]= Fine_computed_heights[x,y,:]

		elif FineBath[u,v] >= 0.0:
			break

		if np.mod(i,maxint)==0:
			print(int(i/len(alpha.flatten())*100),"% complete" )
			print("beta = ", alpha[u,v])
			print("Depth = ", FineBath[u,v])
			print("Forecasted Wave Height = ", Fine_computed_heights[u,v,:])
			print("\n")


	return Fine_computed_heights
