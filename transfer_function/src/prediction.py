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
def interpolate(lat,lon,lat_fine,lon_fine,Z,B,Bath_fine,imin,jmin):
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
	temp_array = abs(Bath_fine[imin,jmin]-B2)
	umin,vmin = np.unravel_index(np.argmin(temp_array, axis=None), temp_array.shape)
	# umin,vmin = np.argmin(abs(Bath_fine[imin,jmin]-B2))
	# print(umin,vmin,Z2[umin,vmin], B2[umin,vmin])
	return Z2[umin,vmin], B2[umin,vmin]

def coarse_to_fine(SimRes, Bath, FineBath, Lat, Lon, FineLat, FineLon, lowlim, highlim, hE):
	'''
	Interpolates the maximum wave heights from the coarse grid to the fine grid.
	'''
	(east, north)=np.shape(Bath)
	(fine_east, fine_north)=np.shape(FineBath)
	N = len(SimRes[0,0,:])
	Fine_computed_heights2=np.zeros((fine_east,fine_north,N))

	'''
	Calculates the height at the minimum depth in the fine grid
	Interpolation with adjacent coarse grid cells
	'''
	imin,jmin=np.where(FineBath==np.amin(FineBath))
	imin,jmin=imin[0],jmin[0]
	'''
	get closest big cell to small cell (imin, jmin)
	'''
	u,v= np.argmin(np.abs(Lat-FineLat[imin])),  np.argmin(np.abs(Lon-FineLon[jmin]))
	print(u,v, Bath[u,v],SimRes[u,v,0])

	print("lowlim_fine = ", lowlim)
	print("highlim_fine = ", highlim)
	print("hE_fine = ", hE)
	# Small coarse latitude and longitude tables over deepest point
	num = 0
	radius = 0
	while Bath[u,v] > FineBath[imin,jmin]:
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
	# print(Lat_grid, FineLat[imin])
	# print(Lon_grid, FineLon[jmin])
	for n in range(N):
		Z_grid = SimRes[u-1-radius:u+2+radius,v-1-radius:v+2+radius,n]
		B_grid = Bath[u-1-radius:u+2+radius,v-1-radius:v+2+radius]
		Z,B=interpolate(Lat_grid,Lon_grid, FineLat,FineLon,Z_grid, B_grid,FineBath,imin,jmin)
		Fine_computed_heights2[imin, jmin, n] = Z*(B/FineBath[imin,jmin])**(1/4)
	print('Interpolated Wave Height')
	print(Fine_computed_heights2[imin,jmin,:])
	print('\n')

	return Fine_computed_heights2

def neighbour_check(neighx,neighy,FineBath,xind,yind,min_x,max_x,min_y,max_y,count):
	tempBath = FineBath[neighx,neighy]
	check = False
	num = 0
	radius = 1
	while check == False and num <= count:
		for k in xind:
			for j in yind:
				if (min_x < k < max_x) and (min_y < j < max_y):
					if (FineBath[k,j] < tempBath):# and all(i > 0 for i in Fine_computed_heights[k,j,:])):
						neighx = k
						neighy = j
						# tempBath = FineBath[neighx,neighy]
						check = True
					else:
						radius += 1
						count += 1
		xind = np.arange(neighx-radius, neighx+radius+1, 1)
		yind = np.arange(neighy-radius, neighy+radius+1, 1)
		# print(xind,yind)
	# if all(i > 0 for i in Fine_computed_heights[neighx,neighy,:]):
	return neighx,neighy,check

def check(finelat, finelon, fine_forecast, FineBath):
    for i in range(len(finelat)):
	    for j in range(len(finelon)):
	       for n in range(len(fine_forecast[i,j,:])):
	       	   if fine_forecast[i,j,n] == 0.0 and FineBath[i,j] < 0.0:
	               fine_forecast[i,j,n] = fine_forecast[i-1,j-1,n]

    return fine_forecast



def forecasting(SimRes, Bath, FineBath, Betas, Lat, Lon, FineLat, FineLon, idx_list, lowlim, highlim, hE):
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
	(east, north)=np.shape(Bath)
	(fine_east, fine_north)=np.shape(FineBath)
	N = len(SimRes[0,0,:])
	# Interpolate wave heights from the coarse grid to the fine one.
	print("Interpolation from coarse to fine")
	Fine_computed_heights= coarse_to_fine(SimRes, Bath, FineBath, Lat, Lon, FineLat, FineLon, lowlim, highlim, hE)
	x,y = int(idx_list[0,0]),int(idx_list[1,0])
	i = 0
	maxint = int(len(Betas.flatten())/10)
	### Domain max/min #####
	max_x = np.max(idx_list[0,:])
	min_x = np.min(idx_list[0,:])
	max_y = np.max(idx_list[1,:])
	min_y = np.min(idx_list[1,:])

	print("Fine Forecast")

	print("lowlim_fine = ", lowlim)
	print("highlim_fine = ", highlim)
	print("hE_fine = ", hE)
	# LON,LAT= np.meshgrid(FineLon, FineLat)
	for i in range(1, fine_east*fine_north):
		'''
		Iterate through the fine grid by increasing bathymetry values
		'''
		u,v= int(idx_list[0,i]),int(idx_list[1,i])
		### Neighbour Region #####
		xind = [u-1,u,u+1]
		yind = [v-1,v,v+1]

		if FineBath[u,v] < lowlim:
			'''
			Use Geen ’s law
			'''
			for n in range (N):
				Fine_computed_heights[u,v,n]= Fine_computed_heights[x,y,n]*(FineBath[x,y]/ FineBath[u,v]) **(1/4)

		elif lowlim <= FineBath [u,v] < highlim:
			'''
			Use extended Green ’s law ( calculate beta using a gradient descent )
			'''
			for n in range (N):
				Fine_computed_heights[u,v,n]= Fine_computed_heights[x,y,n]*(1+ Betas[u,v]*((hE - FineBath[u,v])/hE))*(FineBath [x,y]/FineBath [u,v]) **(1/4)

		elif highlim <= FineBath[u,v] < 0.0:
			'''
			Extrapolate to the shoreline
			'''
			x,y = int(idx_list[0,i-1]),int(idx_list[1,i-1])
			eta1 = np.zeros(N)
			eta2 = np.zeros(N)
			eta3 = np.zeros(N)
			eta4 = np.zeros(N)
			num = 0
			try:
				if FineBath[u-1,v] < FineBath[u,v] and np.all(np.isnan(Fine_computed_heights[u-1,v,:])!= True):
					eta1[:] = Fine_computed_heights[u-1,v,:]#*(Bath[u-1,v]/Bath[u,v])**(1/4)
					num += 1
				if FineBath[u+1,v] < FineBath[u,v] and np.all(np.isnan(Fine_computed_heights[u+1,v,:])!= True):
					eta2[:] = Fine_computed_heights[u+1,v,:]#*(Bath[u+1,v]/Bath[u,v])**(1/4)
					num += 1
				if FineBath[u,v-1] < FineBath[u,v] and np.all(np.isnan(Fine_computed_heights[u,v-1,:])!= True):
					eta3[:] = Fine_computed_heights[u,v-1,:]#*(Bath[u,v-1]/Bath[u,v])**(1/4)
					num += 1
				if FineBath[u,v+1] < FineBath[u,v] and np.all(np.isnan(Fine_computed_heights[u,v+1,:])!= True):
					eta4[:] = Fine_computed_heights[u,v+1,:]#*(Bath[u,v+1]/Bath[u,v])**(1/4)
					num += 1
				for n in range (N):
					if num != 0:
						Fine_computed_heights[u,v,n] = (eta1[n] + eta2[n] + eta3[n] + eta4[n])/num
					else:
						Fine_computed_heights[u,v,n] = Fine_computed_heights[int(idx_list[0,i-1]),int(idx_list[1,i-1]),n]
			except:
				if (not x in xind) and (not y in yind):
					neighx = u
					neighy = v
					count = 4
					x,y,check = neighbour_check(neighx,neighy,FineBath,xind,yind,min_x,max_x,min_y,max_y,count)
					if check == False:
						x,y = int(idx_list[0,i-1]),int(idx_list[1,i-1])
				for n in range (N):
					Fine_computed_heights[u,v,n] = Fine_computed_heights[x,y,n]

		elif FineBath[u,v] >= 0.0:
			break

		if np.mod(i,maxint)==0:
			print(int(i/len(Betas.flatten())*100),"% complete" )
			print("beta = ", Betas[u,v])
			print("Depth = ", FineBath[u,v])
			print("Forecasted Wave Height = ", Fine_computed_heights[u,v,:])
			print("\n")


	return Fine_computed_heights
