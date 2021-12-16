#!/usr/bin/env python
# Daniel Giles (2021), UCD, Ireland
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
from scipy.optimize import least_squares
lib = cdll.LoadLibrary("../src/cfuncs.so")
# gradf = lib.gradf
# gradf.restype = c_double
# gradf.argtypes = [c_int, c_double, c_double,c_double, c_double, ndpointer(c_double), ndpointer(c_double), ndpointer(c_double), ndpointer(c_double)]
# c_descent = lib.fixedstep
# c_descent.restype = c_double
# c_descent.argtypes = [c_int, c_double, c_double, c_double, c_double, c_double, c_double,ndpointer(c_double), ndpointer(c_double)]


### Lalli Formulation c functions #####
grad_alpha = lib.grad_alpha
# grad_alpha.restype = c_double
grad_alpha.argtypes = [c_int, c_double, c_double,c_double, ndpointer(c_double), ndpointer(c_double), ndpointer(c_double), ndpointer(c_double)]
c_lalli_descent = lib.lalli
c_lalli_descent.restype = c_double
c_lalli_descent.argtypes = [c_int, c_double, c_double, c_double, c_double, c_double,ndpointer(c_double), ndpointer(c_double)]
# momentum_descent = lib.momentum
# momentum_descent.restype = c_double
# momentum_descent.argtypes = [c_int, c_double, c_double, c_double, c_double, c_double, c_double,ndpointer(c_double), ndpointer(c_double)]

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

def coarse_to_fine(SimRes, Bath, FineSimRes, FineBath, Lat, Lon, FineLat, FineLon, alpha, alpha_guess, opt = {'grad_des', 'ols'}):
	'''
	Interpolates the maximum wave heights from the coarse grid to the fine grid.
	'''
	# (east, north)=np.shape(Bath)
	# (fine_east, fine_north,N)=np.shape(FineSimRes)
	# '''
	# Calculates the height at the minimum depth in the fine grid
	# Interpolation with adjacent coarse grid cells
	# '''
	'''
	get closest big cell to small cell (imin, jmin)
	'''
	u,v= np.argmin(np.abs(Lat-FineLat)),  np.argmin(np.abs(Lon-FineLon))
	# Small coarse latitude and longitude tables over deepest point
	num = 0
	radius = 0
	while Bath[u,v] > FineBath:
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
	Lon_grid=Lon[v-1-radius:v+2+radius]
	Lat_grid=Lat[u-1-radius:u+2+radius]
	Z = np.zeros(len(SimRes[0,0,:]))
	for n in range(len(SimRes[0,0,:])):
		Z_grid = SimRes[u-1-radius:u+2+radius,v-1-radius:v+2+radius,n]
		B_grid = Bath[u-1-radius:u+2+radius,v-1-radius:v+2+radius]
		Z[n],B=interpolate(Lat_grid,Lon_grid,Z_grid, B_grid,FineBath)
	if opt == 'grad_des':
		alphatest = c_lalli_descent(len(SimRes[0,0,:]),B,FineBath,alpha_guess,learning_rate,eps,Z[:],FineSimRes[:])
		alpha = alphatest
		# eta = np.zeros((len(Z[:]),1))
		# eta[:,0] = Z[:]*alpha*(B/FineBath)**(1/4)
		eta = Z[:]*alpha*(B/FineBath)**(1/4)
	else:
		# print(alpha_guess, np.array(alpha_guess))
		res_lsq = least_squares(extended_greens_L2, np.array(alpha_guess), args=(B,FineBath,Z[:],FineSimRes[:]),loss='soft_l1')
		# print(res_lsq)
		# res_lsq = least_squares(extended_greens_L2, np.array(alpha_guess[u,v]), args=(FineBath[x,y],FineBath[u,v],Fine_computed_heights[x,y,arg_waves],FineSimRes[u,v,arg_waves]))
		alpha = res_lsq.x[0]
		# eta = np.zeros(len(Z[:]))
		eta = Z[:]*alpha*(B/FineBath)**(1/4)

	return eta, alpha

def alpha_calculate(SimRes, Bath, FineSimRes, FineBath, Lat, Lon, FineLat, FineLon, idx_list, idx_neighbour, lowlim, highlim, alpha_guess):
	'''
	Forecast all values below lowlim in the fine grid using Green ’s law
	SimRes , FineSimRes : Simulation results for coarse and fine grids
	Bath , FineBath : Bathymetry tables for coarse and fine grids
	Lat , Lon , FineLat , FineLon : latitude and longitude tables
	idx_list : list of of indices corresponding to the positions in the grid of each
	point ranked by bathymetry value from idx_table(FineBath)
	lowlim , highlim : range for Extended Green ’s Law
	hE : typical value
	'''

	(east, north) = np.shape(Bath)
	(fine_east, fine_north,N) = np.shape(FineSimRes)
	Fine_computed_heights = np.zeros((fine_east,fine_north,N))
	alpha = np.zeros((fine_east, fine_north))
	x,y = int(idx_list[0,0]),int(idx_list[1,0])
	# arg_waves = np.argwhere(FineSimRes[x,y,:]>=1e-2)
	# Interpolate wave heights from the coarse grid to the fine one.
	print("Interpolation from coarse to fine")
	# Fine_computed_heights[x,y,arg_waves], alpha[x,y] = coarse_to_fine(SimRes[:,:,arg_waves], Bath, FineSimRes[x,y,arg_waves], FineBath[x,y], Lat, Lon, FineLat[x], FineLon[y], alpha[x,y], alpha_guess[x,y], 'grad_des')
	Fine_computed_heights[x,y,:], alpha[x,y] = coarse_to_fine(SimRes, Bath, FineSimRes[x,y,:], FineBath[x,y], Lat, Lon, FineLat[x], FineLon[y], alpha[x,y], alpha_guess[x,y], 'grad_des')

	print('Difference from interpolation to fine grid and simulation')
	print((alpha[x,y]-alpha_guess[x,y]), alpha[x,y])
	print('\n')
	maxint = int(len(alpha.flatten())/10)
	print("Alpha Calculation and Fine Forecast")
	print("lowlim_fine = ", lowlim)
	print("highlim_fine = ", highlim)

	for i in range(1, fine_east*fine_north):
		'''
		Iterate through the fine grid by increasing bathymetry values
		Marching algorithm
		'''
		x,y = int(idx_neighbour[0,i]),int(idx_neighbour[1,i])
		u,v= int(idx_list[0,i]),int(idx_list[1,i])

		if FineBath [u,v] <= highlim:
			'''
			Use extended Green ’s law ( calculate alpha using a gradient descent )
			'''
			alphatest = c_lalli_descent(len(Fine_computed_heights[x,y,:]),FineBath[x,y],FineBath[u,v],alpha_guess[u,v],learning_rate,eps,Fine_computed_heights[x,y,:],FineSimRes[u,v,:])
			alpha[u,v] = alphatest
			Fine_computed_heights[u,v,:]= Fine_computed_heights[x,y,:]*alpha[u,v]*(FineBath[x,y]/FineBath[u,v])**(1/4)

		elif highlim < FineBath[u,v] < 0.0:
			'''
			Extrapolate to the shoreline
			'''
			# continue
			Fine_computed_heights[u,v,:]= Fine_computed_heights[x,y,:]
			alpha[u,v]= 0.0

		elif FineBath[u,v] >= 0.0:
			break
		if np.mod(i,maxint)==0:
			print(int(i/len(alpha.flatten())*100),"% complete" )
			# print("Difference = ", (FineSimRes [u,v,:]-Fine_computed_heights [u,v,:]))
			print("Alpha difference = ", alpha[u,v] - alpha_guess[u,v])
			print("Depth = ", FineBath[u,v])
			print("\n")

	return alpha, Fine_computed_heights


def OLS_calculate(SimRes, Bath, FineSimRes, FineBath, Lat, Lon, FineLat, FineLon, idx_list, idx_neighbour, lowlim, highlim, alpha_guess):
	'''
	Forecast all values below lowlim in the fine grid using Green ’s law
	SimRes , FineSimRes : Simulation results for coarse and fine grids
	Bath , FineBath : Bathymetry tables for coarse and fine grids
	Lat , Lon , FineLat , FineLon : latitude and longitude tables
	idx_list : list of of indices corresponding to the positions in the grid of each
	point ranked by bathymetry value from idx_table(FineBath)
	lowlim , highlim : range for Extended Green ’s Law
	hE : typical value
	'''

	(east, north) = np.shape(Bath)
	(fine_east, fine_north,N) = np.shape(FineSimRes)
	Fine_computed_heights = np.zeros((fine_east,fine_north,N))
	alpha = np.zeros((fine_east, fine_north))
	x,y = int(idx_list[0,0]),int(idx_list[1,0])
	# arg_waves = np.argwhere(FineSimRes[x,y,:]>=1e-2)
	# Interpolate wave heights from the coarse grid to the fine one.
	print("Interpolation from coarse to fine")
	# Fine_computed_heights[x,y,arg_waves], alpha[x,y] = coarse_to_fine(SimRes[:,:,arg_waves], Bath, FineSimRes[x,y,arg_waves], FineBath[x,y], Lat, Lon, FineLat[x], FineLon[y], alpha[x,y], alpha_guess[x,y], 'ols')
	Fine_computed_heights[x,y,:], alpha[x,y] = coarse_to_fine(SimRes, Bath, FineSimRes[x,y], FineBath[x,y], Lat, Lon, FineLat[x], FineLon[y], alpha[x,y], alpha_guess[x,y], 'ols')

	print('Difference from interpolation to fine grid and simulation')
	print((alpha[x,y]-alpha_guess[x,y]), alpha[x,y])
	print('\n')
	maxint = int(len(alpha.flatten())/10)
	print("Alpha Calculation and Fine Forecast")
	print("lowlim_fine = ", lowlim)
	print("highlim_fine = ", highlim)
	for i in range(1, fine_east*fine_north):
		'''
		Iterate through the fine grid by increasing bathymetry values
		Marching algorithm
		'''
		x,y = int(idx_neighbour[0,i]),int(idx_neighbour[1,i])
		u,v= int(idx_list[0,i]),int(idx_list[1,i])

		if FineBath [u,v] <= highlim:
			'''
			Use extended Green ’s law ( calculate alpha using a gradient descent )
			'''
			res_lsq = least_squares(extended_greens_L2, np.array(alpha_guess[u,v]), args=(FineBath[x,y],FineBath[u,v],Fine_computed_heights[x,y,:],FineSimRes[u,v,:]), loss='soft_l1')
			alpha[u,v] = res_lsq.x[0]
			Fine_computed_heights[u,v,:]= Fine_computed_heights[x,y,:]*alpha[u,v]*(FineBath[x,y]/FineBath[u,v])**(1/4)

		elif highlim < FineBath[u,v] < 0.0:
			'''
			Extrapolate to the shoreline
			'''
			# continue
			Fine_computed_heights[u,v,:]= Fine_computed_heights[x,y,:]
			alpha[u,v]= 0.0

		elif FineBath[u,v] >= 0.0:
			break
		if np.mod(i,maxint)==0:
			print(int(i/len(alpha.flatten())*100),"% complete" )
			print("Alpha difference = ", alpha[u,v] - alpha_guess[u,v])
			print("Depth = ", FineBath[u,v])
			print("\n")

	return alpha, Fine_computed_heights
