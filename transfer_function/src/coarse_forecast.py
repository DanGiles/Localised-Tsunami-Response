#!/usr/bin/env python
# Daniel Giles (2021), UCD, Ireland
import numpy as np
def coarse_forecast(SimRes, Bath, idx_list, lowlim, highlim):
	'''
	Takes wave height from coarse forecast, crops at lowlim bathymetry value and forecasts using Green's Law to highlim.
	SimRes: Coarse Grid simulation results
	Bath: Coarse Grid Bathymetry
	idx_list: Index List of sorted bathymetry
	lowlim: Cut off for value for which wave heights are taken directly from the coarse results.
	highlim: Upper limit of bathymetry value at which the Green's law is used.
	'''
	print("Regional Forecasts")
	print("lowlim_coarse = ", lowlim)
	print("highlim_coarse = ", highlim)
	east,north,N =np.shape(SimRes)
	#print(N)
	Computed_heights = np.copy(SimRes)
	### Domain max/min #####
	max_x = np.max(idx_list[0,:])
	min_x = np.min(idx_list[0,:])
	max_y = np.max(idx_list[1,:])
	min_y = np.min(idx_list[1,:])
	u,v = int(idx_list[0,0]),int(idx_list[1,0])
	#copy results for bathy>500m
	for i in range(0, east*north):
		#use Green's law between 500 and 50m
		x,y=u,v
		u,v= int(idx_list[0,i]),int(idx_list[1,i])
		### Neighbour Region #####
		xind = [u-1,u,u+1]
		yind = [v-1,v,v+1]
		if Bath[u,v]<lowlim:
			for n in range (N):
				Computed_heights[u,v,n] = SimRes[u,v,n]

		elif lowlim <= Bath[u,v] < highlim:
			'''
			Use Geen ’s law
			'''
			if (not x in xind) and (not y in yind):
				neighx = u
				neighy = v
				count = 3
				x,y,check = neighbour_check(neighx,neighy,Bath,xind,yind,min_x,max_x,min_y,max_y,count)
				if (check == False):
					x,y = int(idx_list[0,i-1]),int(idx_list[1,i-1])
			for n in range (N):
				Computed_heights[u,v,n] = Computed_heights[x,y,n]*(Bath[x,y]/Bath[u,v])**(1/4) #0 if bathy<50m

		elif highlim<=Bath[u,v]<0.0:
			for n in range (N):
				Computed_heights[u,v,n] = 0.0

	print("Regional Forecasts completed \n")
	return Computed_heights

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
						tempBath = FineBath[neighx,neighy]
						check = True
					else:
						radius += 1
						count += 1
		xind = np.arange(neighx-radius, neighx+radius+1, 1)
		yind = np.arange(neighy-radius, neighy+radius+1, 1)
		# print(xind,yind)
	# if all(i > 0 for i in Fine_computed_heights[neighx,neighy,:]):
	return neighx,neighy,check
