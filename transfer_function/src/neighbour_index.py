#-*-coding:utf-8-*-
import numpy as np

def neighbour_index(FineBath,idx_list):

	max_x = np.max(idx_list[0,:])
	min_x = np.min(idx_list[0,:])
	max_y = np.max(idx_list[1,:])
	min_y = np.min(idx_list[1,:])
	u,v = int(idx_list[0,0]),int(idx_list[1,0])
	neighbour = np.zeros((2,len(idx_list[0,:])), dtype=int)
	for i in range(1, len(idx_list[0,:])):
		x,y=u,v
		u,v= int(idx_list[0,i]),int(idx_list[1,i])
		### Neighbour Region #####
		xind = [u-1,u,u+1]
		yind = [v-1,v,v+1]
		if (not x in xind) or (not y in yind):
			count = 400
			x,y,check = neighbour_check(u,v,FineBath,xind,yind,min_x,max_x,min_y,max_y,count)
			if (check == False):
				# x,y = int(idx_list[0,i-1]),int(idx_list[1,i-1])
				x,y = int(idx_list[0,0]),int(idx_list[1,0])
			neighbour[0,i], neighbour[1,i]= int(x),int(y)
		else:
			neighbour[0,i], neighbour[1,i] = int(x),int(y)

	return neighbour



def neighbour_check(neighx,neighy,FineBath,xind,yind,min_x,max_x,min_y,max_y,count):
	tempBath = FineBath[neighx,neighy]
	check = False
	num = 0
	radius = 1
	while check == False and num <= count:
		for k in xind:
			for j in yind:
				if (min_x < k < max_x) and (min_y < j < max_y):
					if (FineBath[k,j] < tempBath):
						neighx = k
						neighy = j
						tempBath = FineBath[neighx,neighy]
						check = True
		radius += 1
		num += 1
		xind = np.arange(neighx-radius, neighx+radius+1, 1)
		yind = np.arange(neighy-radius, neighy+radius+1, 1)

	return neighx,neighy,check
