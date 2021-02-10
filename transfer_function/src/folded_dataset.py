#!/usr/bin/env python
#-*-coding:utf-8-*-
"""
Generating stratified random datasets from the different seismogenic source
regions. Folder contains coarse and fine grid pairs.
"""
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob
from netCDF4 import MFDataset, Dataset
from scipy import stats
import random
import shutil
folder = ''
coarse = []
fine = []
i = 0
files = sorted(glob.glob(folder))
for f in sorted(listdir(folder)):
	if isfile(join(folder, f)):
		if "hmax" in f:
			if i % 2 == 0:
				coarse.append(join(folder, f))
			else:
				fine.append(join(folder, f))
			i += 1
coarse = np.array(coarse)
fine = np.array(fine)
print(len(coarse),len(fine))
my_list = np.arange(0,len(coarse))
print(my_list)

# Not the cleanest implementation but it splits the dataset into 10 fold datasets
train_num = 0
test_num = 0
for i in range(0,len(coarse)):
	if i < len(coarse):
		# Draw random selection from list and then remove
		n = random.choice(my_list)
		ind = np.argmin(abs(my_list-n))
		my_list = np.delete(my_list,ind)
		if i < int(0.1*len(coarse)):
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set0/')
			shutil.copy2(coarse[n], '../Data/set0/')
		elif int(0.1*len(coarse)) <= i < int(0.2*len(coarse)):
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set1/')
			shutil.copy2(coarse[n], '../Data/set1/')
		elif int(0.2*len(coarse)) <= i < int(0.3*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set2/')
			shutil.copy2(coarse[n], '../Data/set2/')
		elif int(0.3*len(coarse)) <= i < int(0.4*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set3/')
			shutil.copy2(coarse[n], '../Data/set3/')
		elif int(0.4*len(coarse)) <= i < int(0.5*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set4/')
			shutil.copy2(coarse[n], '../Data/set4/')
		elif int(0.5*len(coarse)) <= i < int(0.6*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set5/')
			shutil.copy2(coarse[n], '../Data/set5/')
		elif int(0.6*len(coarse)) <= i < int(0.7*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set6/')
			shutil.copy2(coarse[n], '../Data/set6/')
		elif int(0.7*len(coarse)) <= i < int(0.8*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set7/')
			shutil.copy2(coarse[n], '../Data/set7/')
		elif int(0.8*len(coarse)) <= i < int(0.9*len(coarse)):
			# print(fine[n],coarse[ind])
			train_num = train_num+1
			shutil.copy2(fine[n], '../Data/set8/')
			shutil.copy2(coarse[n], '../Data/set8/')
		else:
			# print(fine[ind],coarse[ind])
			test_num = test_num+1
			shutil.copy2(fine[n], '../Data/set9/')
			shutil.copy2(coarse[n], '../Data/set9/')
	else:
		print(fine[my_list[0]],coarse[my_list[0]])
		test_num = test_num+1
		shutil.copy2(fine[my_list[0]], '../Data/set9/')
		shutil.copy2(coarse[my_list[0]], '../Data/set9/')
print(my_list,train_num,test_num)
