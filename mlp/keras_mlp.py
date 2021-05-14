"""
A Multilayer Perceptron approach used to forecast the loacl response


"""
from __future__ import print_function
from keras import layers
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from make_array import *
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset
from keras.callbacks import CSVLogger


global_bathy = ""
folders = [] # List of folders which contain the folded dataset
global_sim = []
local_eta = []

for folder in folders:
	i = 0
	print(folder)
	for f in sorted(listdir(folder)):
		if isfile(join(folder, f)):
			if "hmax" in f:
				if i % 2 == 0:
					global_sim.append(join(folder, f))
				else:
					local_eta.append(join(folder, f))
				i += 1
print(len(global_sim),len(local_eta))
print("Loading coarse data into arrays")
lon, lat, Bath, SimRes = make_array(global_bathy, global_sim)

local_bathy = ""
print("Loading fine data into arrays")
finelon, finelat, FineBath, FineSimRes = make_array(local_bathy, local_eta, coarse=False)


# Cropping the Data to smaller domains
SimRes = SimRes[150:300,300:450,:]
FineSimRes = FineSimRes[525:825,800:1200,:]
num = len(SimRes[0,0,:])

# Network Parameters
n_hidden = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
n_input = (len(SimRes[:,0,0])*len(SimRes[0,:,0]))
n_output = (len(FineSimRes[:,0,0])*len(FineSimRes[0,:,0]))
num_epochs = 200
# Flatten the arrays
SimRes_Flat = SimRes.reshape(n_input,num)
FineSimRes_Flat = FineSimRes.reshape(n_output,num)
print(n_input,n_output)

# Set up the Model
def create_model(n_hidden_1,n_hidden_2,n_input):
	model = Sequential()
	model.add(Dense(n_hidden, input_dim=n_input, activation="relu", name="dense_1", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dropout(0.2))
	model.add(Dense(n_hidden, activation='relu', name="dense_2", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dropout(0.2))
	model.add(Dense(n_hidden, activation='relu', name="dense_3", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dropout(0.2))
	model.add(Dense(n_output, activation="relu", name="predictions"))
	# Compile model
	model.compile(optimizer="Adam",loss="mean_squared_error")
	return model
model = create_model(n_hidden_1,n_hidden_2,n_input)

print("Fit model on training data")
csv_logger = CSVLogger('log.csv', append=True, separator=';')
history = model.fit(SimRes_Flat.T,FineSimRes_Flat.T,epochs=num_epochs,callbacks=[csv_logger])


# #### Prediction #####
folder = "" #Path to folder containing the testing set
global_sim = []
local_eta = []
i = 0
for f in sorted(listdir(folder)):
	if isfile(join(folder, f)):
		if "hmax" in f:
			if i % 2 == 0:
				global_sim.append(join(folder, f))
			else:
				local_eta.append(join(folder, f))
			i += 1
print(len(global_sim),len(local_eta))
print("Loading coarse data into arrays")
lon, lat, Bath, test_data = make_array(global_bathy, global_sim)

num = len(test_data[0,0,:])
test_data = test_data[150:300,300:450,:]
test_data_f = test_data.reshape(n_input,num)

print("Generate predictions")
predictions = model.predict(test_data_f.T)
print("predictions shape:", predictions.shape)
predictions=np.array(predictions)
predictions=predictions.reshape((num,len(FineSimRes[:,0,0]),len(FineSimRes[0,:,0])))
print("predictions shape:", predictions.shape)
np.save("Predictions.npy",predictions)
