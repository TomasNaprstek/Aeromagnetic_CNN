#This code is the associated material for the paper:
#T. Naprstek and R.S. Smith, 2021, Convolution Neural Networks Applied to the Interpretation of Lineaments in Aeromagnetic Data, Geophysics, (preprint).
#
#Any information regarding this code and the associated methodology can be found at the above paper, or at: https://github.com/TomasNaprstek/Aeromagnetic_CNN
#
#Dr. Tomas Naprstek, 2021

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.io

######
#A small function to normalize the windows as they are pulled from the data.
def normalize_numeric_data(data):
    #assuming data is in DataFrame format
    tempdata = data
    mean = np.mean(tempdata)
    std = np.std(tempdata)
    normalized_data = (data-mean) / std
    return normalized_data
######

#First load in the aeromagnetic dataset. Note that the data must be on a regular, equispaced grid.
grid_name = 'test_model.csv'
gridP = pd.read_csv(grid_name,header=None)
gridP = gridP.to_numpy()

#Find the extents of the dataset. Note that this only deals with the number of cells in each direction, not the physical distance.
lengthX = len(gridP)
lengthY = len(gridP[0])

window = 21 #Size of square sliding window. Note that this cannot be changed without re-training the CNN.
win50 = 10 #To deal with padding issues we will start this many cells in, and stop that many cells early.

#Determine the size of the matrices, based on the dataset.
lengthX_w = lengthX-(2*win50)
lengthY_w = lengthY-(2*win50)

#Pre-allocate the matrices.
analyze_window = np.zeros((lengthX_w*lengthY_w,window,window))
window_pos = np.zeros((lengthX_w*lengthY_w,2))
predictdata = np.zeros((lengthX_w*lengthY_w))

setI = 0 #A flag to track the total number of cells analyzed.
#Now loop through the entire dataset. 
#Grab the current window, normalize, save the window to matrix, move on to next window.
for i in range(win50,lengthX-win50):
    for j in range(win50,lengthY-win50):
        tempWin = gridP[i-win50:i+win50+1,j-win50:j+win50+1]
        analyze_window[setI,:,:] = normalize_numeric_data(tempWin)
        window_pos[setI,0] = i
        window_pos[setI,1] = j
        setI = setI + 1

#Reshape the window matrix to the proper format required for the CNN model.
analyze_window = analyze_window.reshape(lengthX_w*lengthY_w,21,21,1)

#Now load in the CNN model and analyze the matrix.
loaded_model = tf.keras.models.load_model('NaprstekSmith_CNN_v1.h5')
predictdata = loaded_model.predict(analyze_window)

#Finally, combine the position and results matrices.
output_results = np.c_[window_pos, predictdata]

#Below are two options for saving the data: csv and Matlab .mat file.
#In both cases the output is: x cell number, y cell number, followed by the % probability for each of the 11 classes. The sum of the probabilites is equal to 1.
np.savetxt("test_model_predictions.csv", output_results, delimiter=",", header="x cell, y cell, 0-25m, 26-50m, 51-75m, 76-100m, 101-125m, 126-150m, 151m-175m, 176-200m, 201-225m, >226m, No lineament")
scipy.io.savemat('test_model_predictions.mat', dict(test_model_predictions=output_results))