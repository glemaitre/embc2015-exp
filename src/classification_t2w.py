# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:19:48 2015

@author: lemaitre
"""

# In order to plot some stuff
import matplotlib.pyplot as plt
# In order to manipulate some stuff
import numpy as np
# In order to classifiy some stuff
### Random forest
from sklearn.ensemble import RandomForestClassifier

# Load the data file from the numpy npz file
data_norm_rician = np.load('../data/clean/data_norm_rician.npy')
data_norm_gaussian = np.load('../data/clean/data_norm_gaussian.npy')
data_t2w_norm = np.load('../data/clean/data_raw_norm.npy')
label = np.load('../data/clean/label.npy')
patient_sizes = np.load('../data/clean/patient_sizes.npy')

print 'Data loaded ----> Go for some LOPO classification'

# Make the classification for each patient
for pt in xrange(len(patient_sizes)):
    
    ##### LOPO TRAINING - TESTING #####
    # Compute the cumulative sum of all the index
    cum_idx = np.cumsum(patient_sizes)
     
    # If we take the first patient to test
    if (pt == 0):
        testing_index = np.array(range(0, cum_idx[pt]))
        training_index = np.array(range(cum_idx[pt], cum_idx[-1]))
    # If we take the last patient to test
    elif (pt == len(patient_sizes) - 1):
        testing_index = np.array(range(cum_idx[pt - 1], cum_idx[pt]))
        training_index = np.array(range(0, cum_idx[pt - 1]))
    # Otherwise
    else:
        testing_index = np.array(range(cum_idx[pt - 1], cum_idx[pt]))
        training_index = np.array(range(0, cum_idx[pt - 1]))
        training_index = np.concatenate((training_index, np.array(range(cum_idx[pt], cum_idx[-1]))))

    print 'For the patient #{}, the training set contains {} samples, the training set contains {} samples. Thus the dataset contain {} samples in total.'.format(pt, int(training_index.size), int(testing_index.size), int(testing_index.size) + int(training_index.size))

    # Create a training dataset and a testing dataset
    print 'Creating the training and testing dataset'
    ### TRAINING DATASET ###
    # Raw data
    training_data = data_t2w_norm[training_index]
    training_label = label[training_index]
    # # Gaussian normalisation
    # training_data = data_norm_gaussian[training_index]
    # training_label = label[training_index]
    # # Rician normalisation
    # training_data = data_norm_rician[training_index]
    # training_label = label[training_index]
    ### TESTING DATASET 
    #### Raw data
    testing_data = data_t2w_norm[testing_index]
    testing_label = label[testing_index]
    # # Gaussian normalisation
    # testing_data = data_norm_gaussian[testing_index]
    # testing_label = label[testing_index]
    # # Rician normalisation
    # testing_data = data_norm_rician[testing_index]
    # testing_label = label[testing_index]

    print 'Data extracted ---> We need to balance the training'

    # Find the number of CaP voxels
    nb_cap = np.count_nonzero(training_label == 1)
    # Find the CaP voxels index - keep the first table
    idx_cap = np.nonzero(training_label == 1)[0]
    # Count the number of healthy voxels
    nb_healthy = training_label.size - nb_cap
    # Find the healthy voxels index - keep the first table
    idx_healthy = np.nonzero(training_label == 0)[0]

    # Generate a number of indeces of the size of CaP with range of healthy voxels
    sub_idx_healthy = np.arange(nb_healthy)
    np.random.shuffle(sub_idx_healthy)
    sub_idx_healthy = sub_idx_healthy[:nb_cap]
    # Select the subset
    training_data = np.concatenate((training_data[idx_cap], training_data[idx_healthy[sub_idx_healthy]]))
    training_label = np.concatenate((training_label[idx_cap], training_label[idx_healthy[sub_idx_healthy]]))

    print 'The training set is of size {} samples now.'.format(int(training_label.size))

    print 'We can play with scikit-learn now'

    # Declare our random forest with the following parameters:
    ### Define the number of trees - n_estimators=nb_trees
    ### Use all cores of the computers because we can - n_jobs=1
    ### See what we optimise - verbose=1
    ### The rest of the parameters are setting up by default
    nb_trees = 2000
    rf = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1, verbose=1)

    # Train our random forest
    ### training_data need to be a matrix of nb_samples x nb_features
    rf.fit(np.asmatrix(training_data).T, training_label)
    
