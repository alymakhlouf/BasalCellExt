"""
RUNS BASIC STARDIST 3D ANALYSIS (WITH LUMEN ANALYSIS)
GENERATES HDF5 NUCLEAR 'LABEL' FILES AND CROPPED SPHEROID TIF FILES AS INPUTS FOR ILASTIK CLASSIFIER (POSITIVE/NEGATIVE)
COMPUTES LOCAL CELL DENSITIES IN USER-DEFINED REGIONS
GENERATES SPHEROID AND NUCLEAR CSV FILES

@author: Aly Makhlouf
"""

# import packages 
import numpy as np
import math
import glob
import os
import tqdm
import h5py
import pandas as pd

from itertools import chain
from skimage.measure import label as connected_component

# custom modules
from basalcellext import roi

#%gui qt5 # use for napari in Jupyter notebook, comment otherwise %gui qt5

# specify files to be analysed

by = 'Nanami'

culture = 'Microcavity experiment'

experiment = '24h_2'

# create output directories

directory = r'\\istore.lmb.internal/mshahbazi_lab_share/Nanami Sato/for Aly/Microcavity experiment/'

output_directory = 'D:\\User Data\\Aly\\StarDist\\' + by + '\\' + culture + '\\' + experiment + '\\'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
supplementary_directory = output_directory + 'Supplementary Analysis\\'
if not os.path.exists(supplementary_directory):
    os.makedirs(supplementary_directory)
    
label_directory = output_directory + 'Labels\\'
if not os.path.exists(label_directory):
    os.makedirs(label_directory)

path = directory + experiment + '/'
files = glob.glob(path + '/*.tif')
    
spheroid_path = path + '/' + 'Spheroid/' + 'Object Predictions'
spheroid_files = glob.glob(spheroid_path + '/*.h5')
    
lumen_path = path + '/' + 'Lumen/' + 'Object Predictions'
lumen_files = glob.glob(lumen_path + '/*.h5')

supplementary_directory = output_directory + 'Supplementary Analysis\\'
if not os.path.exists(supplementary_directory):
    os.makedirs(supplementary_directory)
    
label_directory = output_directory + 'Labels\\'
if not os.path.exists(label_directory):
    os.makedirs(label_directory)

# READ CSV FILES GENERATED AFTER PRE-PROCESSING
spheroid_data_files = glob.glob(output_directory + '/*Spheroid_Data.csv')
nuclear_data_files = glob.glob(supplementary_directory + '/*Nucleus_Data.csv')

# READ FILES GENERATED AFTER PRE-PROCESSING    
label_files = glob.glob(label_directory + '/*.hdf5')
classifier_files = glob.glob(label_directory + '/*.h5')

file_indices = list(range(len(files)))


for file_idx in tqdm.tqdm(list(chain(file_indices[0:]))):
    # specify input directories
    
    selected_files = files[file_idx:file_idx+1]
    selected_spheroid_files = spheroid_files[file_idx:file_idx+1]
    selected_lumen_files = lumen_files[file_idx:file_idx+1]
    
    for index, file in enumerate(selected_files):
        
        start = file.find(experiment) + len(experiment) + 1
        end = file.find('.tif', start)
        file_name = file[start:end]
        print(file_name)
        
        #Input image (dimensions in um)

        VOX_DEPTH = 1
        VOX_WIDTH = 0.2367424
        VOX_HEIGHT = 0.2367424
            
        scale = [VOX_DEPTH, VOX_HEIGHT, VOX_WIDTH]

        spheroid_match = [match for match in spheroid_data_files if file_name in match][0]
        spheroid_data_file = pd.read_csv(spheroid_match)
        
        spheroid_file =  selected_spheroid_files[index]
        with h5py.File(spheroid_file) as infile:
            print(tuple(_ for _ in infile.keys()))
            spheroid_pred = infile["exported_data"]
            spheroid_pred = spheroid_pred[:,0,:,:] # ilastik object classifier gives all spheroids a common label (1)
            
        # find all spheroids in the spheroid channel 
        spheroid_mask = connected_component(spheroid_pred) # convert to an array [z,y,x] with a unique label for each object
        u_spheroid, counts_spheroid = np.unique(spheroid_mask, return_counts=True)
        
        # remove pixels belonging to background class
        u_spheroid = u_spheroid[1:]
        counts_spheroid = counts_spheroid[1:]
       
        u_spheroid = np.array([u_spheroid[counts_spheroid.argmax()]]) # select only largest spheroid for further analysis
        counts_spheroid = np.array([counts_spheroid.max()])
        spheroid_mask[np.where(spheroid_mask!=u_spheroid)] = 0 # erase all other spheroids from the mask
        
        spheroid_count = 0

        for j, spheroid_label in enumerate(u_spheroid[0:]):
            
            window_width_spheroid = 10
            y_min, y_max, x_min, x_max, spheroid_binary = roi.find(spheroid_mask, spheroid_label, window_width_spheroid)
        
            if y_min < 0 or x_min < 0 or y_max > np.shape(spheroid_mask)[1] or x_max > np.shape(spheroid_mask)[2]:
                window_width_spheroid = 0
            
            y_min, y_max, x_min, x_max, spheroid_binary = roi.find(spheroid_mask, spheroid_label, window_width_spheroid)
            if y_min < 0 or x_min < 0 or y_max > np.shape(spheroid_mask)[1] or x_max > np.shape(spheroid_mask)[2]:
                continue
            
            spheroid_file_name = file_name + '_Spheroid_' + str(spheroid_label)
            print(spheroid_file_name)
            
            # skip spheroids that do not have a corresponding nuclear data file
            try:
                nuclear_match = [match for match in nuclear_data_files if spheroid_file_name in match][0]
                nuclear_data_file = pd.read_csv(nuclear_match)
            except IndexError:
                continue

            # load label file storing all segmented nuclear labels
            label_file = [match for match in label_files if spheroid_file_name in match][0]
            hf = h5py.File(label_file, 'r')
            labels = hf.get('dataset')[:] #        
            
            # load ilastik classifier file storing all positive/negative nuclei classifications
            classifier_file = [match for match in classifier_files if spheroid_file_name in match][0] # find classifier file with matching name
            
            with h5py.File(classifier_file) as infile:
                print(tuple(_ for _ in infile.keys()))
                nuc_classifier = infile["exported_data"]
                nuc_classifier = nuc_classifier[:,:,:,0] # ilastik object classifier gives positive and negative channel 2 nuclei
            #%%            
            # # modify nuclei and spheroid data
            positive_count = 0
                   
            for i in range(len(nuclear_data_file)):    
                label = nuclear_data_file.loc[i,'label']
                if np.unique(nuc_classifier[np.where(labels == label)])[0] == 1:  
                    nuclear_data_file.loc[i,'channel_2_expression'] = 'negative'
                    
                elif np.unique(nuc_classifier[np.where(labels == label)])[0] == 2:  
                    nuclear_data_file.loc[i,'channel_2_expression'] = 'positive'
                    positive_count += 1
                    
                else:
                    nuclear_data_file.loc[i,'channel_2_expression'] = 'error'
                

            nuclear_data_file.to_csv(nuclear_match, index=False)
            
            spheroid_data_file.loc[spheroid_count, "# +ve cells"] = positive_count  
            spheroid_data_file.loc[spheroid_count, "% +ve cells"] = float(str(np.round((positive_count / len(nuclear_data_file)) * 100,2)))
        
            spheroid_count += 1
        
        spheroid_data_file.to_csv(spheroid_match, index=False)
