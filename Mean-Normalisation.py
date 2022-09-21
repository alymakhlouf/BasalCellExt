"""
READS ALL NUCLEAR FILES CSV FILES IN AN EXPERIMENT TO DETERMINE THE MEAN CHANNEL 2 INTENSITY AND ITS DISTRIBUTION
USES THIS COMPUTED VALUE TO DETERMINE THE 'MEAN-NORMALISED CHANNEL 2 INTENSITY' AND WRITE IT TO NUCLEAR CSV FILES
PLOTS HISTOGRAMS OF CHANNEL 2 INTENSITIES (INCLUDING LOG-TRANSFORMED DATA)

@author: Aly Makhlouf
"""

## READS OUTPUT NUCLEUS DATA FILES

import math
import tqdm
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

culture = '3D Microcavity'

experiment = '24h_2'

#sub_experiment = 'with dox'
#sub_experiment = 'without dox'

sub_folder = 'Supplementary Analysis\\'
#sub_folder = sub_experiment + '\\' # use for 2D monolayer files

directory = r'\\istore.lmb.internal/mshahbazi_lab_share/Aly/Spheroid Image Analysis/'

sub_path = directory + culture + '//' + experiment
path = sub_path + '/' + 'Supplementary Analysis'
#path = sub_path + '/' + sub_experiment # use for 2D monolayer files

spheroid_files = glob.glob(sub_path + '/*Spheroid_Data.csv')
nuclear_files = glob.glob(path + '/*Nucleus_Data.csv')
#nuclear_files = glob.glob(path + '/*.csv') # use for 2D monolayer files

# set log-mean directory
log_mean_directory = 'C:\\Users\\Lancaster Lab\\.spyder-py3\\Log Mean\\'

#%%
data_array = []
data_MN_array = [] # for mean-normalised data

spheroid_idx = 0
file_track = ''

## load log-mean value (mean) if already computed
try:
    nuc_log_mean = np.load(log_mean_directory + culture + '_' + experiment + '_Log_Mean.npy')
except OSError as e:
    nuc_log_mean = 0
    pass

for nuclear_file in tqdm.tqdm(nuclear_files[0:]):
    
    start = nuclear_file.find(sub_folder) + len(sub_folder)
    end = nuclear_file.find('_Spheroid', start)
    file_name = nuclear_file[start:end]
    print(file_name)
    if file_name == file_track:
        spheroid_idx += 1
    else: 
        spheroid_idx = 0
    
    print(spheroid_idx)
    file_track = file_name
    
    # extract file name compatible with planar mean DAPI dictionary
    start = nuclear_file.find(sub_folder) + len(sub_folder)
    end = nuclear_file.find('_Nucleus', start)
    DAPI_name= nuclear_file[start:end]
    
    file = pd.read_csv(nuclear_file)
    data = file['BSDN channel 2 intensity (mean)'].to_numpy()

    print(data)

    mean_norm_data = math.e**(np.log(data) - nuc_log_mean)
    mean_norm_data[np.isnan(mean_norm_data)] = 0.01
    
    #data = file['DAPI-Normalised aPKCi-TagRFP (left)'].to_numpy() # use for 2D monolayer
    
    data_array += list(data) # compile data
    data_MN_array += list(mean_norm_data) # compile mean-normalised data
    
    ## USE FOR 3D SPHEROID ANALYSIS (WT and cMyc i)
    spheroid_match = [match for match in spheroid_files if file_name in match][0] # find spheroid file with matching name
    spheroid_file = pd.read_csv(spheroid_match)
    
    lumen_idx = spheroid_file.loc[spheroid_idx, 'spheroid'] # get spheroid ID
    spheroid_z = (spheroid_file.loc[spheroid_file['spheroid'] == lumen_idx]['z-coord']).to_numpy()
    
    ## USE FOR ALL FILES
    #update nuclear classification
    for idx, value in enumerate(data):
        file.loc[idx,'mean-normalised channel 2 intensity'] = float(str(round(mean_norm_data[idx],3)))
    
    file.to_csv(nuclear_file, index=False)
    #

data_array = np.array(data_array)
data_MN_array = np.array(data_MN_array)

#compute mean for each experiment

if nuc_log_mean == 0:
    bra_log = np.log(data_array[np.where(data_array != 0)])
    bra_log_mean = bra_log[~np.isnan(bra_log)].mean()
    bra_log_std =  bra_log[~np.isnan(bra_log)].std()
    bra_mean = math.e**(bra_log_mean + bra_log_std / 2)
    np.save(log_mean_directory + culture + '_' + experiment + '_Log_Mean', bra_log_mean)

plt.figure()
plt.hist(data_MN_array, bins = 100)
plt.hist(np.log(data_MN_array[np.where(data_MN_array != 0)]), bins = 100)
plt.show()
    
