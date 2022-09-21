"""
PRE-PROCESSES IMAGES BEFORE RUNNING 'StarDist3D_Basic.py'
GENERATES HDF5 NUCLEAR 'LABEL' FILES AND CROPPED SPHEROID TIF FILES AS INPUTS FOR ILASTIK CLASSIFIER (POSITIVE/NEGATIVE)
GENERATES SPHEROID AND NUCLEAR CSV FILES

@author: Aly Makhlouf
"""


# import packages 
import tensorflow as tf
import napari
import skimage.io
from sklearn.decomposition import PCA
import numpy as np
import math
import glob
import os
import re
import tqdm
from stardist.models import StarDist3D
from stardist.geometry import dist_to_coord3D
from csbdeep.utils import normalize
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import label as connected_component
from skimage import filters
from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import scipy
import cv2 as cv

# custom modules
from basalcellext import roi
from basalcellext import coordinate

#%%
#%gui qt5 # use for napari in Jupyter notebook, comment otherwise %gui qt5

# SPECIFY FILES TO BE ANALYSED

directory = r'\\istore.lmb.internal/mshahbazi_lab_share/Nanami Sato/for Aly/'


sub_experiment_1 = '3D images of aPKCi spheroid upon cMyci'


sub_experiment_2 = 'Exp4'


path = directory + sub_experiment_1 + '\\' + sub_experiment_2
files = glob.glob(path + '/*.tif')
    
spheroid_path = path + '/' + 'Spheroid/' + 'Object Predictions'
spheroid_files = glob.glob(spheroid_path + '/*.h5')

file_indeces = list(range(len(files)))


packing_correction = [1,1,0.5,2*np.sqrt(3)-3, np.sqrt(6)-2,np.sqrt(2)-1,np.sqrt(2)-1]

# CREATE OUTPUT DIRECTORIES


by = 'Nanami'
#by = 'Marta'


culture = '3D'


experiment = 'aPKCi Spheroids_3'


output_directory = 'D:\\User Data\\Aly\\StarDist\\' + by + '\\' + culture + '\\' + experiment + '\\' + sub_experiment_1 + '\\' + sub_experiment_2 + '\\'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_culture = '3D aPKCi Spheroids'


output_experiment = 'Nanami'

sub_folder = 'Supplementary Analysis\\'
#sub_folder = sub_experiment + '\\' # use for 2D monolayer files

## store files in an alternate directory, in case you want to rewrite existing files
directory_alt = r'\\istore.lmb.internal/mshahbazi_lab_share/Aly/Spheroid Image Analysis/'

output_sub_path = directory_alt + output_culture + '\\' + output_experiment
output_path = output_sub_path + '/' + 'Supplementary Analysis'
#path = sub_path + '/' + sub_experiment # use for 2D monolayer files

##

supplementary_directory = output_directory + 'Supplementary Analysis\\'
if not os.path.exists(supplementary_directory):
    os.makedirs(supplementary_directory)
    
label_directory = output_directory + 'Labels\\'
if not os.path.exists(label_directory):
    os.makedirs(label_directory)
#%%
for file_idx in tqdm.tqdm(file_indeces[0:]):
    # specify input directories

    sub_experiment = sub_experiment_1 + '\\' + sub_experiment_2
    
    selected_files = files[file_idx:file_idx+1]
    selected_spheroid_files = spheroid_files[file_idx:file_idx+1]

     
    # import model
    
    model = StarDist3D.from_pretrained('3D_demo')
    
    for index, file in enumerate(selected_files):
        
        start = file.find(sub_experiment) + len(sub_experiment)
        end = file.find('.tif', start)
        file_name = file[start+1:end]
        print(file_name)
#%%        
        #Input image (dimensions in um)

        VOX_DEPTH = 1
        if by == 'Nanami':
            VOX_WIDTH = 0.1136365
            VOX_HEIGHT = 0.1136365
        
        if by == 'Marta':
            VOX_WIDTH = 0.1893939
            VOX_HEIGHT = 0.1893939
        if 'Series005' in file_name and by == 'Marta' and 'Bra' in experiment:
            VOX_WIDTH = 0.3787879
            VOX_HEIGHT = 0.3787879
        
        scale = [VOX_DEPTH, VOX_HEIGHT, VOX_WIDTH]
        
        # create list to compile spheroid info
        rows_spheroid = []
        
        # create a dummy row to start the spheroid counter at 1
        spheroid_info = {}
        spheroid_info["spheroid"] = 0
        spheroid_info["x-coord"] = 0
        spheroid_info["y-coord"] = 0
        spheroid_info["z-coord"] = 0
        spheroid_info["surface area (um^2)"] = 0
        spheroid_info["volume (um^3)"] = 0
        spheroid_info["number of cells"] = 0
        spheroid_info["density (cells/um^3)"] = 0
        spheroid_info["# +ve cells"] = 0
        spheroid_info["% +ve cells"] = 0
        
        
        rows_spheroid.append(spheroid_info)
        
        # import image
        image = skimage.io.imread(file)[:,0,...] # DAPI channel
        image_channel_2 = skimage.io.imread(file)[:,1,...] # channel 2
        image_channel_f = skimage.io.imread(file)[:,-1,...] # final channel (channel 4 or 5, depending on image)
    
        spheroid_file =  selected_spheroid_files[index]
        with h5py.File(spheroid_file) as infile:
            print(tuple(_ for _ in infile.keys()))
            spheroid_pred = infile["exported_data"]
            spheroid_pred = spheroid_pred[:,0,:,:] # ilastik object classifier gives all spheroids a common label (1)
#%%            
        # find all spheroids in the spheroid channel 
        spheroid_mask = connected_component(spheroid_pred) # convert to an array [z,y,x] with a unique label for each object
        u_spheroid, counts_spheroid = np.unique(spheroid_mask, return_counts=True)
        
        # remove pixels belonging to background class
        u_spheroid = u_spheroid[1:]
        counts_spheroid = counts_spheroid[1:]
        
        # create a list variable to store the data tables for each individual spheroid
        final_table = [0]*np.size(u_spheroid)
    
        spheroid_count = 0
        
        for j, spheroid_label in enumerate(u_spheroid):
            
            window_width_spheroid = 10
            y_min, y_max, x_min, x_max, spheroid_binary = roi.find(spheroid_mask, spheroid_label, window_width_spheroid)
        
            if y_min < 0 or x_min < 0 or y_max > np.shape(spheroid_mask)[1] or x_max > np.shape(spheroid_mask)[2]:
                window_width_spheroid = 0
            
            y_min, y_max, x_min, x_max, spheroid_binary = roi.find(spheroid_mask, spheroid_label, window_width_spheroid)
            if y_min < 0 or x_min < 0 or y_max > np.shape(spheroid_mask)[1] or x_max > np.shape(spheroid_mask)[2]:
                continue
            
            spheroid_file_name = file_name + '_Spheroid_' + str(spheroid_label)
            print(spheroid_file_name)
            
            z = np.arange(0,image.shape[0])
            y = np.arange(0,image.shape[1])
            x = np.arange(0,image.shape[2])
            
            zgrid, ygrid, xgrid = np.meshgrid(z,y,x, indexing='ij')
            
            sph_z_c = zgrid[spheroid_binary!=0].mean()
            sph_y_c = ygrid[spheroid_binary!=0].mean()
            sph_x_c = xgrid[spheroid_binary!=0].mean()
            
            scaled_sph_centroid = np.array([sph_z_c, sph_y_c, sph_x_c]) * scale
            
            # crop out one spheroid for faster computation
            image_crop = image[:, y_min:y_max, x_min:x_max]  
            image_channel_2_crop = image_channel_2[:, y_min:y_max, x_min:x_max]  
            image_channel_f_crop = image_channel_f[:, y_min:y_max, x_min:x_max] 
            
            # crop to roi
            spheroid_mask_crop = spheroid_binary[:, y_min:y_max, x_min:x_max]
            
            # pca to return the axes and orientations of spheroid
            coords_spheroid = np.vstack(np.where(spheroid_mask == spheroid_label)).T
            
            spheroid_roi = spheroid_mask[coords_spheroid[0][0]:coords_spheroid[-1][0],y_min:y_max,x_min:x_max]
            spheroid_volume = np.round(counts_spheroid[j]*VOX_DEPTH*VOX_WIDTH*VOX_HEIGHT,3) # in um^3
            
            spheroid_surface_area = 0
            for i in range(0,np.shape(spheroid_roi)[0]):
                spheroid_slice = regionprops(spheroid_roi[i,:,:].astype(int))
                if spheroid_slice == []:
                    continue
                spheroid_surface_area += spheroid_slice[0].perimeter
                
            spheroid_surface_area = spheroid_surface_area*VOX_DEPTH*VOX_WIDTH # in um^2
            
            # add spheroid data
            
            spheroid_info = {}
            spheroid_info["spheroid"] = int(spheroid_label)
            spheroid_info["x-coord"] = math.ceil(sph_x_c)
            spheroid_info["y-coord"] = math.ceil(sph_y_c)
            spheroid_info["z-coord"] = math.ceil(sph_z_c)
            spheroid_info["surface area (um^2)"] = float(str(np.round(spheroid_surface_area,3)))
            spheroid_info["volume (um^3)"] = float(str(spheroid_volume)) # size is calculated based on the original image
                   
            # resize image (make the image scale compatible with the images used to pre-train the model)
            
            resize_factor = (1/VOX_HEIGHT)/2 # based on input image (2,1,1) anisotropy (5.28, shown above) scaled to the pre-trained model anisotropy (2,1,1)
            image_crop_down = resize(image_crop, (image_crop.shape[0], 
                                    image_crop.shape[1] // resize_factor, 
                                    image_crop.shape[2] // resize_factor),
                                    anti_aliasing=False) * 255
            
            image_channel_2_crop_down = resize(image_channel_2_crop, (image_channel_2_crop.shape[0], 
                                    image_channel_2_crop.shape[1] // resize_factor, 
                                    image_channel_2_crop.shape[2] // resize_factor),
                                    anti_aliasing=False) * 255
            
            image_channel_f_crop_down = resize(image_channel_f_crop, (image_channel_f_crop.shape[0], 
                                    image_channel_f_crop.shape[1] // resize_factor, 
                                    image_channel_f_crop.shape[2] // resize_factor),
                                    anti_aliasing=False) * 255
            
            # make sure anti-aliasing is 'False'
            spheroid_mask_crop_down = resize(spheroid_mask_crop, (spheroid_mask_crop.shape[0], 
                                    spheroid_mask_crop.shape[1] // resize_factor, 
                                    spheroid_mask_crop.shape[2] // resize_factor),
                                    anti_aliasing=False) * 255
            
            # find all non-background, downsized connected components in the spheroid mask
            if all(spheroid_mask_crop_down[0].flatten() == 0) == True:
                spheroid_mask_crop_down = connected_component(spheroid_mask_crop_down != spheroid_mask_crop_down[0]) #taking the first Z-plane as a reference for background
            
            else:     
                spheroid_mask_crop_down = connected_component(spheroid_mask_crop_down != spheroid_mask_crop_down[0,0,0])
                u_sph_mask_crop_down, counts_sph_mask_crop_down = np.unique(spheroid_mask_crop_down, return_counts=True)
                u_sph_mask_crop_down = u_sph_mask_crop_down[1:]
                counts_sph_mask_crop_down = counts_sph_mask_crop_down[1:]
                spheroid_mask_crop_down = spheroid_mask_crop_down == u_sph_mask_crop_down[np.argmax(counts_sph_mask_crop_down)]
            
            # normalize image
            img = normalize(image_crop_down, 1,99.8, axis=(0,1,2))
            
            # predict segmentation
            labels, details = model.predict_instances(img, prob_thresh=0.7, nms_thresh=0.2)
            
            labels_export = labels.astype(np.uint32)
            labels_file = h5py.File(label_directory + '\\Labels_' + file_name + '_Spheroid_' + str(spheroid_label) + '.hdf5', "w")
            labels_dset = labels_file.create_dataset("dataset", data = labels_export)
            labels_file.close()
            
            # calculate 3D coordinates from rays and distances
            coord = dist_to_coord3D(details['dist'], details['points'], details['rays_vertices'])
            
            # create list of colors to randomly use for objects
            colormaps = ['yellow', 'green', 'red','blue','cyan','magenta']
            colors = np.random.choice(colormaps, len(coord))
            
            # create a common coordinate system for plotting the objects
            z = np.arange(0,img.shape[0])
            y = np.arange(0,img.shape[1])
            x = np.arange(0,img.shape[2])
            
            zgrid, ygrid, xgrid = np.meshgrid(z,y,x, indexing='ij')
            
            # # create napari viewer
            # viewer = napari.Viewer(ndisplay=3)
            # viewer.add_image(img, blending = 'opaque', scale=(2,1,1))
            # viewer.add_labels(labels, blending = 'translucent', scale=(2,1,1))
            
            # background subtraction and planar DAPI normalisation   
            planar_mean_DAPI = np.zeros(np.shape(image_crop_down)[0])
            img_2_background_sub = np.zeros(np.shape(image_channel_2_crop_down))
            img_2_planar_DAPI_normalised = np.zeros(np.shape(image_channel_2_crop_down))
            img_f_background_sub = np.zeros(np.shape(image_channel_f_crop_down))
            img_f_planar_DAPI_normalised = np.zeros(np.shape(image_channel_f_crop_down))
            
            for z in range(0,np.shape(img)[0]):
                # background subtraction
                try:
                    background_2_array = image_channel_2_crop_down[z,:,:][np.where(((spheroid_mask_crop_down[z,:,:] != 0)*1 & (labels[z,:,:] == 0)*1) == 1)]
                    background_2 = np.percentile(background_2_array,75) # take 75th-percentile of non-nuclear spheroid pixel intensities as background
                    print(background_2)
                    background_f_array = image_channel_f_crop_down[z,:,:][np.where(((spheroid_mask_crop_down[z,:,:] != 0)*1 & (labels[z,:,:] == 0)*1) == 1)]
                    background_f = np.percentile(background_f_array,75) # take 75th-percentile of non-nuclear spheroid pixel intensities as background
                    print(background_f)
                    img_2_background_sub[z,:,:] = (image_channel_2_crop_down[z,:,:] - background_2).clip(min=0) # eliminate negative values
                    img_f_background_sub[z,:,:] = (image_channel_f_crop_down[z,:,:] - background_f).clip(min=0) # eliminate negative values
                
                except IndexError:
                    continue
                
                planar_mean_DAPI[z] = image_crop_down[z,:,:][np.where(labels[z,:,:] != 0)].mean()
                img_2_planar_DAPI_normalised[z] = img_2_background_sub[z,:,:] / planar_mean_DAPI[z] # normalised by planar DAPI intensity to account for variations in Z position
                img_f_planar_DAPI_normalised[z] = img_f_background_sub[z,:,:] / planar_mean_DAPI[z] # normalised by planar DAPI intensity to account for variations in Z position
            
            skimage.io.imsave(label_directory + '\\Crop_DN_' + file_name + '_Spheroid_' + str(spheroid_label) + '.tif', img_2_planar_DAPI_normalised)
            
#%%            
            # add nuclei data
            nucleus_count = 0
            positive_count = 0
            rows_nuclei = []
            mean_DAPI = np.zeros(len(np.unique(labels)[1:]))
            channel_2_raw = np.zeros(len(np.unique(labels)[1:]))
            channel_2_raw_75 = np.zeros(len(np.unique(labels)[1:]))
            channel_2_bs = np.zeros(len(np.unique(labels)[1:]))
            channel_2_bs_75 = np.zeros(len(np.unique(labels)[1:]))
            channel_2 = np.zeros(len(np.unique(labels)[1:]))
            channel_2_75 = np.zeros(len(np.unique(labels)[1:])) # takes 75th percentile expression instead of mean
            channel_f = np.zeros(len(np.unique(labels)[1:]))
            x_c = np.zeros(len(np.unique(labels)[1:])).astype(int)
            y_c = np.zeros(len(np.unique(labels)[1:])).astype(int)
            z_c = np.zeros(len(np.unique(labels)[1:])).astype(int)
            
            for idx, label in enumerate(np.unique(labels)[1:]):
                
                # exclude nuclei outside the spheroid, based on overlap with spheroid mask
                if all(spheroid_mask_crop_down[np.where(labels==label)] == 0) == True:
                    continue
                
                # exclude nuclei with very low DAPI intensity
                if img[labels==label].mean() < 0.15:
                    continue
                
                # exclude tiny mis-segmented nuclei
                if (labels==label).sum()*VOX_DEPTH*resize_factor*VOX_WIDTH*resize_factor*VOX_HEIGHT <= 150:
                    continue
                
                x_c[idx] = int(round(xgrid[labels==label].mean()))
                y_c[idx] = int(round(ygrid[labels==label].mean()))
                z_c[idx] = int(round(zgrid[labels==label].mean()))
                
                mean_DAPI[idx] = float(str(np.round(image_crop_down[labels==label].mean(),3)))
                
                channel_2_raw[idx] = image_channel_2_crop_down[labels==label].mean()
                channel_2_raw_75[idx] = np.percentile(image_channel_2_crop_down[labels==label],75)
                channel_2_bs[idx] = img_2_background_sub[labels==label].mean()
                channel_2_bs_75[idx] = np.percentile(img_2_background_sub[labels==label],75)
                channel_2[idx] = img_2_planar_DAPI_normalised[labels==label].mean() # normalised by local DAPI intensity to account for variations in Z position
                channel_2_75[idx] = np.percentile(img_2_planar_DAPI_normalised[labels==label],75)
                channel_f[idx] = img_f_planar_DAPI_normalised[labels==label].mean() # normalised by local DAPI intensity to account for variations in Z position
            
            for idx, label in enumerate(np.unique(labels)[1:]):
                
                # skip nuclei that were excluded in the previous loop
                if x_c[idx] == 0:
                    continue
                
                nucleus_count += 1
                
                nuc_coord = np.array([zgrid[labels==label].mean(), (ygrid[labels==label].mean()*resize_factor) + y_min, (xgrid[labels==label].mean()*resize_factor) + x_min])
                
                nucleus_info = {}
                nucleus_info["label"] = int(label)
                nucleus_info["x-coord"] = math.ceil(nuc_coord[2])
                nucleus_info["y-coord"] = math.ceil(nuc_coord[1])
                nucleus_info["z-coord"] = math.ceil(nuc_coord[0])
                nucleus_info["volume (um^3)"] = float(str(np.round((labels==label).sum()*VOX_DEPTH*resize_factor*VOX_WIDTH*resize_factor*VOX_HEIGHT,3)))
                nucleus_info["DAPI Intensity"] = mean_DAPI[idx]
                nucleus_info["raw channel 2 intensity (mean)"] = float(str(np.round(channel_2_raw[idx],3)))
                nucleus_info["raw channel 2 intensity (75th-percentile)"] = float(str(np.round(channel_2_raw_75[idx],3)))
                nucleus_info["BS channel 2 intensity (mean)"] = float(str(np.round(channel_2_bs[idx],3)))
                nucleus_info["BS channel 2 intensity (75th-percentile)"] = float(str(np.round(channel_2_bs_75[idx],3)))
                nucleus_info["BSDN channel 2 intensity (mean)"] = float(str(np.round(channel_2[idx],3)))
                nucleus_info["BSDN channel 2 intensity (75th-percentile)"] = float(str(np.round(channel_2_75[idx],3)))
                nucleus_info["mean-normalised channel 2 intensity"] = ''
                nucleus_info["channel 2 expression"] = ''
                
                # only include for pHH3 analysis ('3D images of aPKCi spheroid upon cMyci' dataset)
                nucleus_info["BSDN channel 5 intensity (mean)"] = float(str(np.round(channel_f[idx],3)))
                #
                
                rows_nuclei.append(nucleus_info)

            if nucleus_count == 0:
                continue

            final_table[j] = pd.DataFrame.from_dict(rows_nuclei, orient='columns')
            final_table[j] = final_table[j] # should exclude background label (label 0)
            final_table[j].to_csv(supplementary_directory + '\\' + file_name + '_Spheroid_' + str(spheroid_label) + '_Nucleus_Data.csv')
            
            spheroid_info["number of cells"] = nucleus_count # should exclude background label (label 0)
            spheroid_info["density (cells/um^3)"] = float(str(np.round(((nucleus_count / spheroid_volume)*10**6),2)))
            spheroid_info["# +ve cells"] = positive_count
            spheroid_info["% +ve cells"] = float(str(np.round((spheroid_info["# +ve cells"] / nucleus_count) * 100,2)))
            
            rows_spheroid.append(spheroid_info)

            spheroid_count += 1
        
        final_table_spheroid = pd.DataFrame.from_dict(rows_spheroid, orient='columns')
        final_table_spheroid = final_table_spheroid[1:]
        final_table_spheroid.to_csv(output_directory + file_name + '_Spheroid_Data' + '.csv')
