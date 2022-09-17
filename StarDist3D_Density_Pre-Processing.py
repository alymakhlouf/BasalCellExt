"""
RUNS BASIC STARDIST 3D ANALYSIS (WITH LUMEN ANALYSIS)
GENERATES HDF5 NUCLEAR 'LABEL' FILES AND CROPPED SPHEROID TIF FILES AS INPUTS FOR ILASTIK CLASSIFIER (POSITIVE/NEGATIVE)
COMPUTES LOCAL CELL DENSITIES IN USER-DEFINED REGIONS
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
from itertools import chain
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import scipy
import cv2 as cv

# custom modules
import roi
import coordinate


#%gui qt5 # use for napari in Jupyter notebook, comment otherwise %gui qt5

# specify files to be analysed

by = 'Nanami'

culture = 'Microcavity experiment'

experiment = '24h_2'

packing_correction = [1,1,0.5,2*np.sqrt(3)-3, np.sqrt(6)-2,np.sqrt(2)-1,np.sqrt(2)-1]

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

file_indices = list(range(len(files)))

# import model
model = StarDist3D.from_pretrained('3D_demo')

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
        
        # create lists to compile lumen and spheroid info
        rows_lumen = []
        rows_spheroid = []
        # create a dummy row to start the lumen counter at 1
        lumen_info = {}
        lumen_info["spheroid"] = 0
        lumen_info["x-coord"] = 0
        lumen_info["y-coord"] = 0
        lumen_info["z-coord"] = 0
        lumen_info["volume (um^3)"] = 0
        lumen_info["surface area (um^2)"] = 0
        lumen_info["volume convexity"] = 0
        lumen_info["ellipticity"] = 0
        rows_lumen.append(lumen_info)
        
        # create a dummy row to start the spheroid counter at 1
        spheroid_info = {}
        spheroid_info["spheroid"] = 0
        spheroid_info["x-coord"] = 0
        spheroid_info["y-coord"] = 0
        spheroid_info["z-coord"] = 0
        spheroid_info["volume (um^3)"] = 0
        spheroid_info["surface area (um^2)"] = 0
        spheroid_info["ellipticity"] = 0
        spheroid_info["number of lumens"] = 0
        spheroid_info["number of cells"] = 0
        spheroid_info["# +ve cells"] = 0
        spheroid_info["% +ve cells"] = 0
        
        rows_spheroid.append(spheroid_info)
        
        # import image
        image = skimage.io.imread(file)[..., 0] #DAPI channel
        image_channel_2 = skimage.io.imread(file)[..., 1] # channel 2
        image_lumen = skimage.io.imread(file)[..., 3] #Podocalyxin channel
      
        # convert from 8-bit uint8 to 32-bit uint32 (expand pixel value range from 255 to 4294967295)
        image_lumen = image_lumen.astype(np.uint32)
    
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
        
        # create a list variable to store the data tables for each individual spheroid
        final_table = [0]*np.size(u_spheroid)
        
        lumen_file = selected_lumen_files[index]
        with h5py.File(lumen_file) as infile:
            print(tuple(_ for _ in infile.keys()))
            lumen_mask = infile["exported_data"]
            lumen_mask = lumen_mask[:,0,:,:]
#%%                
            print(lumen_mask.shape, image.shape, spheroid_mask.shape)
            
        lumen_mask = connected_component(lumen_mask) # segments individual lumens
        u_lumen, counts_lumen = np.unique(lumen_mask, return_counts=True)
        
        # remove pixels belonging to background class
        u_lumen = u_lumen[1:]
        counts_lumen = counts_lumen[1:]
        
        # find largest lumen corresponding to largest spheroid for further analysis
        
        for j, lumen_label in enumerate(u_lumen):
            
            # for every lumen label, find corresponding spheroid (as the spheroid with the highest number of overlapping pixels)
            sph_label, lumen_sph_overlap = np.unique(spheroid_mask[np.where(lumen_mask == lumen_label)], return_counts = True)
            if u_spheroid not in sph_label:
                lumen_mask[lumen_mask == lumen_label] = 0
        
        # update lumen list, only including lumens overlapping with largest spheroid
        u_lumen, counts_lumen = np.unique(lumen_mask, return_counts=True)
        u_lumen = u_lumen[1:]
        counts_lumen = counts_lumen[1:]

        # delete lumens in the same spheroid that are more than 'x' times smaller than largest lumen
        lumen_size_factor = 3
        
        try:
            size_filter = counts_lumen.max() >= lumen_size_factor*counts_lumen  # gives indices of lumen labels that need to be deleted, based on size filter
        except ValueError:
            continue # if there is no detected lumen in the spheroid, exclude spheroid and move on
    
 
        for idx, val in enumerate(u_lumen[size_filter]):
            lumen_mask[np.where(lumen_mask == val)] = 0 # erase lumens from the lumen mask  
        
        # update lumen arrays                        
        u_lumen = np.delete(u_lumen, size_filter)
        counts_lumen = np.delete(counts_lumen, size_filter)     
    
        channel_2_lumen_background = np.zeros(len(u_lumen))
        lumen_coords = np.zeros([len(u_lumen),3])
        for j, lumen_label in enumerate(u_lumen):
            
            channel_2_lumen_background[j] = image_channel_2[lumen_mask == lumen_label].mean() / 255         
            
            window_width_lumen = 10
            y_min, y_max, x_min, x_max, lumen_binary = roi.find(lumen_mask, lumen_label, window_width_lumen)
        
            if y_min < 0 or x_min < 0 or y_max > np.shape(lumen_mask)[1] or x_max > np.shape(lumen_mask)[2]:
                print('lumen out of range (image)')
                lumen_coords[j] = [0,0,0]
                u_lumen[j] = 0
                continue
            
            coords_lumen = np.vstack(np.where(lumen_mask == lumen_label)).T
            scaled_coords_lumen = coords_lumen * [VOX_DEPTH, VOX_HEIGHT, VOX_WIDTH]
            lumen_roi = lumen_mask[coords_lumen[0][0]:coords_lumen[-1][0],y_min:y_max,x_min:x_max]
            hull_lumen = convex_hull_image(lumen_roi) # define region of interest for convex hull calculation
            lumen_convex_volume = len(np.where(hull_lumen == True)[0]) # in pixels
            lumen_volume = np.round(counts_lumen[j]*VOX_DEPTH*VOX_WIDTH*VOX_HEIGHT,3) # in um^3
            lumen_convexity = np.round(counts_lumen[j] / lumen_convex_volume,3)
            if lumen_convexity > 1:
                lumen_convexity = 1
           
            print('lumen ' + str(lumen_label))
            lumen_surface_area = 0
            for i in range(0,np.shape(lumen_roi)[0]):
                lumen_slice = regionprops(lumen_roi[i,:,:].astype(int))
                if lumen_slice == []:
                    continue
                lumen_surface_area += lumen_slice[0].perimeter
                
            lumen_surface_area = lumen_surface_area*VOX_DEPTH*VOX_WIDTH # in um^2
           
            # determine long and short axes lengths of lumen
            pca_lumen = PCA(n_components=3)
            pca_lumen.fit(scaled_coords_lumen)
            
            # use eigenvectors to determine directions of rotated axes
            eig_1 = pca_lumen.components_[np.argmax(abs(pca_lumen.components_[:,0]))] # rotated z-axis
            eig_2 = pca_lumen.components_[np.argmax(abs(pca_lumen.components_[:,1]))] # rotated y-axis
            eig_3 = pca_lumen.components_[np.argmax(abs(pca_lumen.components_[:,2]))] # rotated x-axis
            
            rotated_scaled_coords_lumen = np.copy(scaled_coords_lumen)
            
            for idx, vec in enumerate(scaled_coords_lumen):
                rotated_scaled_coords_lumen[idx] = coordinate.rotate(eig_1, eig_2, eig_3, vec)
            
            z_dim = max(rotated_scaled_coords_lumen[:,0]) - min(rotated_scaled_coords_lumen[:,0])
            y_dim = max(rotated_scaled_coords_lumen[:,1]) - min(rotated_scaled_coords_lumen[:,1])
            x_dim = max(rotated_scaled_coords_lumen[:,2]) - min(rotated_scaled_coords_lumen[:,2])
            dim = [z_dim, y_dim, x_dim]
            
            # pca_model.components_ (rotation) and pca_model.explained_variance_ (magnitude)
            # create output files compiling relevant object data
            # add lumen data
            
            lumen_coords[j] = [coords_lumen[:,0].mean(), coords_lumen[:,1:][:,0].mean(), coords_lumen[:,1:][:,1].mean()]
            
            lumen_info = {}
            lumen_info["x-coord"] = math.ceil(lumen_coords[j][2])
            lumen_info["y-coord"] = math.ceil(lumen_coords[j][1])
            lumen_info["z-coord"] = math.ceil(lumen_coords[j][0])
            
            if spheroid_mask[lumen_info["z-coord"],lumen_info["y-coord"],lumen_info["x-coord"]] == 0 and spheroid_mask[lumen_info["z-coord"],lumen_info["y-coord"]+70,lumen_info["x-coord"]+70] == 0: # some spheroid masks have holes in the middle, need to change search space
                lumen_coords[j] = [0,0,0]
                u_lumen[j] = 0
                print('lumen out of range (spheroid)')
                continue
            elif spheroid_mask[lumen_info["z-coord"],lumen_info["y-coord"],lumen_info["x-coord"]] != 0:
                lumen_info["spheroid"] = int(spheroid_mask[lumen_info["z-coord"],lumen_info["y-coord"],lumen_info["x-coord"]])
                print('lumen in range')
                print(lumen_info["z-coord"],lumen_info["y-coord"],lumen_info["x-coord"])
            else:
                lumen_info["spheroid"] = int(spheroid_mask[lumen_info["z-coord"],lumen_info["y-coord"]+70,lumen_info["x-coord"]+70])
                print('lumen in range')
                print(lumen_info["z-coord"],lumen_info["y-coord"],lumen_info["x-coord"])
                
            lumen_info["volume (um^3)"] = float(str(lumen_volume)) # size is calculated based on the original image, since lumen object doesn't downsize correctly
            lumen_info["surface area (um^2)"] = float(str(np.round(lumen_surface_area,3))) # size is calculated based on the original image, since lumen object doesn't downsize correctly
            lumen_info["volume convexity"] = float(str(lumen_convexity)) # measure regularity of lumen surface
            lumen_info["ellipticity"] = float(str(np.round(max(dim) / min(dim), 3)))
            
            if lumen_info["volume (um^3)"] / (counts_spheroid[np.where(u_spheroid == lumen_info["spheroid"])]*VOX_DEPTH*VOX_WIDTH*VOX_HEIGHT) < 0.001:
                print('lumen too small')
                lumen_coords[j] = [0,0,0]
                u_lumen[j] = 0
                continue
            
            rows_lumen.append(lumen_info)
    
        lumen_coords = np.delete(lumen_coords, np.all(lumen_coords == 0, axis=1), axis=0)
        channel_2_lumen_background = np.delete(channel_2_lumen_background, np.where(u_lumen == 0))
        u_lumen = u_lumen[np.where(u_lumen != 0)]
        
        final_table_lumen = pd.DataFrame.from_dict(rows_lumen, orient='columns')
        final_table_lumen = final_table_lumen[1:]   
        final_table_lumen.to_csv(supplementary_directory + '\\' + file_name + '_Lumen_Data.csv')
        cum_lumens = 0

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
            
            # crop to roi
            spheroid_mask_crop = spheroid_binary[:, y_min:y_max, x_min:x_max]
            lumen_mask_crop = lumen_mask[:, y_min:y_max, x_min:x_max]
            
            u_lumen_roi, counts_lumen_roi = np.unique(lumen_mask_crop, return_counts=True)
            # remove pixels belonging to background class
            u_lumen_roi = u_lumen_roi[1:]
            counts_lumen_roi = counts_lumen_roi[1:]
            
            # pca to return the axes and orientations of spheroid
            coords_spheroid = np.vstack(np.where(spheroid_mask == spheroid_label)).T
            scaled_coords_spheroid = coords_spheroid * scale
            pre_sph_dim = np.zeros(3)
            pre_sph_dim[0] = (max(scaled_coords_spheroid[:,0]) - min(scaled_coords_spheroid[:,0])) / 2
            pre_sph_dim[1] = (max(scaled_coords_spheroid[:,1]) - min(scaled_coords_spheroid[:,1])) / 2
            pre_sph_dim[2] = (max(scaled_coords_spheroid[:,2]) - min(scaled_coords_spheroid[:,2])) / 2
            
            spheroid_roi = spheroid_mask[coords_spheroid[0][0]:coords_spheroid[-1][0],y_min:y_max,x_min:x_max]
            spheroid_volume = np.round(counts_spheroid[j]*VOX_DEPTH*VOX_WIDTH*VOX_HEIGHT,3) # in um^3
            
            spheroid_surface_area = 0
            for i in range(0,np.shape(spheroid_roi)[0]):
                spheroid_slice = regionprops(spheroid_roi[i,:,:].astype(int))
                if spheroid_slice == []:
                    continue
                spheroid_surface_area += spheroid_slice[0].perimeter
                
            spheroid_surface_area = spheroid_surface_area*VOX_DEPTH*VOX_WIDTH # in um^2
            
            pca_spheroid = PCA(n_components=3)
            pca_spheroid.fit(scaled_coords_spheroid)
            # pca_model.components_ (rotation) and pca_model.explained_variance_ (magnitude)
           
            # use eigenvectors to determine directions of rotated axes
            eig_1 = pca_spheroid.components_[np.argmax(abs(pca_spheroid.components_[:,0]))] # rotated z-axis
            eig_2 = pca_spheroid.components_[np.argmax(abs(pca_spheroid.components_[:,1]))] # rotated y-axis
            eig_3 = pca_spheroid.components_[np.argmax(abs(pca_spheroid.components_[:,2]))] # rotated x-axis
            
            rotated_scaled_coords_spheroid = np.copy(scaled_coords_spheroid)
            
            for idx, vec in enumerate(scaled_coords_spheroid):
                rotated_scaled_coords_spheroid[idx] = coordinate.rotate(eig_1, eig_2, eig_3, vec)
                
            sph_dim = np.zeros(3)
            sph_dim[0] = (max(rotated_scaled_coords_spheroid[:,0]) - min(rotated_scaled_coords_spheroid[:,0])) / 2
            sph_dim[1] = (max(rotated_scaled_coords_spheroid[:,1]) - min(rotated_scaled_coords_spheroid[:,1])) / 2
            sph_dim[2] = (max(rotated_scaled_coords_spheroid[:,2]) - min(rotated_scaled_coords_spheroid[:,2])) / 2
            
            # add spheroid data
            
            spheroid_info = {}
            spheroid_info["spheroid"] = int(spheroid_label)
            spheroid_info["x-coord"] = math.ceil(sph_x_c)
            spheroid_info["y-coord"] = math.ceil(sph_y_c)
            spheroid_info["z-coord"] = math.ceil(sph_z_c)
            spheroid_info["volume (um^3)"] = float(str(spheroid_volume)) # size is calculated based on the original image
            spheroid_info["surface area (um^2)"] = float(str(np.round(spheroid_surface_area,3)))
            spheroid_info["ellipticity"] = float(str(np.round(max(sph_dim*2) / min(sph_dim*2), 3)))
            
            lumens_idx = np.where(final_table_lumen.spheroid == spheroid_label)[0] # identify lumens belonging to this spheroid
            
            spheroid_info["lumen eccentricity"] = []
            spheroid_info["lumen-spheroid ratio"] = []
            
            for c, idx in enumerate(lumens_idx):
                
                x_eccentricity = (final_table_lumen.iloc[idx][1] - spheroid_info["x-coord"])*VOX_WIDTH
                y_eccentricity = (final_table_lumen.iloc[idx][2] - spheroid_info["y-coord"])*VOX_HEIGHT
                z_eccentricity = (final_table_lumen.iloc[idx][3] - spheroid_info["z-coord"])*VOX_DEPTH
                lumen_eccentricity = coordinate.rotate(eig_1, eig_2, eig_3, [z_eccentricity, y_eccentricity, x_eccentricity])
                
                if abs(np.linalg.norm(lumen_eccentricity / sph_dim)) > 0.75:
                    lumen_coords[idx] = [0,0,0]
                    u_lumen[idx] = 0
                    lumens_idx[c] = -10
                    continue
                    
                else:  
                    spheroid_info["lumen eccentricity"].append(np.round(abs(np.linalg.norm(lumen_eccentricity / sph_dim)),3))
                    spheroid_info["lumen-spheroid ratio"].append(np.round(final_table_lumen.iloc[idx][4] / spheroid_info["volume (um^3)"],4))
                
            lumen_coords = np.delete(lumen_coords, np.all(lumen_coords == 0, axis=1), axis=0)
            u_lumen = u_lumen[np.where(u_lumen != 0)]
            spheroid_info["number of lumens"] = len(lumens_idx[lumens_idx != -10])
            num_lumen = spheroid_info["number of lumens"]
            cum_lumens += spheroid_info["number of lumens"]
#%%            
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
            
            # make sure anti-aliasing is 'False'
            spheroid_mask_crop_down = resize(spheroid_mask_crop, (spheroid_mask_crop.shape[0], 
                                    spheroid_mask_crop.shape[1] // resize_factor, 
                                    spheroid_mask_crop.shape[2] // resize_factor),
                                    anti_aliasing=False) * 255
            
            # resize lumen() to fit image
            lumen_mask_crop_down = resize(lumen_mask_crop, (lumen_mask_crop.shape[0], 
                                    lumen_mask_crop.shape[1] // resize_factor, 
                                    lumen_mask_crop.shape[2] // resize_factor),
                                    anti_aliasing=False) * 255
            
            # find all non-background, downsized connected components in the spheroid and lumen masks
            if all(spheroid_mask_crop_down[0].flatten() == 0) == True:
                spheroid_mask_crop_down = connected_component(spheroid_mask_crop_down != spheroid_mask_crop_down[0]) #taking the first Z-plane as a reference for background
            
            else:     
                spheroid_mask_crop_down = connected_component(spheroid_mask_crop_down != spheroid_mask_crop_down[0,0,0])
                u_sph_mask_crop_down, counts_sph_mask_crop_down = np.unique(spheroid_mask_crop_down, return_counts=True)
                u_sph_mask_crop_down = u_sph_mask_crop_down[1:]
                counts_sph_mask_crop_down = counts_sph_mask_crop_down[1:]
                spheroid_mask_crop_down = spheroid_mask_crop_down == u_sph_mask_crop_down[np.argmax(counts_sph_mask_crop_down)]
            
            lumen_mask_crop_down = connected_component(lumen_mask_crop_down != lumen_mask_crop_down[0,0,0]) #taking the first Z-plane as a reference for background
            
            plane_tip_1 = min(np.where(lumen_mask_crop_down == 1)[1])
            plane_tip_2 = max(np.where(lumen_mask_crop_down == 1)[1])
            
            region_mask = np.zeros(spheroid_mask_crop_down.shape)
            
            region_mask[:, plane_tip_2:, :] = 1 # assign label to tip regions
            region_mask[:, 0:plane_tip_1, :] = 1
            region_mask[:, plane_tip_1:plane_tip_2, :] = 2 # assign label to side regions
            
            region_spheroid_mask = region_mask * spheroid_mask_crop_down
#%%            
            # normalize DAPI image for neural network
            img = normalize(image_crop_down, 1,99.8, axis=(0,1,2))
            
            # predict segmentation
            labels, details = model.predict_instances(img, prob_thresh=0.7, nms_thresh=0.3)
            
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
            # viewer.add_labels(lumen_mask_crop_down, blending = 'additive', scale=(2,1,1))
            # viewer.add_labels(labels, blending = 'translucent', scale=(2,1,1))
            
            # background subtraction and planar DAPI normalisation   
            planar_mean_DAPI = np.zeros(np.shape(image_crop_down)[0])
            img_2_background_sub = np.zeros(np.shape(image_channel_2_crop_down))
            img_2_planar_DAPI_normalised = np.zeros(np.shape(image_channel_2_crop_down))
            
            background_array = image_channel_2_crop_down[np.where(((spheroid_mask_crop_down != 0)*1 & (labels == 0)*1) == 1)]
            background = np.percentile(background_array,75) # take 75th-percentile of non-nuclear spheroid pixel intensities as background
            img_2_background_sub = (image_channel_2_crop_down - background).clip(min=0) # eliminate negative values
            
            for z in range(0,np.shape(img)[0]):
                # background subtraction
                try:
                    background_array = image_channel_2_crop_down[z,:,:][np.where(((spheroid_mask_crop_down[z,:,:] != 0)*1 & (labels[z,:,:] == 0)*1) == 1)]
                    background = np.percentile(background_array,75) # take 75th-percentile of non-nuclear spheroid pixel intensities as background
                    print(background)
                    img_2_background_sub[z,:,:] = (image_channel_2_crop_down[z,:,:] - background).clip(min=0) # eliminate negative values
                
                except IndexError:
                    continue
                
                planar_mean_DAPI[z] = image_crop_down[z,:,:][np.where(labels[z,:,:] != 0)].mean()
                img_2_planar_DAPI_normalised[z] = img_2_background_sub[z,:,:] / planar_mean_DAPI[z] # normalised by planar DAPI intensity to account for variations in Z position
            
            skimage.io.imsave(label_directory + '\\Crop_DN_' + file_name + '_Spheroid_' + str(spheroid_label) + '.tif', img_2_planar_DAPI_normalised)
            
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
            x_c = np.zeros(len(np.unique(labels)[1:])).astype(int)
            y_c = np.zeros(len(np.unique(labels)[1:])).astype(int)
            z_c = np.zeros(len(np.unique(labels)[1:])).astype(int)
            
            for idx, label in enumerate(np.unique(labels)[1:]):
                
                # exclude nuclei outside the spheroid, based on overlap with spheroid mask
                if all(spheroid_mask_crop_down[np.where(labels==label)] == 0) == True:
                    continue
                
                # exclude nuclei with very low DAPI intensity
                if image_crop_down[labels==label].mean() < 0.15:
                    continue
                
                x_c[idx] = int(round(xgrid[labels==label].mean()))
                y_c[idx] = int(round(ygrid[labels==label].mean()))
                z_c[idx] = int(round(zgrid[labels==label].mean()))
                
                mean_DAPI[idx] = float(str(np.round(image_crop_down[labels==label].mean(),3)))
                
                channel_2_raw[idx] = image_channel_2_crop_down[labels==label].mean()
                channel_2_raw_75[idx] = np.percentile(image_channel_2_crop_down[labels==label],75)
                channel_2_bs[idx] = img_2_background_sub[labels==label].mean()
                channel_2_bs_75[idx] = np.percentile(img_2_background_sub[labels==label],75)
                channel_2[idx] = img_2_planar_DAPI_normalised[labels==label].mean() 
                channel_2_75[idx] = np.percentile(img_2_planar_DAPI_normalised[labels==label],75)
            
                tip_nuclei = 0
                side_nuclei = 0
            
            for idx, label in enumerate(np.unique(labels)[1:]):

                #skip nuclei that were excluded in the previous loop
                if x_c[idx] == 0:
                    continue
               
                nucleus_count += 1
                
                nuc_coord = np.array([zgrid[labels==label].mean(), (ygrid[labels==label].mean()*resize_factor) + y_min, (xgrid[labels==label].mean()*resize_factor) + x_min])
                
                if num_lumen == 0: # no lumen detected
                    nearest_lumen = 0
                    nearest_lum_nuc_vec = nuc_coord*scale - scaled_sph_centroid
                    lum_nuc_dist = abs(np.linalg.norm(nearest_lum_nuc_vec))
                
                else:
                    lum_nuc_vec = []
                    lum_nuc_dist = np.zeros(len(lumen_coords))
                    
                    for k, lumen_coord in enumerate(lumen_coords):
                        lum_nuc_vec.append((nuc_coord - lumen_coord)*scale)
                        lum_nuc_dist[k] = abs(np.linalg.norm(lum_nuc_vec[k]))
                    
                    nearest_lumen = u_lumen[np.argmin(lum_nuc_dist)]
                    nearest_lum_nuc_vec = lum_nuc_vec[np.argmin(lum_nuc_dist)]
                    
                nearest_lum_nuc_vec = coordinate.rotate(eig_1, eig_2, eig_3, nearest_lum_nuc_vec) # using rotated coordinates 
                
                nucleus_info = {}
                nucleus_info["label"] = int(label)
                nucleus_info["x-coord"] = math.ceil(nuc_coord[2])
                nucleus_info["y-coord"] = math.ceil(nuc_coord[1])
                nucleus_info["z-coord"] = math.ceil(nuc_coord[0])
                nucleus_info["volume (um^3)"] = float(str(np.round((labels==label).sum()*VOX_DEPTH*resize_factor*VOX_WIDTH*resize_factor*VOX_HEIGHT,3)))
                nucleus_info["DAPI intensity"] = mean_DAPI[idx]
                nucleus_info["raw channel 2 intensity"] = float(str(np.round(channel_2_raw[idx],3)))
                nucleus_info["raw channel 2 intensity"] = float(str(np.round(channel_2_raw_75[idx],3)))
                nucleus_info["BS channel 2 intensity (mean)"] = float(str(np.round(channel_2_bs[idx],3)))
                nucleus_info["BS channel 2 intensity (75th-percentile)"] = float(str(np.round(channel_2_bs_75[idx],3)))
                nucleus_info["BSDN channel 2 intensity (mean)"] = float(str(np.round(channel_2[idx],3)))
                nucleus_info["BSDN channel 2 intensity (75th-percentile)"] = float(str(np.round(channel_2_75[idx],3)))
                nucleus_info["mean-normalised channel 2 intensity"] = ''
                nucleus_info["channel 2 expression"] = ''
                
                u_region, counts_region = np.unique(region_spheroid_mask[np.where(labels == label)], return_counts=True) # find spheroid regions overlapping with nucleus
                
                if 0 in u_region:
                    counts_region = np.delete(counts_region, np.where(u_region == 0)[0])
                    u_region = np.delete(u_region, np.where(u_region == 0)[0])
                
                if u_region[counts_region.argmax()] == 1:
                    tip_nuclei += 1
                    nucleus_info["region"] = 'tip'
                    
                elif u_region[counts_region.argmax()] == 2:
                    side_nuclei += 1
                    nucleus_info["region"] = 'side'
                
                else:
                    nucleus_info["region"] = 'undefined'
                rows_nuclei.append(nucleus_info)
            
            if nucleus_count == 0:
                continue
            
            final_table[j] = pd.DataFrame.from_dict(rows_nuclei, orient='columns')
            final_table[j] = final_table[j] # should exclude background label (label 0)
            final_table[j].to_csv(supplementary_directory + '\\' + file_name + '_Spheroid_' + str(spheroid_label) + '_Nucleus_Data.csv')
                
            spheroid_info["number of cells"] = nucleus_count # should exclude background label (label 0)
            spheroid_info["# +ve cells"] = positive_count
            spheroid_info["% +ve cells"] = float(str(np.round((spheroid_info["# +ve cells"] / nucleus_count) * 100,2)))
            spheroid_info["tip region volume (um^3)"] = float(str(np.round((region_spheroid_mask==1).sum()*VOX_DEPTH*resize_factor*VOX_WIDTH*resize_factor*VOX_HEIGHT,3)))
            spheroid_info["side region volume (um^3)"] = float(str(np.round((region_spheroid_mask==2).sum()*VOX_DEPTH*resize_factor*VOX_WIDTH*resize_factor*VOX_HEIGHT,3)))
            spheroid_info["local density - tip (cells/mm^3)"] = float(str(np.round((tip_nuclei / spheroid_info["tip region volume (um^3)"])*10**6,3)))
            spheroid_info["local density - side (cells/mm^3)"] = float(str(np.round((side_nuclei / spheroid_info["side region volume (um^3)"])*10**6,3)))
            
            rows_spheroid.append(spheroid_info)
        
        print(len(lumen_coords) == len(u_lumen) == cum_lumens)
        final_table_spheroid = pd.DataFrame.from_dict(rows_spheroid, orient='columns')
        final_table_spheroid = final_table_spheroid[1:]
        final_table_spheroid.to_csv(output_directory + file_name + '_Spheroid_Data' + '.csv')