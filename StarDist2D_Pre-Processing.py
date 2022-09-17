"""
PRE-PROCESSES IMAGES BEFORE RUNNING 'StarDist2D.py'
GENERATES HDF5 NUCLEAR 'LABEL' FILES AND CROPPED SPHEROID TIF FILES AS INPUTS FOR ILASTIK CLASSIFIER (POSITIVE/NEGATIVE)
GENERATES MULTICUT SEGMENTATION MASK

@author: Aly Makhlouf
"""
import os
import tensorflow as tf
import skimage.io
import numpy as np
import matplotlib
import glob
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import cv2
import gc

from stardist.models import StarDist2D
from stardist import random_label_cmap

import tqdm
from itertools import chain
from tifffile import imread
from csbdeep.utils import normalize
from skimage.transform import resize
from skimage.measure import label as connected_component
import h5py

# custom modules
import geometry
import mask_generate

# close all hdf5/h5 files
for obj in gc.get_objects():   # Browse through ALL objects
    if isinstance(obj, h5py.File):   # Just HDF5 files
        try:
            obj.close()
        except:
            pass # Was already closed

# FILE IMPORTS AND PRE-PROCESSING

# import model
model = StarDist2D.from_pretrained('2D_versatile_fluo') #2D_versatile_fluo, 2D_demo, 2D_paper_dsb2018
    
# import image
folder_name = 'with DOX'

path = 'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\' + folder_name + '\\'
files = glob.glob(path + '/*.tif')
files_list = list(files)

processed_path = 'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\' + folder_name + '\\Processed\\'
processed_files = glob.glob(path + '/*.tif')
processed_files_list = list(processed_files)

label_directory = 'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\Labels_' + folder_name + '\\'
if not os.path.exists(label_directory):
    os.makedirs(label_directory)
#%%
#for index, file in enumerate(files[0:]):
for file in tqdm.tqdm(list(chain(files_list[0:]))):
    print(file)
    image = skimage.io.imread(file)[0] # Channel 1 (DAPI)
    image_channel_2 = skimage.io.imread(file)[1] # Channel 2 (Brachyury)
    image_channel_3 = skimage.io.imread(file)[2] # Channel 3 (E-Cadherin)
    image_channel_4 = skimage.io.imread(file)[3] # Channel 4 (TagRFP-aPKCi)
    image_channel_5 = skimage.io.imread(file)[4] # Channel 5 (Podxl)
    
    start = file.find(folder_name + '\\') + len(folder_name + '\\')
    end = file.find('.tif', start)
    file_name = file[start:end]
    print(file_name)

    image_processed = skimage.io.imread(processed_path + file_name + '.tif')[0] # Channel 1 (DAPI)
    contrast = 1
    brightness = 0
    thresh = 0
    image_processed = (image_processed*contrast + brightness).clip(min=brightness + thresh) # enhance contrast and brightness in DAPI channel
    
    # import spheroid mask
    spheroid_file =  'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\' + folder_name + '\\Spheroid\\Object Predictions\\' + file_name + '_Object Predictions.h5'
    #spheroid_file =  'D:\\User Data\\Marta\\3D Spheroids\\' + experiment + '\\Spheroids\\Object Predictions\\' + experiment + '_serie_' + n_ser.lstrip('0') + ' - Series0' + n_ser +'_Object Predictions.h5'
    with h5py.File(spheroid_file) as infile:
        print(tuple(_ for _ in infile.keys()))
        spheroid_pred = infile["exported_data"]
        spheroid_pred = spheroid_pred[0,:,:] # ilastik object classifier gives all spheroids a common label (1)
      
    # find all spheroids in the spheroid channel
    spheroid_mask = connected_component(spheroid_pred) # convert to an array [y,x] with a unique label for each object
    u_spheroid, counts_spheroid = np.unique(spheroid_mask, return_counts=True)
    counts_spheroid = counts_spheroid[1:]
    u_spheroid = u_spheroid[1:]
    u_spheroid = u_spheroid[np.where(counts_spheroid == max(counts_spheroid))] # only select the largest spheroid

    image[np.where(spheroid_mask != u_spheroid)] = 0 # erase all other objects from original image
    image_processed[np.where(spheroid_mask != u_spheroid)] = 0 # erase all other objects from original image
    roundness_spheroid = round(geometry.compute_roundness(spheroid_mask),3)
    
    # import lumen mask
    lumen_file =  'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\' + folder_name + '\\Lumen\\Object Predictions\\' + file_name + '_Object Predictions.h5'    
    with h5py.File(lumen_file) as infile:
        print(tuple(_ for _ in infile.keys()))
        lumen_mask = infile["exported_data"]
        lumen_mask = lumen_mask[0,:,:]
  
    lumen_mask = connected_component(lumen_mask) # segments individual lumens
    u_lumen, counts_lumen = np.unique(lumen_mask, return_counts=True)
    u_lumen = u_lumen[1:]
    counts_lumen = counts_lumen[1:]

    # import MultiCut segmentation
    segment_file =  'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\' + folder_name + '\\Segment\\Multicut Segmentation\\' + file_name + '_Multicut Segmentation.h5'
    with h5py.File(segment_file) as infile:
        print(tuple(_ for _ in infile.keys()))
        segment_mask = infile["exported_data"]
        segment_mask = segment_mask[0,:,:]
        segment_mask_0 = segment_mask.copy() # copy of segment mask that retains background and lumen labels
        segment_mask[segment_mask == np.unique(segment_mask)[np.argmax(np.unique(segment_mask, return_counts = True)[1])]] = 0 # convert background segment pixels to 0
        segment_mask_1 = segment_mask.copy() # copy of segment mask that retains lumen labels
        #segment_mask[np.where(spheroid_mask != u_spheroid)] = 0 # only select the largest spheroid

        # clean up spheroid mask
        spheroid_background = connected_component(spheroid_mask == 0)
        spheroid_background_segments = np.unique(spheroid_background)[2:]
        background_int = np.zeros(len(spheroid_background_segments))
        background_size = np.zeros(len(spheroid_background_segments))
        for idx, segment in enumerate(np.unique(spheroid_background_segments)):
            background_int[idx] = np.mean(image_channel_3[np.where(spheroid_background == segment)])
            background_size[idx] = (spheroid_background == segment).sum()
            if background_int[idx] >= 10 and background_size[idx] >= 10000:
                spheroid_mask[spheroid_background == segment] = u_spheroid
            
        # clean up mis-segmented cell segments and make them compatible with spheroid mask
        overlap = (segment_mask != 0)*1 & (spheroid_mask !=0)*1 # find overlap between segment_mask and spheroid_mask
        
        roundness_segment = np.zeros(len(np.unique(segment_mask)))
        for idx, segment in enumerate(np.unique(segment_mask)):
            if (len(np.where(overlap[np.where(segment_mask == segment)] == 0)[0]) >= 0.5*len(np.where(segment_mask == segment)[0])):
                print(segment)
                roundness_segment[idx] = round(geometry.compute_roundness(segment_mask == segment),3)
                segment_mask[np.where(segment_mask == segment)] = 0

        # find spheroid segments
        segment_mask_spheroid = segment_mask != 0
        segment_mask_spheroid = connected_component(segment_mask_spheroid) # convert to an array [y,x] with a unique label for each object
        u_segment_spheroid, counts_segment_spheroid = np.unique(segment_mask_spheroid, return_counts=True)
        u_segment_spheroid = u_segment_spheroid[1:]
        counts_segment_spheroid = counts_segment_spheroid[1:]
        u_segment_spheroid = u_segment_spheroid[np.where(counts_segment_spheroid == max(counts_segment_spheroid))] # only select the largest spheroid
        counts_segment_spheroid = np.max(counts_segment_spheroid)
        segment_mask_spheroid[np.where(segment_mask_spheroid != u_segment_spheroid)] = 0 # erase all other objects from original image
        segment_mask = segment_mask_1 * segment_mask_spheroid

        if len(counts_lumen) != 0:
            segment_lumen = np.unique(segment_mask_1[np.where(lumen_mask != 0)])[np.argmax(np.unique(segment_mask_1[np.where(lumen_mask != 0)], return_counts = True)[1])] # identify lumen segment
            segment_mask[segment_mask_1 == segment_lumen] = 0 # convert lumen segment pixels to 0
            
        else:
            segment_lumen = 10000 # arbitrarily high number, since there is no lumen
            
        # apply edge filter to segment mask
        cell_edges = skimage.feature.canny(image=segment_mask, sigma=0, low_threshold=0, high_threshold=0)


    # find unique cell segments
    u_segment, size_segment = np.unique(segment_mask[np.where(segment_mask_spheroid != 0)], return_counts = True)
    u_segment = u_segment[1:]
    size_segment = size_segment[1:]
    
    # find lumen segments
    segment_mask_lumen = segment_mask_0 == segment_lumen
    segment_mask_lumen_size = len(np.where(segment_mask_lumen == True)[0])
    
    
    # check compatibility of ilastik-segmented lumen and multicut-segmented lumen
    
    if abs((segment_mask_lumen_size - np.max(counts_lumen)) / np.max(counts_lumen)) > 0.5:
        lumen_mask = segment_mask_lumen # revert to multicut-segmented lumen as ground truth
#%%    
    # IMAGE PROCESSING #
    
    PIX_WIDTH = 1
    PIX_HEIGHT = 1
    
    resize_factor = 1/PIX_WIDTH # or PIX_HEIGHT, based on scaled dimensions of pre-trained model (1,1)
    image_down = resize(image, (image.shape[0]//resize_factor, image.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    image_channel_2_down = resize(image_channel_2, (image_channel_2.shape[0]//resize_factor, image_channel_2.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    # make sure anti-aliasing is 'False'
    spheroid_mask_down = resize(spheroid_mask, (spheroid_mask.shape[0] // resize_factor, 
                            spheroid_mask.shape[1] // resize_factor),
                            anti_aliasing=False) * 255
    
    lumen_mask_down = resize(lumen_mask, (lumen_mask.shape[0] // resize_factor, 
                                lumen_mask.shape[1] // resize_factor),
                                anti_aliasing=False) * 255
    
    image_processed_down = resize(image_processed, (image_processed.shape[0]//resize_factor, 
                                image_processed.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    # find all non-background, downsized connected components in the spheroid and lumen masks
    if all(spheroid_mask_down[0].flatten() == 0) == True:
        spheroid_mask_down = connected_component(spheroid_mask_down != spheroid_mask_down[0])
    
    else:     
        spheroid_mask_down = connected_component(spheroid_mask_down != spheroid_mask_down[0,0])
        u_sph_mask_down, counts_sph_mask_down = np.unique(spheroid_mask_down, return_counts=True)
        u_sph_mask_down = u_sph_mask_down[1:]
        counts_sph_mask_down = counts_sph_mask_down[1:]
        spheroid_mask_down = spheroid_mask_down == u_sph_mask_down[np.argmax(counts_sph_mask_down)]
    
    lumen_mask_down = connected_component(lumen_mask_down != lumen_mask_down[0])
    u_lumen_down, counts_lumen_down = np.unique(lumen_mask_down, return_counts=True)
    lumen_mask_down = (lumen_mask_down != 0)*1 # use in case there are multiple lumen objects
    
    # remove pixels belonging to background class
    u_lumen_down = u_lumen_down[1:]
    counts_lumen_down = counts_lumen_down[1:] 
    
    # normalise down-sized image in DAPI channel
    img = normalize(image_processed_down, 1,99.8, axis=(0,1))
    img_channel_2 = image_channel_2_down
    
    # NUCLEAR SEGMENTATION, PROCESSING AND VIEWING #
    
    # exclude tiny (mis-segmented) nuclei (< 1000 pixels) and processing artefacts from (adjusting brightness/contrast)
    labels, details = model.predict_instances(img, prob_thresh=0.45, nms_thresh=0.5)
    all_labels = np.unique(labels)

    nuc_int_array = np.zeros(len(all_labels))
    for idx, label in enumerate(all_labels):
        if len(np.where(labels == label)[0]) < 1000:
            labels[np.where(labels == label)] = 0
        
        else:
            nuc_int_array[idx] = np.mean(image_down[np.where(labels == label)])
            print(nuc_int_array[idx])    
        
        if np.mean(image_down[np.where(labels == label)]) < 30:
            labels[np.where(labels == label)] = 0
#%%    
    # export labels
    labels_export = labels.astype(np.uint32)
    labels_file = h5py.File(label_directory + 'Labels_' + file_name + '.hdf5', "w")
    labels_dset = labels_file.create_dataset("dataset", data = labels_export)
    labels_file.close()
    
    labels_up = resize(labels, (1024,1024), anti_aliasing = True) # resize labels to original image
    nuc_labels = np.unique(labels)[1:]
    
    # apply edge filter to DAPI channel
    nuclei_edges = skimage.feature.canny(image=labels, sigma=0.3, 
                                       low_threshold=0.8, high_threshold=0.9)
    nuclei_edges = resize(nuclei_edges, (1024,1024), anti_aliasing = False) # resize to original image
    
    lbl_cmap = random_label_cmap()
    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(lumen_mask_down, cmap=lbl_cmap, alpha=0.5)
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.axis('off');
    
    # create coordinate system for original image
    x_0 = np.arange(0,image.shape[1])
    y_0 = np.arange(0,image.shape[0])
    
    ygrid_0, xgrid_0 = np.meshgrid(y_0,x_0, indexing='ij')
    
    # create coordinate system for down-sized image
    x = np.arange(0,img.shape[1])
    y = np.arange(0,img.shape[0])
    
    ygrid, xgrid = np.meshgrid(y,x, indexing='ij')
    
    # background subtraction and planar DAPI normalisation
    
    try:
        background_array = image_channel_2_down[np.where(((spheroid_mask_down != 0)*1 & (labels == 0)*1) == 1)]
        background = np.percentile(background_array,75) # take 75th-percentile of non-nuclear spheroid pixel intensities as background
        print(background)
        img_2_background_sub = (image_channel_2_down - background).clip(min=0) # eliminate negative values
    
    except IndexError:
        continue
    
    planar_mean_DAPI = image_down[np.where(labels != 0)].mean()
    img_2_planar_DAPI_normalised = img_2_background_sub / planar_mean_DAPI # normalised by planar DAPI intensity to account for variations in Z position
    
    # export cropped spheroid image
    skimage.io.imsave(label_directory + 'Crop_DN_' + file_name + '.tif', img_2_planar_DAPI_normalised)
        
    # generate and save segment mask
    mask_generate.save(segment_mask, file_name, 'Mask_Segment', path)