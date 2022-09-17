# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:20:59 2021

@author: Aly Makhlouf
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import os
import glob
import math
import sys
import tensorflow as tf
import skimage.io
import skimage.viewer
import numpy as np
import statistics as stat
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import cv2
import array as arr
import napari

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize, download_and_extract_zip_file
from csbdeep.io import save_tiff_imagej_compatible
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import label as connected_component
from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
import h5py
import pandas as pd


# FILE IMPORTS AND PRE-PROCESSING

# import model
model = StarDist2D.from_pretrained('2D_versatile_fluo') #2D_versatile_fluo, 2D_demo, 2D_paper_dsb2018
    
# import image
#folder_name = '2D EpiSC medium'
folder_name = '2D spheroid medium'

#sub_folder_name = 'with dox'
sub_folder_name = 'without dox'

sub_sub_folder_name = '2021-Jul-1'

#sub_sub_folder_name = '2021-Jun-29'
#sub_sub_folder_name = '2021-Jun-3'

#sub_sub_folder_name = '2021-May-31'

# SET UP OUTPUT DIRECTORY 
output_directory = 'D:\\User Data\\Aly\\Nanami\\' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
label_directory = 'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 2D culture for quantification\\Labels_' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\'
if not os.path.exists(label_directory):
    os.makedirs(label_directory)

# SPECIFY INPUT DIRECTORY
directory = 'D:/User Data/Aly//Nanami/TagRFP-aPKCi 2D culture for quantification/' + folder_name + '/' + sub_folder_name + '/' + sub_sub_folder_name
files = glob.glob(directory + '/*.tif')

# LIST CLASSIFIER FILES
classifier_files = glob.glob(label_directory + '/*.h5')
#%%

DAPI = []
Brachyury = []
norm_Brachyury = []

for file in files[0:]:
    
    sub_sub_folder_0 = sub_sub_folder_name + '\\'
    start = file.find(sub_sub_folder_0) + len(sub_sub_folder_0)
    end = file.find('.tif', start)
    file_name = file[start:end]
    print(file_name)
    
    try:
        image = skimage.io.imread('D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 2D culture for quantification\\' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\' + file_name + '.tif')[0] # Channel 1 (DAPI)
    except FileNotFoundError:
        continue
    
    image_channel_2 = skimage.io.imread('D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 2D culture for quantification\\' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\' + file_name + '.tif')[1] # Channel 2 (Brachyury)
    image_channel_3 = skimage.io.imread('D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 2D culture for quantification\\' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\' + file_name + '.tif')[2] # Channel 3 (E-Cadherin)
    image_channel_4 = skimage.io.imread('D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 2D culture for quantification\\' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\' + file_name + '.tif')[3] # Channel 4 (TagRFP-aPKCi)
    image_channel_5 = skimage.io.imread('D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 2D culture for quantification\\' + folder_name + '\\' + sub_folder_name + '\\' + sub_sub_folder_name + '\\' + file_name + '.tif')[4] # Channel 5 (Podxl)

    # IMAGE PROCESSING #
    
    PIX_WIDTH = 0.3787879
    PIX_HEIGHT = 0.3787879
    
    resize_factor = 1/PIX_WIDTH # or PIX_HEIGHT, based on scaled dimensions of pre-trained model (1,1)
    image_down = resize(image, (image.shape[0]//resize_factor, image.shape[1]//resize_factor),
                                anti_aliasing=True) * 255
    
    # normalise down-sized image in DAPI channel for neural network
    img = normalize(image_down, 1,99.8, axis=(0,1))
    
    image_channel_2 = gaussian_filter(image_channel_2, sigma=2)
    img_channel_2 = resize(image_channel_2, (image_channel_2.shape[0]//resize_factor, image_channel_2.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    image_channel_4 = gaussian_filter(image_channel_4, sigma=2) # smooth out saturated spots that can skew signal 
    img_channel_4 = resize(image_channel_4, (image_channel_4.shape[0]//resize_factor, image_channel_4.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    # NUCLEAR SEGMENTATION, PROCESSING AND VIEWING #
    
    labels, details = model.predict_instances(img, prob_thresh=0.5, nms_thresh=0.4)
    labels_up = resize(labels, (1024,1024), anti_aliasing = True) # resize labels to original image

    # export labels
    labels_export = labels.astype(np.uint32)
    labels_file = h5py.File(label_directory + 'Labels_' + file_name + '.hdf5', "w")
    labels_dset = labels_file.create_dataset("dataset", data = labels_export)
    labels_file.close()

    # apply edge filter to DAPI channel
    nuclei_edges = skimage.feature.canny(image=labels, sigma=0.3, 
                                       low_threshold=0.8, high_threshold=0.9)
    nuclei_edges = resize(nuclei_edges, (1024,1024), anti_aliasing = False) # resize edges to original image
    
    lbl_cmap = random_label_cmap()
    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
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
    
    
    nuclei = np.unique(labels)[1:]
#%%    
    
    # DEFINE VARIABLES FOR TABULATING DATA #

    rows_cells = []
    
    # define remaining analysis variables
    nuc_area = np.ones(len(nuclei)) * -10
    nuc_rad = np.ones(len(nuclei)) * -10
    mean_DAPI = np.ones(len(nuclei)) * -10
    mean_channel_2 = np.ones(len(nuclei)) * -10
    mean_channel_2_nuc_bs = np.ones(len(nuclei)) * -10
    roundness_nuc = np.ones(len(nuclei)) * -10
    
    # define variables for compiling radial vector data
    tag_rfp_left = [ [] for i in range(len(nuclei))]
    tag_rfp_right = [ [] for i in range(len(nuclei))]
    tag_rfp_up = [ [] for i in range(len(nuclei))]
    tag_rfp_down = [ [] for i in range(len(nuclei))]
    
    background = np.percentile(img_channel_2[np.where(labels == 0)],75) # take background as 75th-percentile of non-nuclear brachyury intensity
    img_channel_2_bs = (img_channel_2 - background).clip(min=0) # background-subtracted image
    background_nuc = img_channel_2_bs[np.where(labels != 0)].mean() # mean nuclear channel 2 pixel intensity
    img_channel_2_bs_norm = (img_channel_2_bs - background_nuc).clip(min=0)
    
    # export background-subtracted, channel 2-normalised monolayer image
    skimage.io.imsave(label_directory + 'BS_' + file_name + '.tif', img_channel_2_bs_norm)
    
    classifier_file = [match for match in classifier_files if file_name in match][-1] # find classifier file with matching name
    
    with h5py.File(classifier_file) as infile:
                print(tuple(_ for _ in infile.keys()))
                nuc_classifier = infile["exported_data"]
                nuc_classifier = nuc_classifier[:,:,0] # ilastik object classifier gives positive and negative channel 2 nuclei
    
    for idx, value in enumerate(nuclei):
    
        if img[labels==value].mean() < 0.15:
            nuclei[idx] = -10
            continue
        
        mean_DAPI[idx] = image_down[labels==value].mean()
        mean_channel_2[idx] = img_channel_2_bs[labels==value].mean() 
        mean_channel_2_nuc_bs[idx] = img_channel_2_bs_norm[labels==value].mean()
    
    
    nuclei = nuclei[nuclei != -10]
    mean_DAPI = mean_DAPI[mean_DAPI != -10]
    mean_channel_2 = mean_channel_2[mean_channel_2 != -10]
    mean_channel_2_nuc_bs = mean_channel_2_nuc_bs[mean_channel_2_nuc_bs != -10]

    DAPI_norm = image_down[np.where(labels != 0)].mean() # will be used to normalise aPKCi and brachyury signals
    mean_norm_channel_2 = mean_channel_2_nuc_bs / DAPI_norm # normalise channel 2 intensities by mean DAPI

    # define radial vectors from every nucleus, in both vertical and horizontal directions, in multiple channels
    nucleus_counter = 0
    search = 5 # specify depth of search field (downsized pixels) 
    
    x_min = np.zeros(len(nuclei)).astype(int)
    x_max = np.zeros(len(nuclei)).astype(int)
    y_min = np.zeros(len(nuclei)).astype(int)
    y_max = np.zeros(len(nuclei)).astype(int)
    
    for idx, value in enumerate(nuclei):
        
        print(str(idx) + ',' + str(value))

        x_min[idx] = min(xgrid[labels==value])
        x_max[idx] = max(xgrid[labels==value])
        y_min[idx] = min(ygrid[labels==value])
        y_max[idx] = max(ygrid[labels==value])
        
        # exclude nuclei at the boundaries
        try:
            tag_rfp_left[idx] = img_channel_4[ygrid[labels==value][np.argmin(xgrid[labels == value])], range(x_min[idx]-search,x_min[idx]+1)]
        except IndexError:
            continue
        try:
            tag_rfp_right[idx] = img_channel_4[ygrid[labels==value][np.argmax(xgrid[labels == value])], range(x_max[idx],x_max[idx]+search+1)]
        except IndexError:
            continue
        try:
            tag_rfp_up[idx] = img_channel_4[range(y_min[idx]-search,y_min[idx]+1), xgrid[labels==value][np.argmin(ygrid[labels == value])]]
        except IndexError:
            continue
        try:
            tag_rfp_down[idx] = img_channel_4[range(y_max[idx],y_max[idx]+search+1), xgrid[labels==value][np.argmax(ygrid[labels == value])]]
        except IndexError:
            continue
        
        nucleus_counter += 1
        nuc_area[idx] = ((labels==value).sum()) * (1/PIX_WIDTH) * (1/PIX_HEIGHT)
        nuc_rad[idx] = np.sqrt(nuc_area[idx]/math.pi)
        
        cell_info = {}
        cell_info["Nucleus Label"] = value
        cell_info["Size (um^2)"] = round(nuc_area[idx],3)
        cell_info["mean DAPI intensity"] = round(mean_DAPI[idx],3)
        
        cell_info["channel 2 intensity"] = round(mean_channel_2[idx],3) # channel 2 intensity
        cell_info["background-subtracted channel 2 intensity"] = round(mean_channel_2_nuc_bs[idx],3) # nuclear background-subtracted
        cell_info["normalised channel 2 intensity"] = round(mean_norm_channel_2[idx],3) # DAPI-normalised
        
        if np.unique(nuc_classifier[np.where(labels == value)])[0] == 1:  
            cell_info["channel 2 expression"] = 'negative'
        elif np.unique(nuc_classifier[np.where(labels == value)])[0] == 2:  
            cell_info["channel 2 expression"] = 'positive'
        else:
            cell_info["channel 2 expression"] = 'error'
        
        threshold = 0.3
        threshold_counter = 0
        
        try:
            cell_info["DAPI-Normalised aPKCi-TagRFP (left)"] = round(tag_rfp_left[idx].mean() / DAPI_norm,3)
            if cell_info["DAPI-Normalised aPKCi-TagRFP (left)"] >= threshold:
                threshold_counter += 1
        except AttributeError:
            pass
        try:
            cell_info["DAPI-Normalised aPKCi-TagRFP (right)"] = round(tag_rfp_right[idx].mean() / DAPI_norm,3)
            if cell_info["DAPI-Normalised aPKCi-TagRFP (right)"] >= threshold:
                threshold_counter += 1
        except AttributeError:
            pass
        try:
            cell_info["DAPI-Normalised aPKCi-TagRFP (up)"] = round(tag_rfp_up[idx].mean() / DAPI_norm,3)
            if cell_info["DAPI-Normalised aPKCi-TagRFP (up)"] >= threshold:
                threshold_counter += 1
        except AttributeError:
            pass
        try:
            cell_info["DAPI-Normalised aPKCi-TagRFP (down)"] = round(tag_rfp_down[idx].mean() / DAPI_norm,3)
            if cell_info["DAPI-Normalised aPKCi-TagRFP (down)"] >= threshold:
                threshold_counter += 1
        except AttributeError:
            pass
        
        if threshold_counter >= 2:
            cell_info["aPKCi-TagRFP expression"] = 'positive'
        else:
            cell_info["aPKCi-TagRFP expression"] = 'negative'
    
        rows_cells.append(cell_info)
        
    DAPI += list(mean_DAPI)
    Brachyury += list(mean_channel_2_nuc_bs)
    norm_Brachyury += list(mean_norm_channel_2)
    
    final_table_cells = pd.DataFrame.from_dict(rows_cells, orient='columns')
    final_table_cells.to_csv(output_directory + '\\Cell_Data_' + file_name + '.csv')