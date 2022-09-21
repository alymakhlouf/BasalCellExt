"""
RUNS FULL STARDIST 2D ANALYSIS (WITH LUMEN ANALYSIS)
GENERATES SPHEROID AND NUCLEAR CSV FILES
READS LABEL AND CLASSIFIER FILES (GENERATED THROUGH 'StarDist2D_Pre-Processing')
GENERATES LUMEN CONTACT MASK
ASSIGNS POSITIVE/NEGATIVE CLASSIFICATIONS AND POSITIVE CELL COUNT ACCORDING TO ILASTIK CLASSIFIER
GENERATES COLOUR-CODED NUCLEAR MASK (ACCORDING TO POSITIVE/NEGATIVE CLASSIFICATIONS) WITH LABELS

@author: Aly Makhlouf
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import math
import sys
import os
import tensorflow as tf
import skimage.io
import skimage.viewer
import numpy as np
import statistics as stat
import matplotlib
import glob
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import cv2
import array as arr
import napari
import gc

from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

import tqdm
from itertools import chain
from tifffile import imread
from csbdeep.utils import Path, normalize, download_and_extract_zip_file
from csbdeep.io import save_tiff_imagej_compatible
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import label as connected_component
from skimage import filters
from scipy.spatial import distance
import h5py
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# custom modules
from basalcellext import neighbours
from basalcellext import cell_filter
from basalcellext import radial_vectors as rv
from basalcellext import geometry
from basalcellext import hist_plot
from basalcellext import mask_generate

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

channel_2_array = []

path = 'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\' + folder_name + '\\'
files = glob.glob(path + '/*.tif')
files_list = list(files)

label_directory = 'D:\\User Data\\Aly\\Nanami\\TagRFP-aPKCi 3D spheroid for quantification\\Labels_' + folder_name + '\\'
if not os.path.exists(label_directory):
    os.makedirs(label_directory)

# READ FILES GENERATED AFTER PRE-PROCESSING    
label_files = glob.glob(label_directory + '/*.hdf5')
classifier_files = glob.glob(label_directory + '/*.h5')

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
    
    # find lumen neighbour cells
    lumen_edge = filters.sobel(segment_mask_lumen)
    lumen_neighbours = neighbours.find(lumen_edge, segment_mask)[0].astype(int)[1:]
    
    # IMAGE PROCESSING #
    
    PIX_WIDTH = 1
    PIX_HEIGHT = 1
    
    resize_factor = 1/PIX_WIDTH # or PIX_HEIGHT, based on scaled dimensions of pre-trained model (1,1)
    image_down = resize(image, (image.shape[0]//resize_factor, image.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    image_channel_2_down = resize(image_channel_2, (image_channel_2.shape[0]//resize_factor, image_channel_2.shape[1]//resize_factor),
                                anti_aliasing=False) * 255
    
    image_channel_4_down = resize(image_channel_4, (image_channel_4.shape[0]//resize_factor, image_channel_4.shape[1]//resize_factor),
                                anti_aliasing=True) * 255
    
    # make sure anti-aliasing is 'False'
    spheroid_mask_down = resize(spheroid_mask, (spheroid_mask.shape[0] // resize_factor, 
                            spheroid_mask.shape[1] // resize_factor),
                            anti_aliasing=False) * 255
    
    lumen_mask_down = resize(lumen_mask, (lumen_mask.shape[0] // resize_factor, 
                                lumen_mask.shape[1] // resize_factor),
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
    img = normalize(image_down, 1,99.8, axis=(0,1))
    img_channel_2 = image_channel_2_down
    img_channel_4 = image_channel_4_down
    
    # NUCLEAR SEGMENTATION, PROCESSING AND VIEWING #
    
    # labels, details = model.predict_instances(img, prob_thresh=0.45, nms_thresh=0.5)
    # for idx, label in enumerate(np.unique(labels)):
    #     if len(np.where(labels == label)[0]) < 1000:
    #         labels[np.where(labels == label)] = 0
    
    # load label file storing all segmented nuclear labels
    label_file = [match for match in label_files if file_name in match][0]
    hf = h5py.File(label_file, 'r')
    labels = hf.get('dataset')[:] # 
    
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
    
    classifier_file = [match for match in classifier_files if file_name in match][-1] # find classifier file with matching name
    
    with h5py.File(classifier_file) as infile:
                print(tuple(_ for _ in infile.keys()))
                nuc_classifier = infile["exported_data"]
                nuc_classifier = nuc_classifier[:,:,0] # ilastik object classifier gives positive and negative channel 2 nuclei
    
    # DEFINE VARIABLES FOR TABULATING DATA #
    
    rows_lumen = []
    rows_all_nuclei = []
    rows_nuclei = []
    rows_spheroid = []
    rows_analysis = []
    
    # define spheroid variables
    spheroid_x_c_p = round(xgrid_0[spheroid_mask == u_spheroid].mean())
    spheroid_y_c_p = round(ygrid_0[spheroid_mask == u_spheroid].mean())
    spheroid_area = len(spheroid_mask[np.where(spheroid_mask == u_spheroid)])
    spheroid_rad = round(np.sqrt(spheroid_area/math.pi))
    
    # tabulate lumen data
    lumen_x_c_p = round(xgrid_0[lumen_mask != 0].mean()) #approximate pixel x-coordinate of lumen centroid in original image
    lumen_y_c_p = round(ygrid_0[lumen_mask != 0].mean()) #approximate pixel y-coordinate of lumen centroid in original image
    
    segment_lumen_x_c_p = round(xgrid_0[segment_mask_lumen != 0].mean())
    segment_lumen_y_c_p = round(ygrid_0[segment_mask_lumen != 0].mean())
    
    lumen_info = {}
    lumen_info["x_coord"] = lumen_x_c_p
    lumen_info["y_coord"] = lumen_y_c_p
    lumen_info["x_eccentricity"] = lumen_x_c_p - spheroid_x_c_p
    lumen_info["y_eccentricity"] = lumen_y_c_p - spheroid_y_c_p
    lumen_info["eccentricity"] = np.sqrt(((lumen_x_c_p - spheroid_x_c_p)**2)+((lumen_y_c_p - spheroid_y_c_p)**2))
    
    rows_lumen.append(lumen_info)
    
    # tabulate nucleus data
    # exclude misclassified nuclei (false nuclei)
    
    signal_threshold = 0.01 # misclassified nuclei filtered by weak DAPI signal
    nuclei, nuclei_false = cell_filter.misclassify(img, labels, signal_threshold)
    
    # exclude out-of-range nuclei
    num_out_of_range, nuclei, sph_nuc_dist = cell_filter.out_of_range(y, x, nuclei, 
                                            labels, resize_factor, spheroid_y_c_p,
                                            spheroid_x_c_p, spheroid_mask, u_spheroid)
    
    # define remaining analysis variables
    nuc_area = np.zeros(len(nuclei)) 
    nuc_rad = np.zeros(len(nuclei))
    lum_nuc_dist = np.zeros(len(nuclei))
    lum_cell_dist = np.zeros(len(nuclei))
    sph_cell_dist = np.zeros(len(nuclei))
    mean_DAPI = np.zeros(len(nuclei))
    mean_channel_2 = np.zeros(len(nuclei))
    mean_channel_2_norm = np.zeros(len(nuclei))
    tag_rfp = np.zeros(len(nuclei))
    roundness_nuc = np.zeros(len(nuclei))
    roundness_cell = np.zeros(len(nuclei))
    segments = np.zeros(len(nuclei)).astype(int)
    apical_neighbours = [[] for i in range(len(nuclei))]
    first_apical_neighbour = np.zeros(len(nuclei)).astype(int)
    
    cytoplasm = {}
    nucleus = {}
    cell = {}
    mitotic = {}
    
    # alternative method for calculating nuclear signal intensity
    rad = 5
    mean_DAPI_2 = np.zeros(len(nuclei))
    
    # define variables for compiling radial vector data
    x_c_p = np.zeros(len(nuclei)).astype(int)
    y_c_p = np.zeros(len(nuclei)).astype(int)
    cell_x_c_p = np.zeros(len(nuclei)).astype(int)
    cell_y_c_p = np.zeros(len(nuclei)).astype(int)
    DAPI_tow = [ [] for i in range(len(nuclei))]
    DAPI_filter_tow = [ [] for i in range(len(nuclei))]
    DAPI_away = [ [] for i in range(len(nuclei))]
    DAPI_filter_away = [ [] for i in range(len(nuclei))]
    channel_3_tow = [ [] for i in range(len(nuclei))]
    channel_3_filter_tow = [ [] for i in range(len(nuclei))]
    channel_3_away = [ [] for i in range(len(nuclei))]
    channel_3_filter_away = [ [] for i in range(len(nuclei))]
    channel_4_tow = [ [] for i in range(len(nuclei))]
    channel_4_away = [ [] for i in range(len(nuclei))]
    segments_tow = [ [] for i in range(len(nuclei))]
    lumen_contact_0 = ['No'] * len(nuclei) # use an alternative to lumen_contact if it works
    
    lumen_norm_DAPI = img[np.append([np.ceil((np.where(lumen_mask == True)[0] / resize_factor)).astype(int)],
                                  [np.ceil((np.where(lumen_mask == True)[1] / resize_factor)).astype(int)],axis=0)].mean()
    lumen_norm_channel_2 = img_channel_2[np.append([np.ceil((np.where(lumen_mask == True)[0] / resize_factor)).astype(int)],
                                  [np.ceil((np.where(lumen_mask == True)[1] / resize_factor)).astype(int)],axis=0)].mean()
    #nuc_norms_apkci = np.zeros(len(nuclei)) # to normalise cytoplasmic aPKCi by nuclear aPKCi

    positive_count = 0
    
    for idx, value in enumerate(nuclei):
            
            print(str(idx) + ',' + str(value))
    
            # determine positions of nuclei with respect to lumen object
            x_c = xgrid[labels==nuclei[idx]].mean()
            y_c = ygrid[labels==nuclei[idx]].mean()
            
            x_c_p[idx] = round(x_c*resize_factor) #approximate pixel x-coordinate of nucleus centroid in rescaled image
            y_c_p[idx] = round(y_c*resize_factor) #approximate pixel y-coordinate of nucleus centroid in rescaled image
            
            x_step = lumen_x_c_p - x_c_p[idx]
            y_step = lumen_y_c_p - y_c_p[idx]
            
            lum_nuc_dist[idx] = round(np.sqrt((x_step**2)+(y_step**2)))
            nuc_area[idx] = ((labels==nuclei[idx]).sum()) * (1/PIX_WIDTH) * (1/PIX_HEIGHT)
            nuc_rad[idx] = np.sqrt(nuc_area[idx]/math.pi)
            mean_DAPI[idx] = image_down[labels==nuclei[idx]].mean()
            mean_DAPI_2[idx] = image[y_c_p[idx] - rad:y_c_p[idx] + rad,x_c_p[idx] - rad:x_c_p[idx] + rad].mean()
            mean_channel_2[idx] = image_channel_2_down[labels==nuclei[idx]].mean()
            mean_channel_2_norm[idx] = img_2_planar_DAPI_normalised[labels==nuclei[idx]].mean()
            
            # all_nucleus_info = {}
            # all_nucleus_info["Nucleus Label"] = value
            # all_nucleus_info["X-Coord"] = x_c_p[idx]
            # all_nucleus_info["Y-Coord"] = y_c_p[idx]
            # all_nucleus_info["Size (Pixels)"] = (labels==nuclei[idx]).sum()
            # all_nucleus_info["mean DAPI intensity"] = round(mean_DAPI[idx],3)
            # all_nucleus_info["DAPI-normalised channel 2 intensity"] = round(mean_channel_2_norm[idx],3)
            
            # if np.unique(nuc_classifier[np.where(labels == value)])[0] == 1:  
            #     all_nucleus_info["channel 2 expression"] = 'negative'
                    
            # elif np.unique(nuc_classifier[np.where(labels == value)])[0] == 2:  
            #     all_nucleus_info["channel 2 expression"] = 'positive'
            #     positive_count += 1
            
            # else:
            #     all_nucleus_info["channel 2 expression"] = 'error'
        
            # rows_all_nuclei.append(all_nucleus_info)
            
            # isolate each nucleus as a single object in the label mask
            nucleus_single_down = connected_component(labels == nuclei[idx])
            roundness_nuc[idx] = round(geometry.compute_roundness(nucleus_single_down),3)
            nucleus_single = resize(nucleus_single_down, (1024,1024), anti_aliasing = False)
            nucleus_single = connected_component(nucleus_single != nucleus_single[0,0])
            
            # identify cell segment associated with each nucleus
            segments[idx] = segment_mask[y_c_p[idx], x_c_p[idx]]
            if segments[idx] == 0:
                continue
            segment_single = connected_component(segment_mask == segments[idx])
            roundness_cell[idx] = round(geometry.compute_roundness(segment_single),3)
            
            # determine positions of cell segments with respect to lumen segment and spheroid
            cell_x_c_p[idx] = round(xgrid_0[segment_mask == segments[idx]].mean())
            cell_y_c_p[idx] = round(ygrid_0[segment_mask == segments[idx]].mean())
            
            cell_x_step = segment_lumen_x_c_p - cell_x_c_p[idx]
            cell_y_step = segment_lumen_y_c_p - cell_y_c_p[idx]
            
            cell_x_step_sph = spheroid_x_c_p - cell_x_c_p[idx]
            cell_y_step_sph = spheroid_y_c_p - cell_y_c_p[idx]
           
            lum_cell_dist[idx] = round(np.sqrt((cell_x_step**2)+(cell_y_step**2)))
            sph_cell_dist[idx] = round(np.sqrt((cell_x_step_sph**2)+(cell_y_step_sph**2)))
            
            # identify cytoplasmic regions based on nucleus and cell segment
            cytoplasm[value] = segment_single != nucleus_single # use nucleus as key since cell segments are not necessarily unique at this point
            
            # define radial vectors from every nucleus, towards and away from the lumen, in multiple channels
            search_field = 500 # specify depth of edge search field (pixels) 
            
            # changed inputs to cell_x_step, cell_y_step, cell_x_c_p, cell_y_c_p to do radial vector analysis with respect to cell segments instead of nuclei
            DAPI_tow[idx], DAPI_filter_tow[idx], channel_3_tow[idx], channel_3_filter_tow[idx], channel_4_tow[idx], segments_tow[idx] = rv.vec_tow_lum(x_step, 
                y_step, x_c_p[idx], y_c_p[idx], search_field, lumen_x_c_p, lumen_y_c_p, image, nuclei_edges, 
                image_channel_3, cell_edges, image_channel_4, segment_mask)
            
            DAPI_away[idx], DAPI_filter_away[idx], channel_3_away[idx], channel_3_filter_away[idx], channel_4_away[idx] = rv.vec_away_lum(x_step, 
                y_step, x_c_p[idx], y_c_p[idx], search_field, lumen_x_c_p, lumen_y_c_p, image, nuclei_edges, 
                image_channel_3, cell_edges, image_channel_4, segment_mask)
            
            nucleus[value] = [segments[idx]]
            cell[value] = [segments[idx]] # create a dictionary pairing nucleus with cell segment
    
            tag_rfp[idx] = round(image_channel_4[cytoplasm[value] == True].mean(),3)
            
            # use cell segment boundaries to identify radial neighbours of each cell
            boundary_steps = np.diff(segments_tow[idx])[np.where(np.diff(segments_tow[idx]) != 0)]
            apical_neighbours[idx] = np.unique(segments_tow[idx][segments_tow[idx] != segments[idx]]).astype(int)
        
            if cell[value] in lumen_neighbours:
                lumen_contact_0[idx] = 'Yes'
                first_apical_neighbour[idx] = 0
                
            else:
                try:
                    first_apical_neighbour[idx] = segments[idx] + boundary_steps[0]
                except IndexError:
                    pass
            
            nucleus[value].append(nuc_area[idx])
            nucleus[value].append(mean_DAPI[idx])
            
            cell[value].append(roundness_cell[idx])
            cell[value].append(lumen_contact_0[idx]) # append lumen contact information to cell dictionary
            cell[value].append(round(tag_rfp[idx],3))
            cell[value].append(apical_neighbours[idx])
            cell[value].append(first_apical_neighbour[idx])
            
            all_nucleus_info = {}
            all_nucleus_info["Nucleus Label"] = value
            all_nucleus_info["X-Coord"] = x_c_p[idx]
            all_nucleus_info["Y-Coord"] = y_c_p[idx]
            all_nucleus_info["Size (Pixels)"] = (labels==nuclei[idx]).sum()
            all_nucleus_info["mean DAPI intensity"] = round(mean_DAPI[idx],3)
            all_nucleus_info["DAPI-normalised channel 2 intensity"] = round(mean_channel_2_norm[idx],3)
            all_nucleus_info["DAPI-normalised channel 4 intensity"] = round(tag_rfp[idx]/planar_mean_DAPI,3)
            
            if np.unique(nuc_classifier[np.where(labels == value)])[0] == 1:  
                all_nucleus_info["channel 2 expression"] = 'negative'
                    
            elif np.unique(nuc_classifier[np.where(labels == value)])[0] == 2:  
                all_nucleus_info["channel 2 expression"] = 'positive'
                positive_count += 1
            
            else:
                all_nucleus_info["channel 2 expression"] = 'error'
        
            rows_all_nuclei.append(all_nucleus_info)

    cell_no_nucleus = np.zeros(len(u_segment)).astype(int)
    for idx, value in enumerate(u_segment):
        if value not in segments:
            cell_no_nucleus[idx] = value
        
    cell_no_nucleus = np.delete(cell_no_nucleus, np.where(cell_no_nucleus == 0)) # create an array of cell segments without a matched nucleus
    cell_with_nucleus = np.setdiff1d(np.union1d(u_segment, cell_no_nucleus), np.intersect1d(u_segment, cell_no_nucleus))
    
    # detect tiny, non-matched cell segments
    tiny_segments = np.zeros(len(u_segment)).astype(int)
    tiny_segment_threshold = 2000 # do not consider cell segments smaller than this size
    
    for idx, value in enumerate(size_segment):
        if value <= tiny_segment_threshold:
            tiny_segments[idx] = u_segment[idx]
            
    tiny_segments = np.delete(tiny_segments, np.where(tiny_segments ==0))
    
    tiny_enucleated = np.intersect1d(tiny_segments, cell_no_nucleus)
    
    nucleus_unfiltered = nucleus.copy() # make a pre-filtered copy of nucleus dictionary
    cell_unfiltered = cell.copy() # make a pre-filtered copy of cell dictionary
    
    # NUCLEUS DICTIONARY: Key: Nucleus Label
    #                     Values: Cell Segment label, Nuclear Area, Mean DAPI Intensity
    
    # CELL DICTIONARY: Key: Nucleus Label
    #                  Values: Cell Segment Label, Cell Roundness, Lumen Contact, TagRFP-aPKCi, Apical Neighbours, First Apical Neighbour
    
    # filter ambiguous cell segments/nuclei
    
    amb_idx = np.where(np.unique(segments, return_counts = True)[1] > 1)[0] # find indices of non-unique cell segments
    amb_segments = np.zeros(len(amb_idx)).astype(int)
    
    num_amb_nuclei = 0
    for value in amb_idx:
        num_amb_nuclei += np.unique(segments, return_counts = True)[1][value]
    
    amb_nuclei = np.zeros(num_amb_nuclei).astype(int)
    
    for idx, value in enumerate(amb_idx):
        amb_segments[idx] = np.unique(segments)[value] # find labels of non-unique cell segments
        
    amb_segments = amb_segments[amb_segments != 0] # exclude 0
        
    # set thresholds for identifying mitotic cells
    mitotic_cell_roundness = 0.7 # cells more round than this
    mitotic_nucleus_roundness = 0.65 # nuclei less round than this
    mitotic_nucleus_area = 4500 # nuclei smaller than this
    mitotic_DAPI_threshold = np.mean(mean_DAPI) + np.std(mean_DAPI) # bright nuclei
    
    # update nucleus and cell dictionaries to remove ambiguous nuclei/cell segments
    c = 0
    for k,v in list(cell.items()):
        if v[0] in amb_segments and v[1] < mitotic_cell_roundness: # mitotic cells may be ambiguous due to DNA replication
            amb_nuclei[c] = k
            nucleus_unfiltered[k].append('ambiguous')
            cell_unfiltered[k].append('ambiguous')
            del nucleus[k]
            del cell[k]
            c += 1
#%%            
    # filter doublets and mitotic cells
    
    doublet_threshold = 14000
    nuclei_filtered, doublet = cell_filter.doublet(nuclei, labels, PIX_WIDTH, PIX_HEIGHT, doublet_threshold) 
    
    # update nucleus and cell dictionaries to remove mitotic nuclei/cell segments, or undetected out-of-range nuclei
    num_out_of_range_0 = num_out_of_range
    
    for k,v in list(cell.items()):
        if v[0] == 0:
            num_out_of_range += 1
            nucleus_unfiltered[k].append('out-of-range')
            cell_unfiltered[k].append('out-of-range')
            oor_idx = np.where(nuclei_filtered == k)
            nuclei_filtered = np.delete(nuclei_filtered, oor_idx)
            #sph_nuc_dist = np.delete(sph_nuc_dist, oor_idx)
            del nucleus[k]
            del cell[k]
        
        elif v[1] > mitotic_cell_roundness and nucleus[k][1] < mitotic_nucleus_area:
            nucleus_unfiltered[k].append('mitotic')
            cell_unfiltered[k].append('mitotic')
            mitotic[k] = v
            del nucleus[k]
            del cell[k]
            
        elif k in doublet:
            nucleus_unfiltered[k].append('doublet')
            cell_unfiltered[k].append('doublet')
            del nucleus[k]
            del cell[k]
    
    if num_out_of_range > num_out_of_range_0:
        spheroid_area = np.size(np.where(segment_mask_spheroid == u_segment_spheroid)[0]) 
        roundness_spheroid = round(geometry.compute_roundness(segment_mask_spheroid),3)
    
    # create an array of mitotic cells
    c = 0
    mitotic_segment_array = np.zeros(len(mitotic)).astype(int)
    for k, v in list(mitotic.items()):
        mitotic_segment_array[c] = v[0]
        c += 1
    mitotic_segment_array = np.unique(mitotic_segment_array)
    non_mitotic_segment_array = np.setdiff1d(u_segment, mitotic_segment_array)
    
    # update 'apical neighbours' array to remove mitotic cells and tiny, enucleated segments
    closest_dist = np.zeros(len(cell))
    closest_dist_c = 0
    min_apical_neighbours = [ [] for i in range(len(cell))]
    
    for k,v in list(cell.items()):
        
        # find direct neighbours of every cell segment
        segment_edge = filters.sobel(segment_mask == v[0])
        segment_neighbours = neighbours.find(segment_edge, segment_mask)[0].astype(int)[1:]
        segment_neighbours = np.delete(segment_neighbours, np.where(segment_neighbours == v[0]))
        
        #cell[k].append(segment_neighbours)
        
        c = 0
        while c < len(v[4]):
            if v[4][c] in mitotic_segment_array:
                v[4][c] = 10000 # arbitrarily assign unwanted cells to 10000, to later delete
            c += 1
            
        v[4] = np.delete(v[4], np.where(v[4] == 10000))
        
        # update cell-lumen contact classification, if the only obstructing cell was mitotic or an enucleated segment
        if len(v[4]) == 1:
            if v[4] == 0 or v[4] in cell_no_nucleus:  
                v[2] = 'Yes'
                v[5] = 0 
    
        # update cell-lumen contact classification, if direct neighbour is a tiny, enucleated segment in contact with the lumen
    
        final_neighbour_search = np.intersect1d(v[4], segment_neighbours)
        
        c = 0
        while c < len(final_neighbour_search):
            if final_neighbour_search[c] in np.intersect1d(lumen_neighbours, tiny_enucleated):
                #segment_mask[np.where(segment_mask == v[5])] = v[0] # merge segments (reassign label)
                v[2] = 'Yes'
                v[5] = 0
            c += 1
          
        # update cell-lumen contact classification, if closest distance to the lumen is less than a tiny segment
        
        tiny_segment_length = math.sqrt((4*tiny_segment_threshold)/math.pi)
        
        d1 = np.asarray(tuple(zip(*np.where(lumen_edge != 0)))) # define coordinates for all lumen edge points
        #d1 = np.asarray(tuple(zip(*np.where(segment_mask_lumen != 0))))   
        
        if v[2] == 'No' and len(v[4] <= 3): # second condition filters out segments that are highly obstructed from the lumen
        
            d2 = np.asarray(tuple(zip(*np.where(segment_edge != 0)))) # define coordinates for all segment edge points
            closest_dist[closest_dist_c] = round(np.min(distance.cdist(d1,d2)),3) 
            cell[k].append(closest_dist[closest_dist_c])
            
            # define radial vector along the minimum distance between segment and lumen
            
            lumen_min_dist_coord_x = d1[np.argmin(distance.cdist(d1,d2).min(axis=1))][1]
            lumen_min_dist_coord_y = d1[np.argmin(distance.cdist(d1,d2).min(axis=1))][0]
            segment_min_dist_coord_x = d2[np.argmin(distance.cdist(d2,d1).min(axis=1))][1]
            segment_min_dist_coord_y = d2[np.argmin(distance.cdist(d2,d1).min(axis=1))][0]
            
            min_x_step = lumen_min_dist_coord_x - segment_min_dist_coord_x
            min_y_step = lumen_min_dist_coord_y - segment_min_dist_coord_y                                    
            
            min_segments_tow = rv.min_vec_tow_lum(min_x_step, min_y_step, segment_min_dist_coord_x, 
                                                 segment_min_dist_coord_y, search_field, lumen_min_dist_coord_x,
                                                 lumen_min_dist_coord_y, segment_mask)
            
            min_apical_neighbours[closest_dist_c] = np.unique(min_segments_tow[min_segments_tow != v[0]]).astype(int)
            if 0 not in min_apical_neighbours[closest_dist_c]:
                min_apical_neighbours[closest_dist_c] = np.insert(min_apical_neighbours[closest_dist_c], 0, 0)
            
            
            # re-classify cells that are very close to the lumen and are only obstructed by enucleated or mitotic segments
            if  closest_dist[closest_dist_c] < tiny_segment_length:
                v[2] = 'Yes'
            if len(np.intersect1d(min_apical_neighbours[closest_dist_c],cell_with_nucleus)) == 0:
                v[2] = 'Yes'
            if len(np.intersect1d(min_apical_neighbours[closest_dist_c],non_mitotic_segment_array)) == 0:
                v[2] = 'Yes'
                
        closest_dist_c += 1
    
    # output a cell classification mask based on lumen contact
    
    lumen_contact_mask = segment_mask.copy()
    final_segments_array = np.zeros(len(cell)).astype(int)
       
    lumen_contact_mask[np.where(segment_mask_lumen == 1)] = 1
    
    for k,v in list(cell.items()):
        
        idx = list(cell).index(k)
        final_segments_array[idx] = v[0]
        
        if v[2] == 'Yes':
            lumen_contact_mask[np.where(segment_mask == v[0])] = 2
        else:
            lumen_contact_mask[np.where(segment_mask == v[0])] = 3
      
    # show all remaining cell segment types in the mask 
    for idx, value in enumerate(u_segment):
        if value not in final_segments_array: # show non_analysed cells
            lumen_contact_mask[np.where(segment_mask == value)] = 0
        
    # generate and save segment and lumen contact masks
    mask_generate.save(lumen_contact_mask, file_name, 'Mask_Lumen_Contact', path)
    mask_generate.save(segment_mask, file_name, 'Mask_Segment', path)
    
    # output nucleus information for non-ambiguous, non-mitotic cells        
    for k,v in list(cell.items()):
        
        idx = np.where(nuclei == k)
        
        nucleus_info = {}
        nucleus_info["Nucleus Label"] = k
        nucleus_info["X-Coord"] = x_c_p[idx][0]
        nucleus_info["Y-Coord"] = y_c_p[idx][0]
        nucleus_info["Size (Pixels)"] = math.ceil(nucleus[k][1])
        nucleus_info["mean DAPI intensity"] = round(nucleus[k][2],3)
        nucleus_info["DAPI-normalised channel 2 intensity"] = round(mean_channel_2_norm[idx][0],3)
        
        rows_nuclei.append(nucleus_info)
    
    # plot spheroid-centroid distance distributions
    fig1, ax1, mode_dist_nuc, hist_dist_nuc = hist_plot.distribution(sph_nuc_dist, 10)
    fig2, ax2, mode_size_cell, hist_size_cell = hist_plot.distribution(size_segment, 1000)
    
    # specify the range of spheroid_centroid-to-nuclear distances centered at the mode (simple epithelium)
    error = 0.06 # set arbitrary error term to expand the range 
    epi_upper = ((1 - (np.mean(nuc_rad)/spheroid_rad)) + error) * spheroid_rad
    epi_lower = (2*(mode_dist_nuc / spheroid_rad) - (epi_upper/spheroid_rad)) * spheroid_rad
    
    # tabulate spheroid data
    spheroid_info = {}
    spheroid_info["X-Coord"] = spheroid_x_c_p
    spheroid_info["Y-Coord"] = spheroid_y_c_p
    spheroid_info["Total Surface Area (um^2)"] = float(str(round(spheroid_area * PIX_WIDTH * PIX_HEIGHT,3)))
    spheroid_info["Cell Surface Area (um^2)"] = float(str(round(len(np.where(segment_mask != 0)[0]) * PIX_WIDTH * PIX_HEIGHT,3)))
    spheroid_info["Radius"] = spheroid_rad
    spheroid_info["Modal Epithelial Radius"] = mode_dist_nuc
    spheroid_info["Roundness"] = roundness_spheroid
    spheroid_info["Approx. # Nuclei"] = len(nuc_labels) + len(doublet) - len(nuclei_false) - num_out_of_range
    spheroid_info["Approx. % T+ Nuclei"] = float(str(round((positive_count / (len(nuc_labels) - len(nuclei_false) - num_out_of_range)) * 100,3)))
    spheroid_info["# Cells Analysed"] = len(cell)
    spheroid_info["# Mitotic Cells"] = len(mitotic_segment_array)
    spheroid_info["# Ambiguous Cells"] = len(amb_nuclei)
    spheroid_info["# Out-of-Range Nuclei"] = num_out_of_range
    spheroid_info["# Doublets"] = len(doublet)
    spheroid_info["# False Nuclei"] = len(nuclei_false)
    
    rows_spheroid.append(spheroid_info)
    
    # output analysis for non-ambiguous, non-mitotic cells  
    for k,v in list(cell.items()):
        
        idx = np.where(nuclei == k)
        
        analysis_info = {}
        analysis_info["Nucleus Label"] = k
        analysis_info["Cell Label"] = v[0]
        analysis_info["DAPI-normalised channel 4"] = round(v[3]/planar_mean_DAPI,3) # all cells normalised to an average DAPI intensity
        analysis_info["DAPI-normalised channel 2"] = round(mean_channel_2_norm[idx][0],3)
        
        if np.unique(nuc_classifier[np.where(labels == k)])[0] == 1:  
            analysis_info["channel 2 expression"] = 'negative'
                    
        elif np.unique(nuc_classifier[np.where(labels == k)])[0] == 2:  
            analysis_info["channel 2 expression"] = 'positive'
        
        ## if ilastik classifier remains ambiguous, use an arbitrary threshold (based on observation)            
        else:
            if round(mean_channel_2_norm[idx][0],3) > 0.3:
                analysis_info["channel 2 expression"] = 'positive*'
            else:
                analysis_info["channel 2 expression"] = 'negative*'
        
        analysis_info["Lumen Contact"] = v[2]
        analysis_info["Radial Distance (Pixels)"] = round(sph_nuc_dist[idx][0])
        analysis_info["Normalised Radial Distance"] = round(round(sph_nuc_dist[idx][0])/spheroid_rad, 3)
        
        rows_analysis.append(analysis_info)
    
    channel_2_array += list(mean_channel_2_norm)
    
    mask_generate.save(nuc_classifier, file_name, 'Nuclei_Labels', path)
    
    for idx, dict in enumerate(rows_all_nuclei[0:]):
        
        
        img_label = str(rows_all_nuclei[idx]['Nucleus Label'])
        
        img_label_x = rows_all_nuclei[idx]['X-Coord']
        img_label_y = rows_all_nuclei[idx]['Y-Coord']
        
        label_img = Image.open(path + 'Nuclei_Labels_' + file_name + '.png')
        I1 = ImageDraw.Draw(label_img)
        myFont = ImageFont.truetype('arial.ttf', 20)
        I1.text((img_label_x - 10, img_label_y - 10), img_label, font=myFont, fill=(0, 0, 0))
        label_img.save(path + 'Nuclei_Labels_' + file_name + '.png')
    
    
    final_table_lumen = pd.DataFrame.from_dict(rows_lumen, orient='columns') 
    final_table_lumen.to_csv(path + 'Lumen_Data_'+ file_name + '.csv')
    
    final_table_all_nuclei = pd.DataFrame.from_dict(rows_all_nuclei, orient='columns')
    final_table_all_nuclei.to_csv(path + 'All_Nucleus_Data_' + file_name + '.csv')
    
    final_table_nuclei = pd.DataFrame.from_dict(rows_nuclei, orient='columns')
    final_table_nuclei.to_csv(path + 'Nucleus_Data_' + file_name + '.csv')
    
    final_table_spheroid = pd.DataFrame.from_dict(rows_spheroid, orient='columns') 
    final_table_spheroid.to_csv(path + 'Spheroid_Data_'+ file_name + '.csv')

    final_table_analysis = pd.DataFrame.from_dict(rows_analysis, orient='columns') 
    final_table_analysis.to_csv(path + 'Analysis_Data_'+ file_name + '.csv')

channel_2_array = np.array(channel_2_array)
