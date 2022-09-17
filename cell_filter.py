# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 17:39:46 2021

@author: Aly Makhlouf
"""

import numpy as np
import math
from skimage.transform import resize
import radial_vectors as rv

def misclassify(img, labels, signal_threshold):
    
    nuclei = np.unique(labels)[1:] # exclude index 0 as background label

    mean_DAPI = np.zeros(len(nuclei))
    
    for idx in np.arange(len(nuclei)):

        mean_DAPI[idx] = round(img[labels==nuclei[idx]].mean(),3)
    
    nuclei_false_idx = np.where(mean_DAPI <= signal_threshold)[0] 
    nuclei_filter_signal = np.delete(nuclei, nuclei_false_idx)
    nuclei_false = nuclei[nuclei_false_idx]
    
    return nuclei_filter_signal, nuclei_false

def doublet(nuclei, labels, PIX_WIDTH, PIX_HEIGHT, doublet_threshold):

    nuc_area = np.zeros(len(nuclei)) 
    nuc_rad = np.zeros(len(nuclei)) # in pixels in original image, to define appropriate sliding window length for convolution
    
    for idx in np.arange(len(nuclei)):
    
        nuc_area[idx] = ((labels==nuclei[idx]).sum()) * (1/PIX_WIDTH) * (1/PIX_HEIGHT)
        nuc_rad[idx] = np.sqrt(nuc_area[idx]/math.pi)
    
    doublet_idx = np.where(nuc_area > doublet_threshold)[0]
    nuclei_filter_doublet = np.delete(nuclei, doublet_idx)
    doublet = nuclei[doublet_idx]
    
    return nuclei_filter_doublet, doublet
    
def out_of_range(y, x, nuclei, labels, resize_factor, spheroid_y_c_p, 
                 spheroid_x_c_p, spheroid_mask, u_spheroid):
    
    ygrid, xgrid = np.meshgrid(y,x, indexing='ij')       
    sph_nuc_dist = np.zeros(len(nuclei))
    
    for idx in np.arange(len(nuclei)): 

        x_c = xgrid[labels==nuclei[idx]].mean()
        y_c = ygrid[labels==nuclei[idx]].mean()
        
        x_c_p = round(x_c*resize_factor) #approximate pixel x-coordinate of nucleus centroid in rescaled image
        y_c_p = round(y_c*resize_factor) #approximate pixel y-coordinate of nucleus centroid in rescaled image
        
        x_step_sph = spheroid_x_c_p - x_c_p
        y_step_sph = spheroid_y_c_p - y_c_p
        
        # exclude nuclei outside the spheroid
        if spheroid_mask[y_c_p, x_c_p] != u_spheroid:
            sph_nuc_dist[idx] = 0
            continue
        
        sph_nuc_dist[idx] = np.sqrt((x_step_sph**2)+(y_step_sph**2))
        
    
    out_of_range = 0

    if len(np.where(sph_nuc_dist == 0)[0]) > 0: 
    
        out_of_range = len(np.where(sph_nuc_dist == 0)[0])
        nuclei = np.delete(nuclei, np.where(sph_nuc_dist == 0))
        sph_nuc_dist = np.delete(sph_nuc_dist, np.where(sph_nuc_dist == 0))
           
    return  out_of_range, nuclei, sph_nuc_dist

def mitotic(apical_idx, apical, mean_DAPI, mitotic_DAPI_threshold, 
            roundness, mitotic_roundness_threshold, nuc_area, mitotic_nucleus_area, nuclei):
    
    mitotic = np.zeros(len(apical_idx))
    count = 0
    
    for idx in apical_idx:
    
        if  nuc_area[idx] > mitotic_nucleus_area and mean_DAPI[idx] < mitotic_DAPI_threshold:
            count += 1
            continue
        elif nuc_area[idx] < mitotic_nucleus_area:
            mitotic[count] = apical[count]
            count += 1

    mitotic = (np.delete(mitotic, np.where(mitotic == 0))).astype(int)
    non_mitotic = (np.delete(nuclei, (nuclei[:, None] == mitotic).argmax(axis=0))).astype(int)
    
    return mitotic, non_mitotic

def tag_rfp(labels_up, nuclei, image_channel_4, image_channel_3, spheroid_mask, u_spheroid, DAPI_filter_tow, 
            DAPI_tow, DAPI_filter_away, DAPI_away, channel_3_filter_tow, channel_3_filter_away, 
             channel_3_tow, channel_3_away, channel_4_tow, channel_4_away, tag_rfp_search,
             mean_DAPI, mitotic_DAPI_threshold, nuc_rad, nuc_area, mitotic_nucleus_area):
    
    # find mean TagRFP-aPKCi intensity in non-nuclear regions of the spheroid
    image_channel_4_mean = np.mean(image_channel_4[(spheroid_mask == u_spheroid)*(labels_up == np.min(labels_up))])
    # find mean E-cadherin intensity in non-nuclear, non-low-pixel-intensity (>20) regions of the spheroid
    image_channel_3_mean = np.mean(image_channel_3[(spheroid_mask == u_spheroid)*(labels_up == np.min(labels_up))*(image_channel_3 >= 20)])
    
    DAPI_edge_tow = np.zeros(len(nuclei), dtype = 'int64')
    DAPI_edge_away = np.zeros(len(nuclei), dtype = 'int64')
    channel_3_edge_tow = np.zeros(len(nuclei), dtype = 'int64')
    channel_3_edge_away = np.zeros(len(nuclei), dtype = 'int64')
    tag_rfp_tow = np.zeros(len(nuclei))
    tag_rfp_away = np.zeros(len(nuclei))
    
    # set analysis parameters
    DAPI_edge_threshold = 0.8 # DAPI intensity threshold for edge detection
    DAPI_edge_threshold_low = np.array([0.2,0.4,0.6])
    DAPI_window = 5 # sliding window length for edge-filtered DAPI channel
    channel_3_threshold = 0.4 # detect edges with intensity equal to or above this threshold
    channel_3_window = 5 # sliding window length for E-cadherin (both original channel AND edge-filtered)
    channel_4_window = 20 # sliding window length for TagRFP-aPKCi
    
    epithelial = np.zeros(len(nuclei))
    basal = np.zeros(len(nuclei))
    apical = np.zeros(len(nuclei))
    lumen_contact = np.zeros(len(nuclei))
    mitotic = np.zeros(len(nuclei))
    
    for idx, value in enumerate(nuclei):
    
        # determine edge positions of all nuclei using edge-filtered channels
        
        # in case edge-filtered DAPI channel fails to detect nucleus boundaries
        if np.max(rv.moving_average(DAPI_filter_tow[idx], DAPI_window)[0:min(100, len(DAPI_filter_tow[idx]))]) < DAPI_edge_threshold:
            if np.max(rv.moving_average(DAPI_filter_tow[idx], DAPI_window)[0:min(100, len(DAPI_filter_tow[idx]))]) in DAPI_edge_threshold_low:
                DAPI_edge_tow[idx] = np.where(np.isin(rv.moving_average(DAPI_filter_tow[idx], DAPI_window), DAPI_edge_threshold_low))[0][0].item() + 2 # determine edge based on lower filtered threshold
                
            else:  
                DAPI_edge_tow[idx] = np.argmax(rv.moving_average(np.diff(np.diff(rv.moving_average(rv.moving_average(DAPI_tow[idx], 20), 50))),50)).item() # determine edge based on inflection point of original channel pixel intensities
        
        else:
            DAPI_edge_tow[idx] = np.where(rv.moving_average(DAPI_filter_tow[idx], DAPI_window) >= DAPI_edge_threshold)[0][0].item()
            
        # in case edge-filtered E-cadherin channel fails to detect membrane boundaries
        channel_3_tow_peak = np.max(rv.moving_average(channel_3_filter_tow[idx], channel_3_window)[DAPI_edge_tow[idx]:])
        
        if channel_3_tow_peak < channel_3_threshold:
            channel_3_edge_tow[idx] = np.argmax(rv.moving_average(channel_3_tow[idx],channel_3_window)).item() # determine edge using original channel pixel intensities
        
        elif len(np.where(np.diff(np.diff(rv.moving_average(channel_3_filter_tow[idx], 5))) == 0.2)[0]) > 4: # if the number of detected peaks > 1 (novel method for peak detection)
            channel_3_edge_tow[idx] = np.argmax(rv.moving_average(channel_3_filter_tow[idx],channel_3_window)).item()
        
        else:
            channel_3_edge_tow[idx] = np.where(rv.moving_average(channel_3_filter_tow[idx],channel_3_window) == channel_3_tow_peak)[0][0].item()
        
        # determine edge positions of all nuclei using edge-filtered channels
        
        # in case edge-filtered DAPI channel fails to detect nucleus boundaries
        if np.max(rv.moving_average(DAPI_filter_away[idx], DAPI_window)[0:100]) < DAPI_edge_threshold:
            if np.max(rv.moving_average(DAPI_filter_away[idx], DAPI_window)[0:100]) in DAPI_edge_threshold_low:
                DAPI_edge_away[idx] = np.where(np.isin(rv.moving_average(DAPI_filter_away[idx], DAPI_window), DAPI_edge_threshold_low))[0][0].item() + 2 # determine edge based on lower filtered threshold
            
            else:
                DAPI_edge_away[idx] = np.argmax(rv.moving_average(np.diff(np.diff(rv.moving_average(rv.moving_average(DAPI_away[idx], 20), 50))),50)).item() # determine edge based on inflection point of original channel pixel intensities
        
        else:
            DAPI_edge_away[idx] = np.where(rv.moving_average(DAPI_filter_away[idx],DAPI_window) >= DAPI_edge_threshold)[0][0].item()
        
         # in case edge-filtered E-cadherin channel fails to detect membrane boundaries
        channel_3_away_peak = np.max(rv.moving_average(channel_3_filter_away[idx], channel_3_window)[DAPI_edge_away[idx]:])
        
        if channel_3_away_peak < channel_3_threshold:
            channel_3_edge_away[idx] = np.argmax(rv.moving_average(channel_3_away[idx],channel_3_window)[DAPI_edge_away[idx]:]) # determine edge using original channel pixel intensities

        elif len(np.where(np.diff(np.diff(rv.moving_average(channel_3_filter_away[idx], channel_3_window))) == 0.2)[0]) > 4: # if the number of detected peaks > 1 (novel method for peak detection)
            channel_3_edge_away[idx] = np.argmax(rv.moving_average(channel_3_filter_away[idx], channel_3_window)).item()

        else:
            channel_3_edge_away[idx] = np.where(rv.moving_average(channel_3_filter_away[idx],channel_3_window) == channel_3_away_peak)[0][0].item()
        
        # determine if cell is basally-extruded (has lost contact with the lumen)
        zero_crossings_tow = np.where(np.diff(np.sign(rv.moving_average(DAPI_filter_tow[idx], DAPI_window))))[0] + 1
        zero_crossings_away = np.where(np.diff(np.sign(rv.moving_average(DAPI_filter_away[idx], DAPI_window))))[0] + 1
        
        # print(zero_crossings_tow)
        # print(zero_crossings_away)
        # print(idx)
        # print(value)
        
        if len(zero_crossings_tow) == 0 or len(zero_crossings_away) == 0: # in case no edge is detected
            zero_crossings_tow = zero_crossings_away = [0, round(np.mean(nuc_rad))]

        if (rv.moving_average(DAPI_filter_away[idx], DAPI_window)[zero_crossings_away[1]:] == 0).all(): # identify cells in contact with the basement membrane
            if (rv.moving_average(DAPI_filter_tow[idx], DAPI_window)[zero_crossings_tow[1]:] == 0).all(): # if no DAPI edges are detected towards the lumen
                epithelial[idx] = nuclei[idx]
                lumen_contact[idx] = nuclei[idx]
            
            else:
                basal[idx] = nuclei[idx] # identify basally-extruded cells
            
        else: # identify cells not in contact with the basement membrane
            apical[idx] = nuclei[idx]
            if (rv.moving_average(DAPI_filter_tow[idx], DAPI_window)[zero_crossings_tow[1]:] == 0).all(): # if no DAPI edges are detected towards the lumen
                lumen_contact[idx] = nuclei[idx]
            
            if nuc_area[idx] < mitotic_nucleus_area:
                    mitotic[idx] = nuclei[idx]
        
        # determine maximum TagRFP-aPKCi intensity, in both radial directions, between the nucleus edge and cell boundary (cytoplasm)
        
        tag_rfp_tow[idx] = np.max(rv.moving_average(channel_4_tow[idx],channel_4_window)[DAPI_edge_tow[idx].item():(DAPI_edge_tow[idx].item() + tag_rfp_search)]) 
        tag_rfp_away[idx] = np.max(rv.moving_average(channel_4_away[idx],channel_4_window)[DAPI_edge_away[idx].item():(DAPI_edge_away[idx].item() + tag_rfp_search)])
    
    epithelial = (np.delete(epithelial, np.where(epithelial == 0))).astype(int)  
    basal = (np.delete(basal, np.where(basal == 0))).astype(int)
    apical = (np.delete(apical, np.where(apical == 0))).astype(int)
    lumen_contact = (np.delete(lumen_contact, np.where(lumen_contact == 0))).astype(int)
    mitotic = (np.delete(mitotic, np.where(mitotic == 0))).astype(int)
    non_mitotic = (np.delete(nuclei, (nuclei[:, None] == mitotic).argmax(axis=0))).astype(int)    
      
    return image_channel_4_mean, image_channel_3_mean, DAPI_edge_tow, DAPI_edge_away, channel_3_edge_tow, channel_3_edge_away, tag_rfp_tow, tag_rfp_away, epithelial, basal, apical, lumen_contact, mitotic, non_mitotic

