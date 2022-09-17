# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:16:01 2021

@author: Aly Makhlouf
"""

import math
import numpy as np

def vec_tow_lum(x_step, y_step, x_c_p, y_c_p, search_field, lumen_x_c_p, lumen_y_c_p, 
                image, nuclei_edges, image_channel_3, cell_edges, image_channel_4,
                segment_mask):
    
    y_c_p_0 = y_c_p
    x_c_p_0 = x_c_p
            
    if x_step != 0:
    
        if abs(y_step/x_step) <= 1:
            
            tow_field = min(search_field, abs(x_step)) # specify the depth of search field towards the lumen (in pixels)
            vec_tow = np.zeros(tow_field)
            DAPI_tow = np.zeros(tow_field)
            DAPI_filter_tow = np.zeros(tow_field)
            channel_3_tow = np.zeros(tow_field)
            channel_3_filter_tow = np.zeros(tow_field)
            channel_4_tow = np.zeros(tow_field)
            segments_tow = np.zeros(tow_field)
              
            if y_step == 0:   
                x_step_1 = abs(lumen_x_c_p - x_c_p)
                
            else:
                x_step_1 = round(abs(x_step/y_step)) # x-step normalised to y-step-size = 1
            
            # define vectors from the nucleus towards the lumen
                
            if y_step <= 0 and x_step <= 0:
                
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p -= 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p -= 1
    
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                          
            elif y_step >= 0 and x_step >= 0: 
                   
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p += 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p += 1
    
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                        
            elif y_step <= 0 and x_step >= 0: 
                   
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p += 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p -= 1
    
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p] 
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
            
            elif y_step >= 0 and x_step <= 0: 
                   
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p -= 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p += 1
    
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                   
        else:
            
            tow_field = min(search_field, abs(y_step)) # specify the depth of search field towards the lumen (in pixels)
            vec_tow = np.zeros(tow_field)
            DAPI_tow = np.zeros(tow_field)
            DAPI_filter_tow = np.zeros(tow_field)
            channel_3_tow = np.zeros(tow_field)
            channel_3_filter_tow = np.zeros(tow_field)
            channel_4_tow = np.zeros(tow_field)
            segments_tow = np.zeros(tow_field)
            
            if x_step == 0:
                y_step_1 = abs(lumen_y_c_p - y_c_p) # y_step normalised to x-step-size = 1
            
            else:
                y_step_1 = round(abs(y_step/x_step)) # y_step normalised to x-step-size = 1
            
            # define vectors from the nucleus towards the lumen
                
            if y_step <= 0 and x_step < 0:
                
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p -= 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p -= 1
                               
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                          
            elif y_step >= 0 and x_step > 0: 
                   
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p += 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p += 1
    
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                        
            elif y_step <= 0 and x_step > 0: 
                   
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p -= 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p += 1
    
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
            
            elif y_step >= 0 and x_step < 0: 
                   
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p += 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p -= 1
     
                        DAPI_tow[i] = image[y_c_p, x_c_p]   
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                        
    else:
        
        tow_field = search_field # specify the depth of search field towards the lumen (in pixels)
        vec_tow = np.zeros(tow_field)
        DAPI_tow = np.zeros(tow_field)
        DAPI_filter_tow = np.zeros(tow_field)
        channel_3_tow = np.zeros(tow_field)
        channel_3_filter_tow = np.zeros(tow_field)
        channel_4_tow = np.zeros(tow_field)
        segments_tow = np.zeros(tow_field)
                  
        if y_step <= 0:
                
                for i in range(tow_field):
                    if abs(y_c_p - y_c_p_0) % y_step == 0:
                        y_c_p -= 1
        
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                              
        else: 
                       
                for i in range(tow_field):
                    if abs(y_c_p - y_c_p_0) % y_step == 0:
                        y_c_p += 1
        
                        DAPI_tow[i] = image[y_c_p, x_c_p]
                        DAPI_filter_tow[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_tow[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_tow[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_tow[i] = image_channel_4[y_c_p, x_c_p]
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
    
    return DAPI_tow, DAPI_filter_tow, channel_3_tow, channel_3_filter_tow, channel_4_tow, segments_tow

def vec_away_lum(x_step, y_step, x_c_p, y_c_p, search_field, lumen_x_c_p, lumen_y_c_p, 
                image, nuclei_edges, image_channel_3, cell_edges, image_channel_4,
                segment_mask):
    
    y_c_p_0 = y_c_p
    x_c_p_0 = x_c_p
                    
    if x_step != 0:
    
        if abs(y_step/x_step) <= 1:
            away_field = search_field  # specify the depth of search field away from the lumen (in pixels)
            vec_away = np.zeros(away_field)
            DAPI_away = np.zeros(away_field)
            DAPI_filter_away = np.zeros(away_field)
            channel_3_away = np.zeros(away_field)
            channel_3_filter_away = np.zeros(away_field)
            channel_4_away = np.zeros(away_field)
            
            if y_step == 0:   
                x_step_1 = abs(lumen_x_c_p - x_c_p)
                
            else:
                x_step_1 = round(abs(x_step/y_step)) # x-step normalised to y-step-size = 1
            
            # define vectors from the nucleus towards the lumen
            
            if y_step <= 0 and x_step < 0:
                
                for i in range(away_field):
                    try:
                        x_c_p += 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p += 1
                            
                        DAPI_away[i] = image[y_c_p, x_c_p]
                        DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                    
                        if y_c_p <= 0 or x_c_p <= 0:
                                break
                    
                    except IndexError:
                        break
                           
            elif y_step >= 0 and x_step > 0: 
                   
                for i in range(away_field):
                    try:
                        x_c_p -= 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p -= 1
    
                        DAPI_away[i] = image[y_c_p, x_c_p]
                        DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                    
                        if y_c_p <= 0 or x_c_p <= 0:
                                break
                    
                    except IndexError:
                        break
                           
            elif y_step <= 0 and x_step > 0: 
                   
                for i in range(away_field):
                    try:
                        x_c_p -= 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p += 1
    
                        DAPI_away[i] = image[y_c_p, x_c_p]
                        DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_away[i] = image_channel_4[y_c_p, x_c_p] 
                    
                        if y_c_p <= 0 or x_c_p <= 0:
                                break
                    
                    except IndexError:
                        break
            
            elif y_step >= 0 and x_step < 0: 
                   
                for i in range(away_field):
                    try:
                        x_c_p += 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p -= 1
    
                        DAPI_away[i] = image[y_c_p, x_c_p]
                        DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                        channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                        channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                        channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                    
                        if y_c_p <= 0 or x_c_p <= 0:
                            break
                    
                    except IndexError:
                        break
                    
        else:
                away_field = search_field  # specify the depth of search field away from the lumen (in pixels)
                vec_away = np.zeros(away_field)
                DAPI_away = np.zeros(away_field)
                DAPI_filter_away = np.zeros(away_field)
                channel_3_away = np.zeros(away_field)
                channel_3_filter_away = np.zeros(away_field)
                channel_4_away = np.zeros(away_field)
                    
                if x_step == 0:
                    y_step_1 = abs(lumen_y_c_p - y_c_p) # y_step normalised to x-step-size = 1
                
                else:
                    y_step_1 = round(abs(y_step/x_step)) # y_step normalised to x-step-size = 1
                
                # define vectors from the nucleus towards the lumen
                
                if y_step <= 0 and x_step <= 0:
                    
                    for i in range(away_field):
                        try:
                            y_c_p += 1
                            
                            if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                                x_c_p += 1
        
                            DAPI_away[i] = image[y_c_p, x_c_p]
                            DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                            channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                            channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                            channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                        
                            if y_c_p <= 0 or x_c_p <= 0:
                                break
                        
                        except IndexError:
                            break
                               
                elif y_step >= 0 and x_step >= 0: 
                       
                    for i in range(away_field):
                        try:
                            y_c_p -= 1
                            
                            if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                                x_c_p -= 1
        
                            DAPI_away[i] = image[y_c_p, x_c_p]
                            DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                            channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                            channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                            channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                        
                            if y_c_p <= 0 or x_c_p <= 0:
                                break
                        
                        except IndexError:
                            break
                               
                elif y_step <= 0 and x_step >= 0: 
                       
                    for i in range(away_field):
                        try:
                            y_c_p += 1
                            
                            if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                                x_c_p -= 1
        
                            DAPI_away[i] = image[y_c_p, x_c_p]
                            DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                            channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                            channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                            channel_4_away[i] = image_channel_4[y_c_p, x_c_p] 
                        
                            if y_c_p <= 0 or x_c_p <= 0:
                                    break
                        
                        except IndexError:
                            break
                
                elif y_step >= 0 and x_step <= 0: 
                       
                    for i in range(away_field):
                        try:
                            y_c_p -= 1
                            
                            if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                                x_c_p += 1
                                
                            DAPI_away[i] = image[y_c_p, x_c_p]
                            DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                            channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                            channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                            channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                            
                            if y_c_p <= 0 or x_c_p <= 0:
                                break
                        
                        except IndexError:
                            break
                            
    else:
        
        away_field = search_field # specify the depth of search field towards the lumen (in pixels)
        vec_away = np.zeros(away_field)
        DAPI_away = np.zeros(away_field)
        DAPI_filter_away = np.zeros(away_field)
        channel_3_away = np.zeros(away_field)
        channel_3_filter_away = np.zeros(away_field)
        channel_4_away = np.zeros(away_field)
                  
        if y_step <= 0:
                
                for i in range(away_field):
                    if abs(y_c_p - y_c_p_0) % y_step == 0:
                        try:
                            y_c_p += 1
            
                            DAPI_away[i] = image[y_c_p, x_c_p]
                            DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                            channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                            channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                            channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                        
                            if y_c_p <= 0 or x_c_p <= 0:
                                break
                        
                        except IndexError:
                                break      
        else: 
                       
                for i in range(away_field):
                    if abs(y_c_p - y_c_p_0) % y_step == 0:
                        try:
                            y_c_p -= 1
            
                            DAPI_away[i] = image[y_c_p, x_c_p]
                            DAPI_filter_away[i] = nuclei_edges[y_c_p, x_c_p]
                            channel_3_away[i] = image_channel_3[y_c_p, x_c_p]
                            channel_3_filter_away[i] = cell_edges[y_c_p, x_c_p]
                            channel_4_away[i] = image_channel_4[y_c_p, x_c_p]
                        
                            if y_c_p <= 0 or x_c_p <= 0:
                                break
                        
                        except IndexError:
                                break
                   
    return DAPI_away, DAPI_filter_away, channel_3_away, channel_3_filter_away, channel_4_away
         
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w #define function to calculate the moving average for a sequence 'x' with a sliding window length 'w'

def min_vec_tow_lum(x_step, y_step, x_c_p, y_c_p, search_field, lumen_x_c_p, lumen_y_c_p, segment_mask):
    
    y_c_p_0 = y_c_p
    x_c_p_0 = x_c_p
            
    if x_step != 0:
    
        if abs(y_step/x_step) <= 1:
            
            tow_field = min(search_field, abs(x_step)) # specify the depth of search field towards the lumen (in pixels)
            vec_tow = np.zeros(tow_field)
            segments_tow = np.zeros(tow_field)
              
            if y_step == 0:   
                x_step_1 = abs(lumen_x_c_p - x_c_p)
                
            else:
                x_step_1 = round(abs(x_step/y_step)) # x-step normalised to y-step-size = 1
            
            # define vectors from the nucleus towards the lumen
                
            if y_step <= 0 and x_step <= 0:
                
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p -= 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p -= 1
    
                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                          
            elif y_step >= 0 and x_step >= 0: 
                   
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p += 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p += 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                        
            elif y_step <= 0 and x_step >= 0: 
                   
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p += 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p -= 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
            
            elif y_step >= 0 and x_step <= 0: 
                   
                for i in range(tow_field):
                    if abs(x_c_p - lumen_x_c_p) != 0:
                        x_c_p -= 1
                        
                        if abs(x_c_p - x_c_p_0) % x_step_1 == 0:
                            y_c_p += 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                   
        else:
            
            tow_field = min(search_field, abs(y_step)) # specify the depth of search field towards the lumen (in pixels)
            vec_tow = np.zeros(tow_field)
            segments_tow = np.zeros(tow_field)
            
            if x_step == 0:
                y_step_1 = abs(lumen_y_c_p - y_c_p) # y_step normalised to x-step-size = 1
            
            else:
                y_step_1 = round(abs(y_step/x_step)) # y_step normalised to x-step-size = 1
            
            # define vectors from the nucleus towards the lumen
                
            if y_step <= 0 and x_step < 0:
                
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p -= 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p -= 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                          
            elif y_step >= 0 and x_step > 0: 
                   
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p += 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p += 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                        
            elif y_step <= 0 and x_step > 0: 
                   
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p -= 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p += 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
            
            elif y_step >= 0 and x_step < 0: 
                   
                for i in range(tow_field):
                    if abs(y_c_p - lumen_y_c_p) != 0:
                        y_c_p += 1
                        
                        if abs(y_c_p - y_c_p_0) % y_step_1 == 0:
                            x_c_p -= 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                        
    else:
        
        tow_field = search_field # specify the depth of search field towards the lumen (in pixels)
        vec_tow = np.zeros(tow_field)
        segments_tow = np.zeros(tow_field)
                  
        if y_step <= 0:
                
                for i in range(tow_field):
                    if abs(y_c_p - y_c_p_0) % y_step == 0:
                        y_c_p -= 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
                              
        else: 
                       
                for i in range(tow_field):
                    if abs(y_c_p - y_c_p_0) % y_step == 0:
                        y_c_p += 1

                        segments_tow[i] = segment_mask[y_c_p, x_c_p]
    
    return segments_tow