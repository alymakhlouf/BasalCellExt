# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:32:49 2021

@author: Aly Makhlouf
"""

import numpy as np
from skimage.measure import label as connected_component

def find(image, label, window_width):
    
    binary = connected_component(image == label) # isolate a single object as a connected component
    position = np.where(binary == 1) # return the positions of all pixels belonging to object, along all dimensions

    max_z = np.bincount(position[0]).argmax() # return z-slice that has the maximum number of object pixels
    max_y = np.bincount(position[1]).argmax() # return y-slice that has the maximum number of object pixels
    max_x = np.bincount(position[2]).argmax() # return x-slice that has the maximum number of object pixels

    if max_y < 10 or max_x < 10 or max_y > 1014 or max_x > 1014:
        y_min = y_max = x_min = x_max = -1 # to break and skip the object
        return y_min, y_max, x_min, x_max, binary

    # specify region of interest (roi)
    
    search = 10
        
    y_min = np.sort(np.where(binary[max_z,:,max_x-search:max_x+search]==1)[0])[0] - window_width
    y_max = np.sort(np.where(binary[max_z,:,max_x-search:max_x+search]==1)[0])[-1] + window_width
    
    x_min = np.sort(np.where(binary[max_z,max_y-search:max_y+search]==1)[1])[0] - window_width
    x_max = np.sort(np.where(binary[max_z,max_y-search:max_y+search]==1)[1])[-1] + window_width
    
    return y_min, y_max, x_min, x_max, binary