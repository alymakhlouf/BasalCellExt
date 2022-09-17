# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:46:47 2021

@author: Aly Makhlouf
"""

import numpy as np

def find(lumen_edge, segment_mask):
    
    pixels = np.zeros(8 * len(np.where(lumen_edge != 0)[0])) # find 8 pixel neighbours of every pixel in the edge-filtered lumen segment
    
    lumen_edge_y = np.where(lumen_edge != 0)[0]
    lumen_edge_x = np.where(lumen_edge != 0)[1]
    
    for idx, value in enumerate(lumen_edge_y):
        try:
            pixels[idx] = segment_mask[value, lumen_edge_x[idx] + 1]
        except IndexError:
            pass
        try:
            pixels[idx + 1] = segment_mask[value, lumen_edge_x[idx] - 1]
        except IndexError:
            pass
        try:
            pixels[idx + 2] = segment_mask[value + 1, lumen_edge_x[idx]]
        except IndexError:
            pass
        try:
            pixels[idx + 3] = segment_mask[value + 1, lumen_edge_x[idx] + 1]
        except IndexError:
            pass
        try:
            pixels[idx + 4] = segment_mask[value + 1, lumen_edge_x[idx] - 1]
        except IndexError:
            pass
        try:
            pixels[idx + 5] = segment_mask[value - 1, lumen_edge_x[idx]]
        except IndexError:
            pass
        try:
            pixels[idx + 6] = segment_mask[value - 1, lumen_edge_x[idx] + 1]
        except IndexError:
            pass
        try:
            pixels[idx + 7] = segment_mask[value - 1, lumen_edge_x[idx] - 1]
        except IndexError:
            pass
        
    neighbours = np.unique(pixels, return_counts = True)
    
    return neighbours