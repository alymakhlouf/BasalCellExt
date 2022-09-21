# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:26:07 2021

@author: Lancaster Lab
"""

import math
import numpy as np
import statistics as stat
import matplotlib
import matplotlib.pyplot as plt

def distribution(array, bin_scale):

    # plot distance distribution
    fig, ax = plt.subplots()
    
    # the histogram of the data
    num_bins = math.ceil((max(array) - min(array))/bin_scale) # create bins of approximately size 10
    mu = np.mean(array)
    sigma = stat.stdev(array)
    n, bins, patches = ax.hist(array, num_bins, density=True)
    
    # add a 'best fit' line
    y_best = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y_best, '--')             
          
    # find the mode of spheroid_centroid-to-nuclear distance distribution
    hist_peak = np.max(np.histogram(array)[0])
    peak_idx = np.where(np.histogram(array)[0] == hist_peak)[0]
    mode = round((np.histogram(array)[1][peak_idx][0] + np.histogram(array)[1][peak_idx+1][0])/2)
    
    hist = np.histogram(array, num_bins)
    
    return fig, ax, mode, hist
