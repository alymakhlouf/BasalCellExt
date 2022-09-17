# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:11:10 2021

@author: Aly Makhlouf
"""

import numpy as np
import matplotlib.pyplot as plt

def save(image, filename, output, directory):
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=image.min(), vmax=image.max())
    
    # map the normalized data to colors
    image = cmap(norm(image))

    
    # save the image
    plt.imsave(directory + output + '_' + filename + '.png', image)    
