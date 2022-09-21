# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:28:18 2021

@author: Aly Makhlouf
"""

import math
import numpy as np
import cv2

# compute roundness of objects (nuclei)
def compute_roundness(label_image):
    contours, hierarchy = cv2.findContours(np.array(label_image, dtype=np.uint8), cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return a / b

# custom function returns arctan() in a quadrant-specific manner
myatan = lambda x,y: np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))