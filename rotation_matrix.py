# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:48:58 2022

@author: Aly Makhlouf
"""

import numpy as np

def matrix(eig_1, eig_2, eig_3):

    new_zaxis = eig_1
    new_yaxis = eig_2
    new_xaxis = eig_3
    
    # new axes:
    nnz, nny, nnx = new_zaxis, new_yaxis, new_xaxis
    # old axes:
    noz, noy, nox = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, -1)
    
    # define rotation matrix
    top = [np.dot(nnz, n) for n in [noz, noy, nox]]
    mid = [np.dot(nny, n) for n in [noz, noy, nox]]
    bot = [np.dot(nnx, n) for n in [noz, noy, nox]]
    
    rotation_matrix = np.hstack((zn, yn, xn))
    
    return rotation_matrix