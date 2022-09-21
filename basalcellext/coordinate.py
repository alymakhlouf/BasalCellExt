# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:41:29 2021

@author: Aly Makhlouf
"""

import numpy as np

def rotate(eig_1, eig_2, eig_3, vector):

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
    
    def newit(vec):
        zn = sum([p*q for p,q in zip(top, vec)])
        yn = sum([p*q for p,q in zip(mid, vec)])
        xn = sum([p*q for p,q in zip(bot, vec)])
        return np.hstack((zn, yn, xn))
    
    vec_rotated = newit(vector)
    
    return vec_rotated


def rotation_matrix(eig_1, eig_2, eig_3):

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
    
    rotation_matrix = np.vstack((top, mid, bot))
    
    return rotation_matrix