#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:12:31 2020

@author: Sylvain
"""

###############################################################################
###############################################################################
################ This file gather many subprograms ############################
###############################################################################
###############################################################################


from math import *
import numpy as np


###############################################################################
################################# Part III ####################################
############################ manipulate  CSDFD ################################
###############################################################################
###############################################################################
###############################################################################


#from CSDFD to CSFD
def CSDFD_to_CSFD(CSDFD, depth_bin):
    CSFD = np.full(np.size(CSDFD[:,0]), np.nan)
    for i_Diam in range(0, np.size(CSDFD[:,0])):
        CSFD[i_Diam] = np.sum(CSDFD[i_Diam,:]*(depth_bin[1:]-depth_bin[:-1])/1000)
    return CSFD


def CSDFD_to_proba(CSDFD, Diam_bin, depth_bin):
    Diam_grid, depth_grid = np.meshgrid(Diam_bin, depth_bin)
    proba = CSDFD * (np.transpose(Diam_grid[:-1,1:]-Diam_grid[:-1,:-1]) * 
                np.transpose(depth_grid[1:,:-1]-depth_grid[:-1,:-1])/1000)
    return proba


def proba_to_CSDFD(proba, Diam_bin, depth_bin):
    Diam_grid, depth_grid = np.meshgrid(Diam_bin, depth_bin)
    CSDFD = proba / (np.transpose(Diam_grid[:-1,1:]-Diam_grid[:-1,:-1]) * 
                np.transpose(depth_grid[1:,:-1]-depth_grid[:-1,:-1])/1000)
    return CSDFD


def to_Diam_1D(matrix):
    N_Diam=np.size(matrix[:,0])
    vector=np.zeros((1,N_Diam))
    for i_Diam in range(0, N_Diam):
        vector[0,i_Diam]=np.sum(matrix[i_Diam,:])
    return vector



























