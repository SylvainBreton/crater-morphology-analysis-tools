#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:42:08 2020

@author: Sylvain
"""

###############################################################################
###############################################################################
################ This file gather many subprograms ############################
###############################################################################
###############################################################################


from math import *
import numpy as np
from crater_morphology_analysis_tools.CraterStatisticalMorphologyTools import modeling_sub as model_subs



###############################################################################
################################# Part II ####################################
############################ create  CSDFD ################################
###############################################################################
###############################################################################
###############################################################################




### This program comvolute a matrix with a gaussian filter defined by D_err_f and d_err
# The binning of the matrix must be proportional to the filter width
def convolute_proba(proba_CSDFD, Diam_bin, depth_bin, D_err_f=0.1, d_err=6):
    conv_CSDFD_proba=np.zeros((len(proba_CSDFD[:,0]),len(proba_CSDFD[0,:])))
    
    # #is the grid large enough?
    # #if not add bins
    # if 3*D_err_f*<Diam_bin[0]:  
    # if 3*D_err_f*>Diam_bin[-1]:
    # if >depth_bin[0]:
    # if >depth_bin[-1]:
        
    
    #Design the filter
    #we don't go further than 6 sigma    
    Diam_dist_max = np.min(np.where((Diam_bin-Diam_bin[0])>(3*D_err_f*sqrt(Diam_bin[0]*Diam_bin[1]))))+1
    Diam_dist_min = len(proba_CSDFD[:,0])-np.min(np.where((Diam_bin[-1]-Diam_bin)<(3*D_err_f*sqrt(Diam_bin[-1]*Diam_bin[-2]))))+1
    
    depth_dist_max= int((3*d_err)  /(depth_bin[1]-depth_bin[0]))+1
    
    
    
    
    kernel_filter = np.zeros((Diam_dist_max+Diam_dist_min, 2*depth_dist_max+1))
    
    
    

    X_bin_size = np.log10(Diam_bin[1]/Diam_bin[0])
    X_min=np.log10(Diam_bin[0])
    i_Diam=X_min + np.arange(0,Diam_dist_min+Diam_dist_max+1)* X_bin_size


    # rel_Diam_vec = (Diam_bin[0:Diam_dist_min+Diam_dist_max+1]-Diam_bin[Diam_dist_min])/(sqrt(Diam_bin[Diam_dist_min]*Diam_bin[Diam_dist_min+1])*D_err_f)
    rel_Diam_vec = (10**i_Diam-10**i_Diam[Diam_dist_min])/(sqrt(10**(i_Diam[Diam_dist_min]+i_Diam[Diam_dist_min+1]))*D_err_f)

    rel_depth_vec= np.arange(-depth_dist_max,depth_dist_max+1)*6/(2*depth_dist_max)
    
    
    
    Diam_proba  = np.exp(-1/2*(rel_Diam_vec)**2)
    depth_proba = np.exp(-1/2*(rel_depth_vec)**2)
    
    Diam_proba_int = (Diam_proba[1:]+Diam_proba[:-1])/2
    depth_proba_int = (depth_proba[1:]+depth_proba[:-1])/2
    
    
    # print('Diam_proba_int '+ str(np.sum(Diam_proba_int)))
    # print('depth_proba_int '+ str(np.sum(depth_proba_int)))
    Diam_grid, depth_grid = np.meshgrid(Diam_proba_int, depth_proba_int)
    kernel_filter = np.transpose(Diam_grid * depth_grid)/np.sum(Diam_grid * depth_grid)
    
    
    for i_Diam in range(0,len(proba_CSDFD[:,0])):
        for i_depth in range(0,len(proba_CSDFD[0,:])):
            if proba_CSDFD[i_Diam, i_depth] !=0:
                i_Diam_s=i_Diam-Diam_dist_min
                i_Diam_e=i_Diam+Diam_dist_max
            
                i_depth_s=i_depth-depth_dist_max
                i_depth_e=i_depth+depth_dist_max
                
                kernel_filter_temp=kernel_filter
                if i_Diam_s<0:
                    kernel_filter_temp=kernel_filter_temp[-i_Diam_s:,:]
                    i_Diam_s=0
                if i_Diam_e>len(proba_CSDFD[:,0]):
                    kernel_filter_temp=kernel_filter_temp[:-(i_Diam_e-len(proba_CSDFD[:,0])),:]
                    i_Diam_e=len(proba_CSDFD[:,0])
                if i_depth_s<0:
                    kernel_filter_temp=kernel_filter_temp[:,-i_depth_s:]
                    i_depth_s=0
                if i_depth_e>len(proba_CSDFD[0,:]):
                    kernel_filter_temp=kernel_filter_temp[:,:-(i_depth_e-len(proba_CSDFD[0,:]))]
                    i_depth_e=len(proba_CSDFD[0,:])
                
                    
                conv_CSDFD_proba[i_Diam_s:i_Diam_e,
                           i_depth_s:i_depth_e] = conv_CSDFD_proba[i_Diam_s:i_Diam_e,i_depth_s:i_depth_e]+(
                                   kernel_filter_temp * proba_CSDFD[i_Diam, i_depth])
                
    return conv_CSDFD_proba



#This program produce a CSDFD from a list of crater depth and diam
#The depth and diameter grids need to be given
#Robbins 2017 : PDP
def PDP_proba_1(Diam, depth, areas, Diam_bin, depth_bin, 
                               D_err_vec, d_err=6):
    proba=np.zeros((N_Diam,N_depth))
    
    #iterate through craters
    for i_crat in range(0,len(Diam)):
        if areas[i_crat]!=-1 and areas[i_crat]!=0:
            D_err = D_err_vec[i_crat]
            
            #compute an array with the distance to the crat in diam
            Diam_proba  = 1/(D_err*sqrt(2*pi))*np.exp(-1/2*((Diam[i_crat]-Diam_bin)/D_err)**2)
            depth_proba = 1/(d_err*sqrt(2*pi))*np.exp(-1/2*((depth[i_crat]-depth_bin)/d_err)**2)
            
            Diam_proba_int = (Diam_proba[1:]+Diam_proba[:-1])/2 * (Diam_bin[1:]-Diam_bin[:-1])
            depth_proba_int = (depth_proba[1:]+depth_proba[:-1])/2 * (depth_bin[1:]-depth_bin[:-1])
            
            Diam_grid, depth_grid = np.meshgrid(Diam_proba_int, depth_proba_int)
            
            proba = proba + np.transpose(Diam_grid * depth_grid)/areas[i_crat]
            
    return proba



#This program produce a CSDFD from a list of crater depth and diam
#The depth and diameter grids need to be given
#FASTER VERSION AS OF NOW
#Robbins 2017 : PDP
def PDP_proba_2(Diam, depth, Area, Diam_bin_lim, depth_bin_lim, D_err_f=0.1, d_err=6):
    count = np.histogram2d(Diam, depth, bins = [Diam_bin_lim, depth_bin_lim])[0] 

    proba_CSDFD = convolute_proba(count/Area, Diam_bin_lim, depth_bin_lim)
    
    return proba_CSDFD










#Use a kernel estimator from python on the crater pop
#Compute the probability on the grid
# mode can be 'linear', 'log', 'power_law'
import scipy.stats as st
import matplotlib.pyplot as plt


def kernel_estimator(Diam_crat, depth_crat, Diam_bin_lim, depth_bin_lim, mode='log', mirroring=False):
    
    if mode=='linear':
        Diam_bin_lim_temp= Diam_bin_lim
        Diam_crat_temp= Diam_crat
        
    if mode=='log':
        # Diam_bin_lim_temp=np.log10(Diam_bin_lim)
        # Diam_crat_temp   =np.log10(Diam_crat)
        Diam_bin_lim_temp=np.log2(Diam_bin_lim)
        Diam_crat_temp   =np.log2(Diam_crat)
        
        
    if mode=='power_law':
        Diam_bin_lim_temp= Diam_bin_lim**-3
        Diam_crat_temp   = Diam_crat**-3
                
    if mode=='isochron':
        Diam_bin_lim_temp= np.full(len(Diam_bin_lim), np.nan)
        Diam_crat_temp= np.full(len(Diam_crat), np.nan)
        for i_Diam in range(0,len(Diam_bin_lim)):
            Diam_bin_lim_temp[i_Diam]= model_subs.dif_Ivanov(Diam_bin_lim[i_Diam],1)
        for i_crat in range(0,len(Diam_crat)):
            Diam_crat_temp[i_crat]   = model_subs.dif_Ivanov(Diam_crat[i_crat],1)
            
    
    # depth_bin_lim_temp=np.log2(depth_bin_lim)
    # depth_crat_temp   =np.log2(depth_crat)
    # depth_bin_lim_temp = depth_bin_lim**(-2)
    # depth_crat_temp    = depth_crat**(-2)
    # depth_bin_lim_temp = depth_bin_lim**(1/2)
    # depth_crat_temp    = depth_crat**(1/2)
    depth_bin_lim_temp = depth_bin_lim
    depth_crat_temp    = depth_crat
    
    
    
    
    if mirroring:
        #try mirroring
        depth_crat_temp=np.concatenate((depth_crat_temp,depth_crat_temp,-depth_crat_temp,-depth_crat_temp))
        Diam_crat_temp=np.concatenate((Diam_crat_temp,2*np.min(Diam_crat_temp)-Diam_crat_temp,2*np.min(Diam_crat_temp)-Diam_crat_temp,Diam_crat_temp))
        ###########

    
    
    # xx, yy = np.meshgrid(Diam_bin_lim_temp, depth_bin_lim)
    xx, yy = np.meshgrid((Diam_bin_lim_temp[1:]+Diam_bin_lim_temp[:-1])/2, 
                         (depth_bin_lim_temp[1:]+depth_bin_lim_temp[:-1])/2)
    
    
    Diam_bin_size  = np.absolute(Diam_bin_lim_temp[1:]-Diam_bin_lim_temp[:-1])
    depth_bin_size = np.absolute(depth_bin_lim_temp[1:] - depth_bin_lim_temp[:-1])
    dxx, dyy = np.meshgrid(Diam_bin_size, depth_bin_size)
    
    
    # #use d/D to perform the kernel estimator
    # depth_crat_temp = depth_crat/Diam_crat
    # yy  = yy / 2**xx
    # dyy = dyy/ 2**xx
    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([Diam_crat_temp, depth_crat_temp])
    kernel = st.gaussian_kde(values)
    
    f = (np.reshape(kernel(positions).T, xx.shape) * np.size(Diam_crat_temp))
    
    # proba_kernel = np.transpose((f[1:,1:]+f[:-1,1:]+f[1:,:-1]+f[:-1,:-1])/4 * dxx * dyy)
    proba_kernel = np.transpose(f* dxx * dyy)
    
    return proba_kernel




#Use an adaptative kernel estimator
#The kernel bandwidth is inversly proportional to the local density**alpha
from awkde import GaussianKDE

import awkde_modified as test



def adaptative_kernel_estimator(Diam_crat, depth_crat, Diam_bin_lim, depth_bin_lim, alpha_kde, mirroring=False):
    # depth_bin_lim_temp=np.log2(depth_bin_lim)
    # depth_crat_temp   =np.log2(depth_crat)
    depth_bin_lim_temp=depth_bin_lim
    depth_crat_temp   =depth_crat
    
    Diam_bin_lim_temp=np.log2(Diam_bin_lim)
    Diam_crat_temp   =np.log2(Diam_crat)
    # Diam_bin_lim_temp= Diam_bin_lim
    # Diam_crat_temp= Diam_crat
    # Diam_bin_lim_temp= Diam_bin_lim**-3
    # Diam_crat_temp   = Diam_crat**-3
    
    
    
    XX, YY = np.meshgrid((Diam_bin_lim_temp[1:]+Diam_bin_lim_temp[:-1])/2, 
                             (depth_bin_lim_temp[1:]+depth_bin_lim_temp[:-1])/2)
    
    Diam_bin_size  = np.abs(Diam_bin_lim_temp[1:]-Diam_bin_lim_temp[:-1])
    depth_bin_size = depth_bin_lim_temp[1:] - depth_bin_lim_temp[:-1]
    dxx, dyy = np.meshgrid(Diam_bin_size, depth_bin_size)
    
    
    ##########################
    # use d/D to perform the kernel estimator (ie standardize depth)
    # depth_crat_temp = depth_crat_temp/(Diam_crat*1000)
    # YY  = YY / 2**XX /1000
    # dyy = dyy/ 2**XX /1000
    
    
    

    #Standardize the data
    # norm_depth=np.std(depth_crat_temp)
    # depth_crat_temp = depth_crat_temp /norm_depth
    # YY    = YY     /norm_depth
    # dyy   = dyy    /norm_depth
    
    # norm_Diam=np.std(Diam_crat_temp)
    # Diam_crat_temp= Diam_crat_temp /norm_Diam
    # XX   = XX    /norm_Diam
    # dxx  = dxx   /norm_Diam
    

    #try mirroring
    if mirroring:
        depth_crat_temp=np.concatenate((depth_crat_temp,depth_crat_temp,-depth_crat_temp,-depth_crat_temp))
        Diam_crat_temp=np.concatenate((Diam_crat_temp,2*np.min(Diam_crat_temp)-Diam_crat_temp,2*np.min(Diam_crat_temp)-Diam_crat_temp,Diam_crat_temp))
    ###########

    
    sample = np.vstack((Diam_crat_temp, depth_crat_temp)).T
    
    
    #perform the aKDE
    # kde = GaussianKDE(glob_bw="scott", alpha=alpha_kde, diag_cov=True)
    kde = test.GaussianKDE(glob_bw="scott", alpha=alpha_kde, diag_cov=False)
    kde.fit(sample)
    grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
    zz = kde.predict(grid_pts)
    ZZ = zz.reshape(XX.shape)  * np.size(Diam_crat_temp)
    
    
    
    adaptative_KDE_proba = np.transpose(ZZ * dxx * dyy)
    
    return adaptative_KDE_proba












