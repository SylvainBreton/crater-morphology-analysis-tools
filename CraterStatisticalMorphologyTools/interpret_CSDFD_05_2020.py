#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:34:14 2020

@author: Sylvain
"""

###############################################################################
###############################################################################
################ This file gather many subprograms ############################
###############################################################################
###############################################################################


from math import *
import numpy as np
from sub_programs import modeling_sub_05_2020 as model_subs



import shapely.geometry as geometry
from sub_programs import alpha_shape


###############################################################################
################################# Part IV #####################################
############################ Interpret  CSDFD #################################
###############################################################################
###############################################################################
###############################################################################

def obliteration(proba, Diam_bin, depth_bin, mask_proba, t_param=[4.8,15]):
    nb_bin_diam =np.shape(proba)[0]
    nb_bin_depth=np.shape(proba)[1]
    
    
    
    #create a cumulative density grid in depth
    Density_grid_cum     = np.zeros(np.shape(proba))
    for i_Diam in range(0,nb_bin_diam):        
        Density_grid_cum[i_Diam,:] = np.cumsum(proba[i_Diam,::-1])[::-1]
        
        
         
    
    
    
    #   Compute obliteration
    #Maximum age
    t_max=t_param[0]          #in Gy
    #Number of time steps
    Nb_time_step=t_param[1]

    
    
    #Decrease time step with age (because impact rate is increasing)
    # age_bin =  (t_max+1) - (t_max+1)**(np.arange(Nb_time_step,-1,-1)/Nb_time_step)
    # age_bin[0]=0.
    # age_vec=(age_bin[1:]+age_bin[:-1])/2
    
    step=0.1
    age_bin=[0.]
    i_age=0
    while age_bin[-1]<t_max:
        age_bin.append(age_bin[-1]+ step*log(1.000001+t_max-age_bin[-1]))
    
    age_bin=np.array(age_bin)
    Nb_time_step=np.size(age_bin)-1
    
    
    Erosion_grid=np.zeros((Nb_time_step,nb_bin_diam))
    mask_Erosion=np.full((Nb_time_step,nb_bin_diam), 1.)
    Depth_grid=np.full((Nb_time_step+1,nb_bin_diam),np.nan)
    mask_Erosion1=np.full((Nb_time_step+1,nb_bin_diam), 1.)
      
    
    for i_Diam in range(0, nb_bin_diam):
        X_inf = Diam_bin[i_Diam]
        X_sup = Diam_bin[i_Diam+1]
                
        if Density_grid_cum[i_Diam,0]!=0 and Density_grid_cum[i_Diam,0]==Density_grid_cum[i_Diam,0]:
            i_depth_low_prev=np.max(np.where(Density_grid_cum[i_Diam,:]==Density_grid_cum[i_Diam,:]))
            
            
            for i_time in range(0,Nb_time_step+1):     
                #Compute the model density for this age and depth
                density_theo_age=(model_subs.Ivanov(X_inf,age_bin[i_time])-model_subs.Ivanov(X_sup,age_bin[i_time]))
                
                #find where the cumulative depth density is equal to theoritical densities
                if Density_grid_cum[i_Diam,0]>=density_theo_age and Density_grid_cum[i_Diam,i_depth_low_prev]<=density_theo_age:
                    i_depth_high=np.max(np.where(density_theo_age<=Density_grid_cum[i_Diam,:]))
                    i_depth_low=np.min(np.where(density_theo_age>=Density_grid_cum[i_Diam,:]))
                    
                    
                    #Interpolate the depth linearly in the bin
                    if i_depth_high!=i_depth_low:
                        Where_in_bin=((Density_grid_cum[i_Diam,i_depth_low]-density_theo_age)/
                               (Density_grid_cum[i_Diam,i_depth_low]-Density_grid_cum[i_Diam,i_depth_high]))
                    
                    #Case where we are exactly on the bin edge
                    else:
                        Where_in_bin=0
                        

                    Depth_grid[i_time, i_Diam]=depth_bin[i_depth_low] - (
                                                        depth_bin[i_depth_low]-depth_bin[i_depth_high])*Where_in_bin
                    
                    if mask_proba[i_Diam, i_depth_low]!=1 and mask_proba[i_Diam, i_depth_high]!=1:
                        mask_Erosion1[i_time, i_Diam] = np.nan

    for i_time in range(0,Nb_time_step):
        for i_Diam in range(0, nb_bin_diam):            
            Erosion_grid[i_time, i_Diam]=(Depth_grid[i_time, i_Diam]-Depth_grid[i_time+1, i_Diam])/(
                                                           age_bin[i_time+1]- age_bin[i_time])
            
            if mask_Erosion1[i_time, i_Diam]!=1 and mask_Erosion1[i_time+1, i_Diam]!=1:
                mask_Erosion[i_time, i_Diam]=np.nan


                
    
    return Erosion_grid, Depth_grid, age_bin, mask_Erosion





#Create a mask matrix that remoove the tail of the distribution
# param_cumuN=[Area, nb_lim]
# param_alphaShp=[Diam_bin_lim, depth_bin_lim, [Diam, depth]]
# param_errorl=[r_error, err_lim]
def create_mask(methods, proba, param_cumuN, param_sqrtcumuN, param_alphaShp, param_errorl):
    mask_matrix = np.full(np.shape(proba), np.nan)
    
    if "cumu_N" in methods:
        Area=param_cumuN[0]
        nb_lim=param_cumuN[1]
        
        cumu1=np.full(np.shape(proba),np.nan)
        cumu2=np.full(np.shape(proba),np.nan)
        
        for i in range(0, np.shape(proba)[0]):
            cumu1[i,:]=np.cumsum(proba[i,:])
            cumu2[i,::-1]=np.cumsum(proba[i,::-1])
            
        for i in range(0, np.shape(proba)[1]):
            cumu2[:,i]=np.cumsum(cumu2[:,i])
            cumu1[::-1,i]=np.cumsum(cumu1[::-1,i])
            
        mask_matrix[np.where(cumu2*Area<nb_lim)] = 1
        mask_matrix[np.where(cumu1*Area<nb_lim)] = 1
        
    if "cumu_sqrt_N" in methods:
        Area=param_sqrtcumuN[0]
        lim=param_sqrtcumuN[1]
        
        cumu1=np.full(np.shape(proba),np.nan)
        cumu2=np.full(np.shape(proba),np.nan)
            
        for i in range(0, np.shape(proba)[0]):
            cumu2[i,:]=np.cumsum(proba[i,:])
            cumu1[i,::-1]=np.cumsum(proba[i,::-1])
            
        # mask_matrix[np.where(np.sqrt(cumu2*Area)<lim)] = 1
        mask_matrix[np.where(np.sqrt(cumu1*Area)<lim)] = 1
    
    
    if "alpha_shp" in methods:
        Diam_bin_lim=param_alphaShp[0]
        depth_bin_lim=param_alphaShp[1]
        Crats=param_alphaShp[2]
        
        crat_log_array=np.transpose(Crats)
        points_crat = geometry.MultiPoint(crat_log_array)
        cascaded_union, edge_points=alpha_shape.alpha_shape(points_crat,0.001)
        
        
        for i_Diam in range(0, np.shape(proba)[0]):
            for i_depth in range(0, np.shape(proba)[1]):
                x_coord = (Diam_bin_lim[i_Diam+1]+Diam_bin_lim[i_Diam])/2
                y_coord = (depth_bin_lim[i_depth+1]+depth_bin_lim[i_depth])/2
                
                point_mat= geometry.Point(x_coord , y_coord)
                
                if not(point_mat.within(cascaded_union)):
                    mask_matrix[i_Diam, i_depth]=1
                    
        
    if "error_limit" in methods:
        r_error= param_errorl[0]
        err_lim= param_errorl[1]
        
        
        mask_matrix[np.where(r_error>err_lim)]=1
                
    
    return mask_matrix







##### Transform a 2D obliteration grid (age, diam) in a 1D array obliteration=f(age)
def get_1D_age_ob(age_vec, Erosion_grid, mask_Erosion, stat_lim, Diam_range=[np.nan, np.nan]):
    mean_vec=np.full(np.size(age_vec), np.nan)
    med_vec =np.full(np.size(age_vec), np.nan)
    std_vec =np.full(np.size(age_vec), np.nan)
            
        
    for i_age in range(0, np.size(age_vec)):
        if np.size(np.where(mask_Erosion[i_age,:]!=mask_Erosion[i_age,:]))>stat_lim:
            mean_vec[i_age]=np.nanmean(Erosion_grid[i_age, np.where(mask_Erosion[i_age,:]!=mask_Erosion[i_age,:])])
            med_vec[i_age] =np.nanmedian(Erosion_grid[i_age, np.where(mask_Erosion[i_age,:]!=mask_Erosion[i_age,:])])
            std_vec[i_age] =np.nanstd(Erosion_grid[i_age, np.where(mask_Erosion[i_age,:]!=mask_Erosion[i_age,:])])

    return mean_vec, med_vec, std_vec

















