#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:24:54 2020

@author: Sylvain
"""

# Date from a list of crater diameters
# uses a list of diameter in km, a range of the fit and a reference area
# Uses Michael 2016 Poisson fit
# cite "Planetary surface dating from crater size-frequency distribution measurements: Poisson timing analysis", G.G. Michael, T. Kneissl, A. Neesemann, Icarus, https://doi.org/10.1016/j.icarus.2016.05.019


import numpy as np
import matplotlib.pyplot as plt
from math import *

import os
from osgeo import ogr

# Param
range_Diam = [0.6, 15]
area = 41286918825.50000000000
folder = '/home/user/Bureau/Recherche/ELM/dating/'
file = 'CRATER_Hercules.shp'

file_path = folder + file

#####Model
# Mars
# Ivanov (2001)
# Coef_PF = [-3.383677, -3.197453 , 1.256814 , 7.915374*10**(-1) ,
#                 -4.860814*10**(-1) , -3.630098*10**(-1) , 1.015683*10**(-1) ,
#                 6.755923*10**(-2) , -1.180639*10**(-2) , -4.753462*10**(-3) ,
#                 6.232845*10**(-4) , 5.805492*10**(-5)]

# #Ivanov (2001)
# Coef_chrono = [2.68*10**(-14) , 6.93 , 0 , 4.13*10**(-4)]


# Moon
# Neukum (2001)
Coef_PF = [-3.0768, -3.557528, 0.781027, 1.021521,
           -0.156012, -0.444058, 0.019977, 0.086850,
           -0.005874, -0.006809, 8.25 * 10 ** (-4), 5.54 * 10 ** (-5)]
# Neukum (2001)
Coef_chrono = [5.44 * 10 ** (-14), 6.93, 0, 8.38 * 10 ** (-4)]

################################################################################
####### Import the crater shapefile as a np.array of diameters
# Open shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')
crater_shp = driver.Open(file_path, 1)

crat_layer = crater_shp.GetLayer()
featureCount = crat_layer.GetFeatureCount()

for i in range(1, crat_layer.GetFeature(0).GetFieldCount()):
    field = crat_layer.GetFeature(0).GetDefnRef().GetFieldDefn(i).GetName()

# Fill with appropriate name
diam_field = 'diam_km'
area_field = 'area_sqkm'

Diam_crat = np.zeros(featureCount)
crat_areas = np.zeros(featureCount)
for index in range(0, featureCount):
    crater = crat_layer.GetFeature(index)

    Diam_crat[index] = crater.GetField(diam_field)
    crat_areas[index] = crater.GetField(area_field)

################################################################################
################################################################################


###############################################################################
################################    Dating    #################################
###############################################################################

# Compute a CSFD with PDP according to Robbins 2017
# take crater areas to apply non sparse correction

def fast_CSFD(Diam, Diam_bin, crat_areas, D_err_f=0.1):
    proba = np.zeros(len(Diam_bin) - 1)
    # iterate through craters
    for i_crat in range(0, len(Diam)):
        D_err = D_err_f * Diam[i_crat]
        # compute an array with the distance to the crat in diam
        Diam_proba = 1 / (D_err * sqrt(2 * pi)) * np.exp(-1 / 2 * ((Diam[i_crat] - Diam_bin) / D_err) ** 2)
        Diam_proba_int = (Diam_proba[1:] + Diam_proba[:-1]) / 2 * (Diam_bin[1:] - Diam_bin[:-1])
        proba = proba + Diam_proba_int / crat_areas[i_crat]

    CSFD = proba / (Diam_bin[1:] - Diam_bin[:-1])

    return CSFD

# compute the isochron
def compute_isochron(Diam_bin, age):
    PF = np.full(len(Diam_bin), np.nan)

    for i_Diam in range(0, len(Diam_bin)):
        PF[i_Diam] = np.sum(Coef_PF[1:] * np.log10(Diam_bin[i_Diam]) ** np.arange(1, len(Coef_PF)))

    chrono_vec = (Coef_chrono[0] * (np.exp(Coef_chrono[1] * age) - 1)
                  + Coef_chrono[3] * age)

    return chrono_vec * (10 ** PF)



def Dating_Poisson(Diam_crat, range_Diam, area, Coef_PF, Coef_chrono):
    Diam_crat = Diam_crat[np.where((Diam_crat > range_Diam[0]) & (Diam_crat < range_Diam[1]))]

    dt = 0.001
    # We test different ages from 0 to 5 Gy
    age_tested_vec = np.arange(dt, 5, dt)

    proba_age_vec = np.zeros(len(age_tested_vec))

    PF_Diam_min = np.sum(Coef_PF[1:] * np.log10(range_Diam[0]) ** np.arange(1, len(Coef_PF)))
    PF_Diam_max = np.sum(Coef_PF[1:] * np.log10(range_Diam[1]) ** np.arange(1, len(Coef_PF)))

    chrono_vec = (Coef_chrono[0] * (np.exp(Coef_chrono[1] * age_tested_vec) - 1)
                  + Coef_chrono[3] * age_tested_vec)

    # compute the probability of each age Michael 2016
    proba_log = (-area * chrono_vec * (10 ** PF_Diam_min - 10 ** PF_Diam_max) +
                 np.log(chrono_vec) * len(Diam_crat))

    # python can't handle numbers above exp(700) and below exp(-700)
    # so we remoove the lowest values
    max_proba_log = np.max(proba_log)
    proba_log = proba_log + 700 - max_proba_log
    proba_age_vec[np.where(proba_log > -700)] = (np.exp(proba_log[np.where(proba_log > -700)])
                                                 / (np.sum(np.exp(proba_log[np.where(proba_log > -700)])) * dt))

    return age_tested_vec, proba_age_vec


# Compute the probability of ages from 0 to 5 Gy
age_vec, proba_age_vec = Dating_Poisson(Diam_crat, range_Diam, crat_areas, Coef_PF, Coef_chrono)

# Compute a cumulative proba
proba_cumu = np.cumsum(proba_age_vec) / np.sum(proba_age_vec)

# Get The median and the 1 sigma enveloppe
i_med = np.min(np.where(proba_cumu > 0.5))

i_sigma_inf = np.min(np.where(proba_cumu > 0.5 - 0.34))
i_sigma_sup = np.min(np.where(proba_cumu > 0.5 + 0.34))

crat_numb = np.size(Diam_crat[np.where((Diam_crat > range_Diam[0]) & (Diam_crat < range_Diam[1]))])

print('Median model age is ' + str(age_vec[i_med]) + ' Gy')
print('From ' + str(crat_numb) + ' craters')
print('On ' + str(area) + ' km2')

#########################################################################
# Now make a figure of the probability density
Prob_dens_fig = plt.figure()
Prob_dens_sub = Prob_dens_fig.add_subplot(111)

Prob_dens_sub.plot(age_vec, proba_age_vec, color='#e07109')
Prob_dens_sub.fill_between(age_vec[i_sigma_inf:i_sigma_sup], proba_age_vec[i_sigma_inf:i_sigma_sup], alpha=0.2,
                           color='#e07109')
Prob_dens_sub.plot([age_vec[i_med], age_vec[i_med]], [0, proba_age_vec[i_med]], color='#e07109')

Prob_dens_sub.tick_params(direction='in', top=True, which='both', labelsize=13)
Prob_dens_sub.set_ylabel(r'Probability density', fontsize=14)
Prob_dens_sub.set_xlabel(r'Age (Gy)', fontsize=14)


#########################################################################
# Now make a figure of the CSFD (to best choose diam range for example)

# Build diameter grid
N_Diam = 4000
X_min = np.log10(0.68 * np.min(Diam_crat))
X_max = np.log10(1.32 * np.max(Diam_crat))
X_bin_size = (X_max - X_min) / (N_Diam)
i_Diam = X_min + np.arange(0, N_Diam + 1) * X_bin_size
Diam_bin = 10 ** i_Diam
Diam_vec = 10 ** ((i_Diam[1:] + i_Diam[:-1]) / 2)


# We compute the CSFD of your crater pop
CSFD = fast_proba(Diam_crat, Diam_bin, crat_areas, D_err_f = 0.1) / area / (Diam_bin[1:] - Diam_bin[:-1])

Age_factor = (Coef_chrono[0] * (np.exp(Coef_chrono[1] * age_vec[i_med]) - 1)
              + Coef_chrono[3] * age_vec[i_med])

Best_isochron_cumu = np.full(N_Diam + 1, np.nan)
for i_Diam in range(0, N_Diam + 1):
    Best_isochron_cumu[i_Diam] = Age_factor * 10 ** np.sum(
        Coef_PF[1:] * np.log10(Diam_bin[i_Diam]) ** np.arange(1, len(Coef_PF)))

Best_isochron = -(Best_isochron_cumu[1:] - Best_isochron_cumu[:-1]) / (Diam_bin[1:] - Diam_bin[:-1])

# now let's do the figure
CSFD_fig = plt.figure()
CSFD_sub = CSFD_fig.add_subplot(111)

CSFD_sub.plot(Diam_vec, Best_isochron, color='g')
CSFD_sub.plot(Diam_vec, CSFD, color='b')

CSFD_sub.set_xscale('log')
CSFD_sub.set_yscale('log')

CSFD_sub.tick_params(direction='in', top=True, which='both', labelsize=13)
CSFD_sub.set_ylabel(r'Crater density (km-3)', fontsize=14)
CSFD_sub.set_xlabel(r'Diameter (km)', fontsize=14)
