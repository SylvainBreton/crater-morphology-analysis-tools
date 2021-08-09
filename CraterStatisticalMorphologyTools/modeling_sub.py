#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:46:57 2020

@author: Sylvain
"""

###############################################################################
###############################################################################
################ This file gather many subprograms ############################
###############################################################################
###############################################################################


###############################################################################
################################### Part I ####################################
######################## tools to create synthetic data #######################
###############################################################################
###############################################################################
###############################################################################


from math import *
import numpy as np


##############################################################################
##############################################################################
# Production functions
#####################
# They are the base of modeling crater pop
#####################
# diameter in km
# age in Gyr
# number is in /km2
#####################


###########################################################################
# Ivanov (2001) 
# Conpute the expected density of crater > diameter for a given age
def Ivanov(diameter, age):
    if age < 0 or age > 5:
        raise Exception('age should be positive and less than 5 Gy')
    elif age == 0:
        return 0
    else:
        coef_pf = [-3.383677, -3.197453, 1.256814, 7.915374 * 10 ** (-1),
                   -4.860814 * 10 ** (-1), -3.630098 * 10 ** (-1), 1.015683 * 10 ** (-1),
                   6.755923 * 10 ** (-2), -1.180639 * 10 ** (-2), -4.753462 * 10 ** (-3),
                   6.232845 * 10 ** (-4), 5.805492 * 10 ** (-5)]
        coef_chrono = [2.68 * 10 ** (-14), 6.93, 0, 4.13 * 10 ** (-4)]
        time_dep = coef_chrono[0] * (exp(coef_chrono[1] * age) - 1) + coef_chrono[3] * age
        tot = log(time_dep, 10)
        if diameter <= 0:
            raise Exception('diameter should be positive')
        else:
            for i_coef in range(1, len(coef_pf)):
                tot = tot + coef_pf[i_coef] * (log(diameter, 10)) ** i_coef

    if tot < -300:
        return 0.
    elif tot > 300:
        raise Exception('Increase the minimum diameter')
    else:
        return 10 ** tot


###########################################################################
# derivate Ivanov (2001) on diameter
def dif_Ivanov(diameter, age):
    if age <= 0:
        number = 0
    else:
        coef_pf = [-3.383677, -3.197453, 1.256814, 7.915374 * 10 ** (-1),
                   -4.860814 * 10 ** (-1), -3.630098 * 10 ** (-1), 1.015683 * 10 ** (-1),
                   6.755923 * 10 ** (-2), -1.180639 * 10 ** (-2), -4.753462 * 10 ** (-3),
                   6.232845 * 10 ** (-4), 5.805492 * 10 ** (-5)]
        coef_chrono = [2.68 * 10 ** (-14), 6.93, 0, 4.13 * 10 ** (-4)]
        time_dep = coef_chrono[0] * (exp(coef_chrono[1] * age) - 1) + coef_chrono[3] * age
        tot1 = log(time_dep, 10)
        tot2 = 0
        if diameter > 0:
            for i_coef in range(1, len(coef_pf)):
                tot1 = tot1 + coef_pf[i_coef] * (log(diameter, 10)) ** i_coef
                tot2 = tot2 + coef_pf[i_coef] * i_coef * (log(diameter, 10)) ** (i_coef - 1)
        number = - 10 ** tot1 / diameter * tot2
    return number


###########################################################################
# derivate Ivanov (2001) on diameter and age
# i.e. conpute the expected increase in density of diameter+Ddiam > crater > diameter
# in a time dt for a given age
def der_Ivanov(diameter, age):
    coef_pf = [-3.383677, -3.197453, 1.256814, 7.915374 * 10 ** (-1),
               -4.860814 * 10 ** (-1), -3.630098 * 10 ** (-1), 1.015683 * 10 ** (-1),
               6.755923 * 10 ** (-2), -1.180639 * 10 ** (-2), -4.753462 * 10 ** (-3),
               6.232845 * 10 ** (-4), 5.805492 * 10 ** (-5)]
    coef_chrono = [2.68 * 10 ** (-14), 6.93, 0, 4.13 * 10 ** (-4)]
    d_time_dep = coef_chrono[0] * coef_chrono[1] * exp(coef_chrono[1] * age) + coef_chrono[3]
    tot1 = coef_pf[1] * (log(diameter, 10))
    tot2 = coef_pf[1]
    if diameter > 0:
        for i_coef in range(2, len(coef_pf)):
            tot1 = tot1 + coef_pf[i_coef] * (log(diameter, 10)) ** i_coef
            tot2 = tot2 + i_coef * coef_pf[i_coef] * (log(diameter, 10)) ** (i_coef - 1)
    number = -d_time_dep * tot2 * 10 ** (tot1) * log(10)
    return number


##############################################################################
##############################################################################
# Scaling functions
# give the initial depth of a crater
# Diam in km
# depth in m
#####################
def Garvin(Diam):
    if Diam < 7:
        depth = 0.21 * (Diam) ** 0.81 * 1000
    else:
        depth = 0.36 * (Diam) ** 0.49 * 1000

    return depth


##############################################################################
##############################################################################
# Model data
#####################


###########################################################################
# Compute a theoric CSDFD
# Diam_bin in km
# depth_bin in m
# ob_array[0] is the age in Gy
# ob_array[1] is corresponding obliteration in m/Gy
# obl_dep give the dependancy of obliteration on depth and Diam
#####################
def build_density_mat(Diam_bin, depth_bin, obl_vec, obl_dep=[0, 0]):
    N_Diam = len(Diam_bin) - 1
    N_depth = len(depth_bin) - 1

    diam_dep = obl_dep[0]
    depth_dep = obl_dep[1]

    age = np.array(obl_vec[0])
    obliteration = np.array(obl_vec[1])

    # Chronology system
    prod_func = Ivanov
    # scaling law depth=f(diam) should give a depth in m
    depth_func = Garvin

    # The ob_array should start with the older ages
    if age[-1] > age[0]:
        age = age[::-1]
        obliteration = obliteration[::-1]

        # create more detailled age and obliteration vec with a small time step
    dt = 0.01
    Nb_step = int(age[0] / dt + 1)
    Real_tot_time = (Nb_step - 1) * dt
    obliteration_vec = np.zeros(Nb_step)
    Age_vec = np.zeros(Nb_step)
    for i_time in range(0, Nb_step):
        Real_time = i_time * dt
        Real_age = Real_tot_time - Real_time
        Age_vec[i_time] = Real_age
        i_age_modeled = np.max(np.where(age >= Real_age))
        obliteration_vec[i_time] = obliteration[i_age_modeled]

    age = Age_vec
    obliteration = obliteration_vec

    # Check if we need to add depth bins
    # In case expected max depth is higher than the depth bin limit
    nb_bins_to_add = 0
    for i_Diam in range(0, N_Diam):
        X_sup = Diam_bin[i_Diam + 1]
        delta_depth = (depth_bin[-1] - depth_bin[-2])
        if depth_func(X_sup) > depth_bin[-1]:
            nb_bins_to_add = max(nb_bins_to_add, int((depth_func(X_sup) - depth_bin[-1]) / delta_depth) + 1)
    # now add depth_bins
    if nb_bins_to_add != 0:
        depth_bin2 = np.zeros(N_depth + nb_bins_to_add)
        depth_bin2[:N_depth + 1] = depth_bin
        depth_bin2[N_depth + 1:] = np.arange(1, nb_bins_to_add) * delta_depth + depth_bin[N_depth]
        N_depth = N_depth + nb_bins_to_add
        depth_bin = depth_bin2

    # Build of a crat density matrix
    synthetic_dens = np.full((N_Diam, N_depth), 0.)
    # the computation is made independently on each diam bin
    for i_Diam in range(0, N_Diam):
        test_sum = 0.
        X_inf = Diam_bin[i_Diam]
        X_sup = Diam_bin[i_Diam + 1]
        mean_Diam = sqrt(X_inf * X_sup)

        # Get the depth of a fresh crater
        # i.e. the depth of the crater of age=0
        if depth_func(mean_Diam) > depth_bin[0] and depth_func(mean_Diam) < depth_bin[-1]:
            i_depth = np.max(np.where(depth_bin < depth_func(mean_Diam)))
            delta_depth = (depth_func(mean_Diam) - depth_bin[i_depth])
        else:
            i_depth = -1

        time = np.float64(0.)
        count_time = 0.
        i_age = len(age) - 1

        # then we go back in time increasing age
        # Until we reach the age set in the model
        # or depth reach 0 meaning all older crater are totaly erased
        while i_depth >= 0 and time < age[0]:
            time = np.float64(time)
            i_age = np.min(np.where(age <= time))

            depth_to_use = depth_bin[i_depth] + delta_depth

            obliteration_loc = obliteration[i_age] * (mean_Diam * 1000) ** diam_dep * (depth_to_use) ** depth_dep

            depth_inc = 0

            # If no obliteration, no need to change the depth bin
            # since crater of age and age+dt will have the same depth
            if obliteration_loc == 0.:
                if i_age == 0:
                    time_step = age[0] - time
                else:
                    time_step = age[i_age - 1] - time

            else:
                # The time needed to erode acrater to the next depth bin
                time_step_depth = delta_depth / obliteration_loc

                # If time_step_depth doesn't go further than the next time step
                # Then we just change depth bin and store the age we reached
                if time + time_step_depth <= age[i_age - 1]:
                    time_step = time_step_depth
                    depth_inc = 1

                # If time_step_depth bring us latter than the next time step
                # we may have missed a change in obliteration
                # so we shorten the time step and store the difference between curent depth and the next bin
                else:
                    time_step = age[i_age - 1] - time
                    delta_depth = delta_depth - obliteration_loc * time_step

            # The number of crater is determined from production function
            rate = (((prod_func(X_inf, time + time_step) - prod_func(X_sup, time + time_step)) -
                     (prod_func(X_inf, time) - prod_func(X_sup, time)))
                    / (X_sup - X_inf))

            time = time + time_step

            # We add this number of crater to the density grid
            synthetic_dens[i_Diam, i_depth] = synthetic_dens[i_Diam, i_depth] + rate / (
                        (depth_bin[i_depth + 1] - depth_bin[i_depth]) / 1000)

            # if we reached a new depth bin
            # we set the difference between curent depth and the next bin
            # as the bin width
            if depth_inc == 1:
                i_depth = i_depth - 1
                delta_depth = (depth_bin[i_depth + 1] - depth_bin[i_depth])

    return synthetic_dens, depth_bin


###########################################################################
# Compute a theoric CSDFD
# Diam_min and max in km
# ob_array[0] is the age in Gy
# ob_array[1] is corresponding obliteration in m/Gy
# obl_dep give the dependancy of obliteration on depth and Diam
# nb_crat_opti optimize the algorithm
#       using an increasing area for incresing diameter
#       such as the number of crater in a bin is constant
#       set to 0 to model all the population
# discr determine how the craters are discretized 'poisson' or 'not_poisson'
#####################
def model_crat(area_tot, Diam_min, Diam_max, obl_vec, nb_crat_opti=5, obl_dep=[0., 0.], discr='poisson',
               backwasting=False):
    diam_dep = obl_dep[0]
    depth_dep = obl_dep[1]

    age = obl_vec[0]
    obliteration = obl_vec[1]

    # Production function
    prod_func = Ivanov
    prod_der_func = der_Ivanov
    # scaling law depth=f(diam)
    depth_func = Garvin

    ##The ob_array should start with the older ages
    if age[-1] > age[0]:
        age = age[::-1]
        obliteration = obliteration[::-1]

    age = np.array(age)
    obliteration = np.array(obliteration)

    # create more detailled age and obliteration vec with a small time step
    dt = 0.01
    Nb_step = int(age[0] / dt + 1)
    Real_tot_time = (Nb_step - 1) * dt
    Age_vec = np.zeros(Nb_step)
    obliteration_vec = np.zeros(Nb_step)
    for i_time in range(0, Nb_step):
        Real_time = i_time * dt
        Real_age = Real_tot_time - Real_time
        Age_vec[i_time] = Real_age
        i_age_modeled = np.max(np.where(age <= Real_age))
        obliteration_vec[i_time] = obliteration[i_age_modeled] * dt

    # Diameter discretization
    nb_bin = 500
    diameterBin = np.zeros(nb_bin + 1)
    Area = np.full(nb_bin, area_tot)

    Diam_vec = np.zeros(nb_bin)
    for i_bin in range(0, nb_bin + 1):
        diameterBin[i_bin] = Diam_min * (Diam_max / Diam_min) ** (i_bin / nb_bin)
        if i_bin != nb_bin:
            Diam_vec[i_bin] = Diam_min * (Diam_max / Diam_min) ** ((i_bin + 0.5) / nb_bin)

    # each diameter step get its own area such as the number of crater in this bin
    # dont exceed nb_crat_opti
    if nb_crat_opti != 0:
        for i_bin in range(0, nb_bin):
            if area_tot == -1:
                ob_bin = np.cumsum(obliteration_vec) * Diam_vec[i_bin] ** diam_dep
                depth_max = depth_func(Diam_vec[i_bin])
                if ob_bin[-1] > depth_max:
                    time_to_obliterate = Age_vec[np.min(np.where(ob_bin > depth_max))]
                else:
                    time_to_obliterate = Real_tot_time
                Area[i_bin] = nb_crat_opti / (
                            prod_func(diameterBin[i_bin], Real_tot_time) - prod_func(diameterBin[i_bin + 1], Real_tot_time))


    # crat is composed of diam_crat that contains the diameter
    # depth_crat that contain the depth of the craters
    # and area_crat that contains the area associated with the crater
    diam_crat = np.zeros(0)
    diam0_crat = np.zeros(0)
    depth_crat = np.zeros(0)
    area_crat = np.zeros(0)

    # time loop
    for i_time in range(0, Nb_step):
        Real_time = i_time * dt
        Real_age = Real_tot_time - Real_time


        ######################## Poisson law version ######################
        if discr == 'poisson':
            new_crats = get_craters_poisson(diameterBin, Real_age, dt, Area, prod_func, depth_func)

        ###################### Non Poisson law version #####################
        if discr == 'nonpoisson':
            new_crats = get_craters_from_distribution(diameterBin, Diam_min, Real_age, dt, Area, prod_func, depth_func)

        diam_crat = np.concatenate((diam_crat, new_crats[0]))
        diam0_crat = np.concatenate((diam0_crat, new_crats[0]))
        depth_crat = np.concatenate((depth_crat, new_crats[1]))
        area_crat = np.concatenate((area_crat, new_crats[2]))

        # Now we erode craters
        crat_to_remove = np.full(0, 0)

        for i_crat in range(len(depth_crat)):
            # each crater is eroded
            obliteration = (obliteration_vec[i_time] *
                            (depth_crat[i_crat]) ** depth_dep *
                            (diam_crat[i_crat] * 1000) ** diam_dep)
            depth_crat[i_crat] = depth_crat[i_crat] - obliteration


            # craters with a depth lower than 0 are suppressed
            depth_dec_lim = 10 * diam0_crat[i_crat]
            if depth_crat[i_crat] < depth_dec_lim:
                crat_to_remove = np.append(crat_to_remove, i_crat)


            if backwasting:
                # Backwasting
                # t_scaled = age_crat[i_crat] * kappa/kappa0 * (D_0/diam0_crat[i_crat])**2 
                # diam_crat[i_crat] = diam0_crat[i_crat] + coefBackWash1 * t_scaled**2 + coefBackWash2 * t_scaled

                # Melosh (1989) does not work
                # depth0 = depth_func(diam0_crat[i_crat])
                # diam_crat[i_crat] = diam0_crat[i_crat] * (1 / (1 - 5 / 4 * (1 - depth_crat[i_crat] / depth0))**(1/3))

                # Basilevsky (2015)
                depth0 = depth_func(diam0_crat[i_crat])

                test = diam0_crat[i_crat] * (
                    (depth0 / 1000 / diam0_crat[i_crat] - 4/3 * (depth0 / 1000 / diam0_crat[i_crat]) ** 3) /
                    (depth_crat[i_crat] / 1000 / diam_crat[i_crat] - 4/3 * (depth_crat[i_crat] / 1000/diam_crat[i_crat]) ** 3)
                ) ** (1/3)

                if (diam_crat[i_crat] == diam_crat[i_crat] and test != test):
                    print('********')
                    print(diam0_crat[i_crat])
                    print(diam_crat[i_crat])
                    print(depth0)
                    print(depth_crat[i_crat])


                diam_crat[i_crat] = diam0_crat[i_crat] * (
                    (depth0 / 1000 / diam0_crat[i_crat] - 4/3 * (depth0 / 1000 / diam0_crat[i_crat]) ** 3) /
                    (depth_crat[i_crat] / 1000 / diam_crat[i_crat] - 4/3 * (depth_crat[i_crat] / 1000/diam_crat[i_crat]) ** 3)
                ) ** (1/3)






        if np.size(crat_to_remove) > 0:
            depth_crat = np.delete(depth_crat, crat_to_remove)
            diam_crat = np.delete(diam_crat, crat_to_remove)
            diam0_crat = np.delete(diam0_crat, crat_to_remove)
            area_crat = np.delete(area_crat, crat_to_remove)

    return diam_crat, depth_crat, area_crat


def get_craters_poisson(diameter_bin, Real_age, dt, Area, prod_func, depth_func):
    new_diams = np.array([])
    new_depths = np.array([])
    new_areas = np.array([])

    nb_bin = np.size(diameter_bin) - 1

    # at each time step craters are added in each diameter step
    for i_bin in range(0, nb_bin):
        i_Diam_min = diameter_bin[i_bin]
        i_Diam_max = diameter_bin[i_bin + 1]

        rate = Area[i_bin] * ((prod_func(i_Diam_min, Real_age + dt) - prod_func(i_Diam_max, Real_age + dt)) -
                              (prod_func(i_Diam_min, Real_age) - prod_func(i_Diam_max, Real_age)))

        nb_crat_to_add = np.random.poisson(rate, 1)[0]


        for i_add in range(0, nb_crat_to_add):
            # The diam of the crater is chosen randomly into the bin it has been added
            new_diam = 10 ** np.random.uniform(log(i_Diam_min, 10), log(i_Diam_max, 10))

            new_diams = np.append(new_diams, new_diam)
            new_areas = np.append(new_areas, Area[i_bin])

            # We also associate a depth with this crater
            # We use one of the scaling law of the literature
            new_depth = depth_func(new_diam)
            new_depths = np.append(new_depths, new_depth)

    return [new_diams, new_depths, new_areas]


def get_craters_from_distribution(diameter_bin, diameter_min, Real_age, dt, Area, prod_func, depth_func):
    new_diams = np.array([])
    new_depths = np.array([])
    new_areas = np.array([])

    nb_bin = np.size(diameter_bin) - 1

    # built the SFD of craters occuring during this time
    PF = np.zeros(nb_bin)
    for i_bin in range(0, nb_bin):
        i_Diam_min = diameter_bin[i_bin]
        i_Diam_max = diameter_bin[i_bin + 1]

        PF[i_bin] = prod_func(i_Diam_min, Real_age) - prod_func(i_Diam_max, Real_age)

    nb_tot_crat_to_add = int((prod_func(diameter_min, Real_age + dt) - prod_func(diameter_min, Real_age)) * Area[0])


    # randomly pick nb_tot_crat_to_add craters using the PF as a probability density
    linear_i_sample = np.random.choice(np.arange(0, nb_bin), nb_tot_crat_to_add, p=PF / np.sum(PF))
    for i_bin in linear_i_sample:
        i_Diam_min = diameter_bin[i_bin]
        i_Diam_max = diameter_bin[i_bin + 1]

        # The diam of the crater is chosen randomly into the bin it has been added
        new_diam = 10 ** np.random.uniform(log(i_Diam_min, 10), log(i_Diam_max, 10))
        # new_diam=uniform(i_Diam_min,i_Diam_max)
        new_diams = np.append(new_diams, new_diam)
        new_areas = np.append(new_areas, Area[0])

        # We also associate a depth with this crater
        # We use one of the scaling law of the litterature
        new_depth = depth_func(new_diam)
        new_depths = np.append(new_depths, new_depth)

    return [new_diams, new_depths, new_areas]


###########################################################################
# Compute a theoric CSDFD
# crater are only degradated by diffusion
# apply backwasting of craters
#####################
def model_crat_diffusion(area_tot, Diam_min, Diam_max, age, discr='poisson', backwasting=True):
    ##### Params
    kappa = 7
    kappa0 = 0.7
    D_0 = 0.3
    coefBackWash1 = 0.499
    coefBackWash2 = 0.555
    rho = 1

    # Production function
    prod_func = Ivanov
    prod_der_func = der_Ivanov
    # scaling law depth=f(diam)
    depth_func = Garvin

    ##The ob_array should start with the older ages
    if age[-1] > age[0]:
        age = age[::-1]

    age = np.array(age)

    # create more detailled age and obliteration vec with a small time step
    dt = 0.01
    Nb_step = int(age[0] / dt + 1)
    Real_tot_time = (Nb_step - 1) * dt
    Age_vec = np.zeros(Nb_step)
    for i_time in range(0, Nb_step):
        Real_time = i_time * dt
        Real_age = Real_tot_time - Real_time
        Age_vec[i_time] = Real_age
        i_age_modeled = np.max(np.where(age <= Real_age))

    # Diameter discretization
    nb_bin = 500
    Diam = np.zeros(nb_bin + 1)
    Area = np.zeros(nb_bin)

    Diam_vec = np.zeros(nb_bin)
    for i_bin in range(0, nb_bin + 1):
        Diam[i_bin] = Diam_min * (Diam_max / Diam_min) ** (i_bin / nb_bin)
        if i_bin != nb_bin:
            Diam_vec[i_bin] = Diam_min * (Diam_max / Diam_min) ** ((i_bin + 0.5) / nb_bin)

    # crat is composed of diam_crat that contains the diameter
    # depth_crat that contain the depth of the craters
    # and area_crat that contains the area associated with the crater
    diam_crat = np.zeros(0)
    diam0_crat = np.zeros(0)
    age_crat = np.zeros(0)
    depth_crat = np.zeros(0)
    area_crat = np.zeros(0)

    # time loop
    for i_time in range(0, Nb_step):
        Real_time = i_time * dt
        Real_age = Real_tot_time - Real_time

        ######################## Poisson law version ######################
        if discr == 'poisson':
            # the number of crater added follow a poisson law with an average
            # number of event "rate"
            # at each time step craters are added in each diameter step
            for i_bin in range(0, nb_bin):
                i_Diam_min = Diam[i_bin]
                i_Diam_max = Diam[i_bin + 1]
                i_Diam = Diam_vec[i_bin]

                rate = area_tot * ((prod_func(i_Diam_min, Real_age + dt) - prod_func(i_Diam_max, Real_age + dt)) -
                                   (prod_func(i_Diam_min, Real_age) - prod_func(i_Diam_max, Real_age)))

                nb_crat_to_add = np.random.poisson(rate, 1)[0]

                for i_add in range(0, nb_crat_to_add):
                    # The diam of the crater is chosen randomly into the bin it has been added
                    new_diam = 10 ** np.random.uniform(log(i_Diam_min, 10), log(i_Diam_max, 10))
                    # new_diam=uniform(i_Diam_min,i_Diam_max)        #that does not change much

                    diam_crat = np.append(diam_crat, new_diam)
                    area_crat = np.append(area_crat, Area[i_bin])

                    # We also associate a depth with this crater
                    # We use one of the scaling law of the litterature
                    new_depth = depth_func(new_diam)
                    depth_crat = np.append(depth_crat, new_depth)

                    diam0_crat = np.append(diam_crat, new_diam)
                    age_crat = np.append(age_crat, 0.)

        ###################### Non Poisson law version #####################
        if discr == 'nonpoisson':
            # built the SFD of craters occuring during this time
            PF = np.zeros(nb_bin)
            for i_bin in range(0, nb_bin):
                i_Diam_min = Diam[i_bin]
                i_Diam_max = Diam[i_bin + 1]

                PF[i_bin] = prod_func(i_Diam_min, Real_age) - prod_func(i_Diam_max, Real_age)

            nb_tot_crat_to_add = int((prod_func(Diam_min, Real_age + dt) - prod_func(Diam_min, Real_age)) * Area[0])
            # randomly pick nb_tot_crat_to_add craters using the PF as a probability density
            linear_i_sample = np.random.choice(np.arange(0, nb_bin), nb_tot_crat_to_add, p=PF / np.sum(PF))
            for i_bin in linear_i_sample:
                i_Diam_min = Diam[i_bin]
                i_Diam_max = Diam[i_bin + 1]

                # The diam of the crater is chosen randomly into the bin it has been added
                new_diam = 10 ** np.random.uniform(log(i_Diam_min, 10), log(i_Diam_max, 10))
                # new_diam=uniform(i_Diam_min,i_Diam_max)
                diam_crat = np.append(diam_crat, new_diam)
                area_crat = np.append(area_crat, Area[0])

                # We also associate a depth with this crater
                # We use one of the scaling law of the litterature
                new_depth = depth_func(new_diam)
                depth_crat = np.append(depth_crat, new_depth)

                diam0_crat = np.append(diam_crat, new_diam)
                age_crat = np.append(0)

        # Now we erode craters
        crat_to_remove = np.zeros(0)
        for i_crat in range(len(depth_crat)):

            # Buggiolachi 2020
            # depth depletion
            obliteration = - kappa / rho * 16 * depth_crat[i_crat] / diam_crat[i_crat]

            depth_crat[i_crat] = depth_crat[i_crat] - obliteration

            if backwasting:
                # Backwasting
                # t_scaled = age_crat[i_crat] * kappa/kappa0 * (D_0/diam0_crat[i_crat])**2 
                # diam_crat[i_crat] = diam0_crat[i_crat] + coefBackWash1 * t_scaled**2 + coefBackWash2 * t_scaled

                # Ivanov (2018)
                depth0 = depth_func(diam0_crat[i_crat])
                diam_crat[i_crat] = 1 / (1 - 5 / 4 * (1 - depth_crat[i_crat] / depth0))

            # craters with a depth/Diam lower than 1% are supressed
            depthDetectionLimit = 10 * diam0_crat[i_crat]

            if depth_crat[i_crat] < depthDetectionLimit:
                crat_to_remove = np.append(crat_to_remove, i_crat)

        if np.size(crat_to_remove) > 0:
            depth_crat = np.delete(depth_crat, crat_to_remove)
            diam_crat = np.delete(diam_crat, crat_to_remove)
            area_crat = np.delete(area_crat, crat_to_remove)
            age_crat = np.delete(age_crat, crat_to_remove)
            diam0_crat = np.delete(diam0_crat, crat_to_remove)

        age_crat = age_crat + dt

    return diam_crat, depth_crat, area_crat


def applyDiffusion(Diam, depth, kappa, dt):
    # assuming a flux following a difusion law with coeficient kappa
    # and a triangular shaped crater
    # and conservation of mass

    # Diam is D
    # Radius is R = D/2
    # depth is d
    # slope is S = d/R

    # dd/dt = 4/9 * kappa * S/(R**3 - R**2)
    # dS/dt = 3/2 dd/dt
    # D(t+1) = (d + dd/dt)/(S + 2/3 * dd/dt)

    Slope = 2 * depth / Diam

    deltaDepth = 4 / 9 * kappa * Slope / ((Diam / 2) ** 3 - (Diam / 2) ** 2)

    newDepth = depth + deltaDepth * dt

    newDiam = (depth + deltaDepth) / (Slope + 2 / 3 * deltaDepth)

    return newDiam, newDepth
