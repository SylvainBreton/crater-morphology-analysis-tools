#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 11:37:17 2019
"""


import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math



def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points))

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
                # already added
                return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
        
    coords = np.array([point.coords[0] for point in points])
    
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points





#def find_crater_cloud(Diam_bins, depth_bins, crat_array):
#    nb_bin_Diam =len(Diam_bins)-1
#    nb_bin_depth=len(depth_bins)-1
#    
#    vert_low=[[],[]]
#    vert_high=[[],[]]
#    
#    Diam_crat =crat_array[0]
#    depth_crat=crat_array[1]
#    
#    min_Diam=np.min(Diam_crat)
#    
#    i_min_crat=np.where(Diam_crat==min_Diam)[0]
#    
#    
#    
#    if len(i_min_crat)>1:
#        prev_h_d=np.min(depth_crat[i_min_crat])
#        prev_l_d=np.max(depth_crat[i_min_crat])
#        
#        vert_low[0].append(min_Diam)
#        vert_low[1].append(prev_l_d)
#        
#        vert_high[0].append(min_Diam)
#        vert_high[1].append(prev_h_d)
#        
#    else:
#        prev_h_d=depth_crat[i_min_crat]
#        prev_l_d=depth_crat[i_min_crat]
#        
#        vert_low[0].append(min_Diam)
#        vert_low[1].append(prev_l_d)
#    
#    
#    
#    
#    is_in_cloud=np.full((nb_bin_Diam, nb_bin_depth),0)
#    
#    i_min=np.min(np.where(Diam_bins>min_Diam))
#    i_max=np.min(np.where(Diam_bins<np.max(Diam_crat)))
#    
#    for i_Diam in range(i_min,i_max):
#        #is there craters in this bin ?
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





