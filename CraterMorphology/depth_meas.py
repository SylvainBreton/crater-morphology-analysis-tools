#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:33:14 2018

@author: Sylvain Breton
"""


import csv
from math import *
import numpy as np

from osgeo import gdal
from osgeo import ogr
import osgeo.osr as osr
import os

import sys


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtCore as core


class depth_measurement_GUI(QDialog): #
    """ Création d'une classe pour l'interface graphique"""

    def __init__(self):

        super().__init__()

        self.title_ = "Crater depth measurement"
        self.left_ = 600
        self.top_ = 300
        self.width_ = 500
        self.height_ = 300

        self.textboxLoadDEM_ = None
        self.textboxLoadshp_ = None

        self.DEM_file = None
        self.Crater_shp_file = None
        

        self.initGUI()

    def initGUI(self):

        self.setWindowTitle(self.title_)
        self.setGeometry(self.left_, self.top_, self.width_, self.height_)
        

        self.createGridLayoutLoadData()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.InstructionsBox)
        windowLayout.addWidget(self.horizontalGroupBoxLoadData)

        self.setLayout(windowLayout)

        self.show()

    def createGridLayoutLoadData(self):
        self.InstructionsBox = QGroupBox("This software compute crater depth " + 
                                             "from a shapefile map and an alligned DEM "+
                                             "both file must be in the same projection system")
        layout1 = QGridLayout()
        layout1.setColumnStretch(1, 10)
        self.InstructionsBox.setLayout(layout1)
        
        
        self.horizontalGroupBoxLoadData = QGroupBox("Charger les données")
        layout = QGridLayout()
        layout.setColumnStretch(0, 10)
        layout.setColumnStretch(1, 3)

        self.textboxLoadDEM_ = QLineEdit(self)
        layout.addWidget(self.textboxLoadDEM_, 0, 0)
        layout.addWidget(self.create_button('DEM', self.accesBrowser4DEM), 0, 1)
        
        self.textboxLoadshp_ = QLineEdit(self)
        layout.addWidget(self.textboxLoadshp_, 1, 0)
        layout.addWidget(self.create_button('Crater shapefile', self.accesBrowser4cratershp), 1, 1)
        
        
        layout.addWidget(self.create_button('Compute', self.measure_depth), 2, 0)

        self.horizontalGroupBoxLoadData.setLayout(layout)
        
        
        
    def create_button(self, libelle, method):
        """Creation d'un bouton et association avec l'action à lancer au clic"""
        button = QPushButton(libelle,self)
        # core.QObject.connect(button, core.SIGNAL('clicked()'))
        button.clicked.connect(method)
        return button

    
    
    def accesBrowser4DEM(self):
        self.DEM_file= QFileDialog.getOpenFileName(self,
                     "Ouvrir un fichier",
                     "C:\\",
                     "tif (*.tif);;img (*.img);;jp2 (*.jp2)")[0]

        self.textboxLoadDEM_.setText(self.DEM_file)
        
    def accesBrowser4cratershp(self):
        self.Crater_shp_file = QFileDialog.getOpenFileName(self,
                     "Ouvrir un fichier",
                     "C:\\",
                     "shp (*.shp)")[0]

        self.textboxLoadshp_.setText(self.Crater_shp_file)

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #Compute percentiles from a sorted array of a values and an associated array of cummulative frequency
    #input = ( sorted elevations, areas_covered, perc_to_compute)
    def percentile_from_freq(self, within_crat,freq_cum,perc):
        percentile_freq=freq_cum[-1]*perc/100
        if len(within_crat)>1:
            if perc==100:
                comp_perc=(within_crat[-1])
            elif perc==0 or percentile_freq<freq_cum[0]:
                comp_perc=(within_crat[0])
            else:
                #start in the perc_i index of the array to optimize computation time
                start_i=int(perc/100*len(freq_cum))
                #iteration direction (-1 or 1)
                iter_step=int((percentile_freq-freq_cum[start_i])/abs(percentile_freq-freq_cum[start_i]))
                
                index_while=start_i
                while freq_cum[index_while]*iter_step<percentile_freq*iter_step:
                    index_while=index_while+iter_step
                
                #linear interpolation from elevations value and cum freq array
                comp_perc=(within_crat[index_while]-
                                  ((within_crat[index_while]-within_crat[index_while-iter_step])/
                                  (freq_cum[index_while]-freq_cum[index_while-iter_step])*
                                  (freq_cum[index_while]-percentile_freq)))
        elif len(within_crat)==1:
            comp_perc=within_crat[0]
        else:
            comp_perc=np.nan
    
    
        return comp_perc
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
               
    #################################################################
    ########################## GUI ##################################  
    #################################################################
    #################################################################
    
    def measure_depth(self):
        ######################################
        #DEM
        # Open dataset (image)
        DEM = gdal.Open(self.DEM_file, gdal.GA_ReadOnly)
        
        geotransform = DEM.GetGeoTransform()
        proj = DEM.GetProjection()
        
        DEM_band=DEM.GetRasterBand(1)
        
        X_size=DEM.RasterXSize
        Y_size=DEM.RasterYSize
        
        ######################################
        #Open shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        crater_shp = driver.Open(self.Crater_shp_file,1)
        
        crat_layer = crater_shp.GetLayer()
        featureCount = crat_layer.GetFeatureCount()
        
        
        
        #Add a column that store depth in m
        test_field=0
        fields = []
        for i in range(1, crat_layer.GetFeature(0).GetFieldCount()):
            field = crat_layer.GetFeature(0).GetDefnRef().GetFieldDefn(i).GetName()
            fields.append(field)
            if field=="Depth_m":
                test_field=1
        
        if test_field==0:
            Depth_field = ogr.FieldDefn("Depth_m", ogr.OFTReal)
            Depth_field.SetWidth(15)
            Depth_field.SetPrecision(15)
            crat_layer.CreateField(Depth_field)
        
        
        
        
        for index in range(0,featureCount):         
            crater = crat_layer.GetFeature(index)
            
            crat_geom = crater.GetGeometryRef()
            env=crat_geom.GetEnvelope()
            
            
            R_crat=sqrt(crat_geom.GetArea()/pi)
            long_center=(env[1]+env[0])/2
            lat_center=(env[3]+env[2])/2
            
            
            
            #convert geo coordinates into matrix coordinates
            index_long_low = int((env[0]-geotransform[0])/geotransform[1])-1
            index_lat_low = int((env[3]-geotransform[3] )/ geotransform[5])-1
               
            nb_px_long_init = int((env[1]-env[0]) / geotransform[1]) +3
            nb_px_lat_init =  int((env[3]-env[2]) / geotransform[5]) -3
            
            #you can choose to apply resampling to your DEM 
            #(reduce computation time but increase error)
            resampling= 0
            if resampling!=0:
                nb_px_long = max(int(nb_px_long_init * geotransform[1] / resampling),1)
                nb_px_lat  = min(-int(nb_px_lat_init  * geotransform[5] / resampling),-1)
            else:
                nb_px_long = nb_px_long_init
                nb_px_lat  = nb_px_lat_init
            new_geot_long = nb_px_long_init * geotransform[1]/ nb_px_long 
            new_geot_lat  = nb_px_lat_init  * geotransform[5]/ nb_px_lat
            
            
            NaN_value=DEM_band.GetNoDataValue()
            #Open the raster on a square  around the crater
            #Read the raster around this crater
            data = DEM_band.ReadAsArray(index_long_low, index_lat_low,
                                        nb_px_long_init,-nb_px_lat_init,
                                        nb_px_long,-nb_px_lat)
            
            
            
            
            
            within_crat=[]
            within_buff=[]
            freq=[]
            #Iterate through the pixels opened to check if they are actually in the crater
            for it_lat in range(0,-nb_px_lat):
                for it_long in range(0,nb_px_long):
                    long_px = geotransform[0]+ (it_long+1/2)*new_geot_long + index_long_low*geotransform[1]
                    lat_px  = geotransform[3]+ (it_lat+1/2) *new_geot_lat  + index_lat_low *geotransform[5]
        
                    
                    elevation_val=-1
                    freq_val=-1
                    #if the pixel is totaly comprised in the circle
                    if (((R_crat-sqrt(new_geot_lat**2+new_geot_long**2)/4)**2
                                     >(long_px-long_center)**2+(lat_px-lat_center)**2) 
                                    and data[it_lat,it_long]!=NaN_value):
                
                        elevation_val=data[it_lat,it_long]
                        freq_val=-(new_geot_long*new_geot_lat)
                                        
                    #if the pixel is on the edge of the circle
                    elif (((R_crat+sqrt(new_geot_lat**2+new_geot_long**2))**2
                                         >(long_px-long_center)**2+(lat_px-lat_center)**2)
                                     and data[it_lat,it_long]!=NaN_value):
                
                        elevation_val=data[it_lat,it_long]
                        #Whithin buf is used to compute rim elevation
                        within_buff.append(elevation_val)
                        
                        #wich is the crater area covered by the pixel
                        px_border = ogr.Geometry(ogr.wkbLinearRing)
                        px_border.AddPoint(long_px-1/2*new_geot_long,lat_px+1/2*new_geot_lat)
                        px_border.AddPoint(long_px-1/2*new_geot_long,lat_px-1/2*new_geot_lat)
                        px_border.AddPoint(long_px+1/2*new_geot_long,lat_px-1/2*new_geot_lat)
                        px_border.AddPoint(long_px+1/2*new_geot_long,lat_px+1/2*new_geot_lat)
                        px_border.AddPoint(long_px-1/2*new_geot_long,lat_px+1/2*new_geot_lat)
                        poly = ogr.Geometry(ogr.wkbPolygon)
                        poly.AddGeometry(px_border)
                        
                        Area_intersect=poly.Intersection(crat_geom)
                        freq_val=Area_intersect.GetArea()
                        
                    #add the elevation value in the vector used to compute percentiles
                    if elevation_val!=-1:
                        within_crat=np.append(within_crat,elevation_val)
                        freq=np.append(freq,freq_val)
                        
                    
            if len(within_crat)>0:     
                #Compute percentiles (crater floor elevation)
                freq = [x for _,x in sorted(zip(within_crat,freq))]                   
                within_crat=np.sort(within_crat)         
                comp_depth=[]
                freq_cum=np.cumsum(freq)
        
                floor_depth=self.percentile_from_freq(within_crat,freq_cum,3)
                    
                    
                #Computes rim elevation
                rim_depth=np.percentile(within_buff,60)
            
            
                comp_depth=rim_depth-floor_depth
        
            else:
                comp_depth=0
                
            
            
            
            
            
            #add depth value in the shapefile
            crater.SetField("Depth_m",comp_depth)
            crat_layer.SetFeature(crater)
        
        crat_layer.ResetReading()
##################################################################


# creation d'un application
app = QApplication(sys.argv)  # on peut aussi passer en paramètre les arguments du programme sys.argv

# creation de l'interface
gui = depth_measurement_GUI()

sys.exit(app.exec_())









