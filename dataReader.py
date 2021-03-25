import numpy as np
import os
import sys
import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt 
import DataTools.pointCloud as pc
import defs


class DataReader:
    def __init__(self, pcType = defs.StructureType.VOXEL, dataFolder="default", samplingMethod = pc.DownsampleType.RANDOM):
        self.pcType = pcType

        if dataFolder == "default":
            dataFolder = self.findCloudFolder(".")

        #check folder exists
        if not os.path.isdir(dataFolder):
            print("Error! Can't find folder with data in it: " + dataFolder)
            sys.exit()
        self.dataFolder = os.path.join(dataFolder,self.findCloudFolder(dataFolder))
        #sanity check new folder exists
        if not os.path.isdir(dataFolder):
            print("Error! Can't find folder with data in it: " + self.dataFolder)
            sys.exit()

        self.pointClouds = []
        for filename in os.listdir(self.dataFolder):
            folder = os.path.join(self.dataFolder, filename)
            self.pointClouds.append(pc.PointCloud(folder, pcType, samplingMethod, (8,8,12)))

    def findCloudFolder(self, rootDir):
            #if left as default, read in and fix spacing
            for folder in os.listdir(rootDir):
                if "Cloud" in folder:
                    return folder