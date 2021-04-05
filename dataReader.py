import numpy as np
import os
import sys
import math
import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt 
import DataTools.pointCloud as pc
from DataTools import defs
from DataAugmentation import dataAugmentation as da

class DataReader:
    def __init__(self, pcType = defs.StructureType.VOXEL, dataFolder="default", samplingMethod = defs.DownsampleType.RANDOM):
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
        self.minPoints = 1000000000
        self.maxPoints = 0
        for filename in os.listdir(self.dataFolder):
            folder = os.path.join(self.dataFolder, filename)
            self.pointClouds.append(pc.PointCloud(folder, pcType, samplingMethod, (8,8,12)))
            numPoints = len(self.pointClouds[-1].faceseg)
            if(numPoints < self.minPoints):
                self.minPoints = numPoints
            if(numPoints > self.maxPoints):
                self.maxPoints = numPoints

        #Use data augmentation to increase data variation
        #self.dataAugmentation = da.DataAugmentation(self.pointClouds)
        #self.pointClouds = self.dataAugmentation.augmentAllData()

        print("Read in all point clouds, and have Max points: ", self.maxPoints, " and Min Points: ", self.minPoints) 

    def findCloudFolder(self, rootDir):
            #if left as default, read in and fix spacing
            for folder in os.listdir(rootDir):
                if "Cloud" in folder:
                    return folder

    def getTrainTestSplit(self):
        trainRatio = 0.8
        numTrain = math.floor(len(self.pointClouds) * trainRatio)
        trainingData = self.pointClouds[:numTrain]
        testData = self.pointClouds[numTrain:]
        if len(testData) < 1:
            print("Error! Data set too small to make any test data!")
            sys.exit()
        print("Splitting into training: ", len(trainingData), ", test: ", len(testData))
        return [trainingData, testData]