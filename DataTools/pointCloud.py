import sys
sys.path.append("....")
import defs
import open3d as o3d
import numpy as np
import os
from enum import Enum
from Visualization import visualizer as vizu
from Preprocessing import randomSampler

class DownsampleType(Enum):
    NODOWNSAMPLE = 1
    RANDOM = 2

class PointCloud:
    def __init__(self, rootpath, pcType, samplingMethod):
        self.viz = vizu.Visualizer()
        self.pcType = pcType
        self.rootpath = rootpath
        self.facepath = os.path.join(rootpath, "face_segment.pcd")
        self.allpath = os.path.join(rootpath, "PointCloudCapture.pcd")
        self.face3d = o3d.io.read_point_cloud(self.facepath)
        self.all3d = o3d.io.read_point_cloud(self.allpath)
        self.faceseg = np.asarray(self.face3d.points)
        self.allseg = np.asarray(self.all3d.points)
        print("Read in Point Cloud object " + rootpath)
        print("Face shape: ", self.faceseg.shape)
        print("All shape: ", self.allseg.shape)

        self.randomSampler = randomSampler.RandomSampler()
        self.samplingMethod = samplingMethod
        self.sampleNumber = 1000
        self.downsample()

    def downsample(self):
        if(self.samplingMethod == DownsampleType.RANDOM):
            self.downsampledFace = self.randomSampler.randomlySamplePoints(self.faceseg, self.sampleNumber)
            self.downsampledAll = self.randomSampler.randomlySamplePoints(self.allseg, self.sampleNumber) 
            self.viz.visualizeFaceAndAllPlot(self.downsampledFace, self.downsampledAll)
            
    def getDataFrame(self):
        print("Getting data frame")