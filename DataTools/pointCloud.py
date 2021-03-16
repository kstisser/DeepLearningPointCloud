import sys
sys.path.append("....")
import defs
import open3d as o3d
import numpy as np
import os
from enum import Enum
from Visualization import visualizer as vizu

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

        self.samplingMethod = samplingMethod
        if(samplingMethod == DownsampleType.RANDOM):
            self.downsampleRandomly(1000)
            self.viz.visualizeFaceAndAllPlot(self.downsampledFace, self.downsampledAll)

    def downsampleRandomly(self, downsampleNumber):
        if self.samplingMethod != DownsampleType.RANDOM:
            print("Error! Wrong downsampling method, can't visualize!")
            return
        face_rows = self.faceseg.shape[0]
        randomIndices = (np.random.choice(face_rows, size=downsampleNumber, replace=False))
        self.downsampledFace = self.faceseg[randomIndices,:]

        all_rows = self.allseg.shape[0]
        randomIndices = (np.random.choice(all_rows, size=downsampleNumber, replace=False))
        self.downsampledAll = self.allseg[randomIndices,:]   

        print("Downsampled Face shape: ", self.downsampledFace.shape)
        print("Downsampled All shape: ", self.downsampledAll.shape)     

    def getDataFrame(self):
        print("Getting data frame")