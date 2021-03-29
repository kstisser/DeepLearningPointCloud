import sys
sys.path.append("....")
import defs
import open3d as o3d
import numpy as np
import os
from enum import Enum
from Visualization import visualizer as vizu
from Preprocessing import randomSampler
from Embedding import voxels
from Embedding import pointpillars
import time

class DownsampleType(Enum):
    NODOWNSAMPLE = 1
    RANDOM = 2

class EmbeddingType(Enum):
    VOXELS = 1
    POINTPIXELS = 2
    PARABOLAS = 3

class PointCloud:
    def __init__(self, rootpath, pcType, samplingMethod, embeddingSize, embeddingType=EmbeddingType.POINTPIXELS):
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

        print("Face seg shape: ", self.faceseg.shape)
        print("All seg shape: ", self.allseg.shape)        

        #Sampling
        self.randomSampler = randomSampler.RandomSampler()
        self.samplingMethod = samplingMethod
        self.sampleNumber = 1000
        self.downsample()     

        #Embedding
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize
        self.embedPointCloud()

    def downsample(self):
        if(self.samplingMethod == DownsampleType.RANDOM):
            self.downsampledFace = self.randomSampler.randomlySamplePoints(self.faceseg, self.sampleNumber)
            self.downsampledAll = self.randomSampler.randomlySamplePoints(self.allseg, self.sampleNumber) 
            self.viz.visualizeFaceAndAllPlot(self.downsampledFace, self.downsampledAll)
            print("Downsampled Face shape: ", self.downsampledFace.shape)
            print("Downsampled All shape: ", self.downsampledAll.shape)
            
    def embedPointCloud(self):
        if self.embeddingType == EmbeddingType.VOXELS:
            vox = voxels.Voxels(self.embeddingSize)
            self.embedding = vox.voxelate(self.all3d)
        elif self.embeddingType == EmbeddingType.POINTPIXELS:
            startTime = time.time()
            self.pointPillars = pointpillars.PointPillars(self.downsampledAll)
            self.pointPillars.buildPillars()
            endTime = time.time()
            print("Point Pillars time took: ", (endTime - startTime))
        elif self.embeddingType == EmbeddingType.PARABOLAS:
            print("Parabolas")
        else:
            print("Error! Don't recognize embedding type: ", self.embeddingType)

    def getDataFrame(self):
        print("Getting data frame")