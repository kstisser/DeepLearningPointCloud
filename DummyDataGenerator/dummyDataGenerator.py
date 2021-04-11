from random import random
from Preprocessing import randomSampler
import numpy as np
from DataTools import pointCloud

class DummyDataGenerator:
    def __init__(self):
        print("Opening dummy data generator")
        self.randomSampler = randomSampler.RandomSampler()

    def getGeneratedDataset(self):
        widths = [400, 100, 50, 300, 80]
        heights = [300, 200, 50, 200, 120]
        pointClouds = []

        for i in range(len(widths)):
            allData, allLabels, withoutFace, faceOnly = self.getRectangularPointCloud(widths[i], heights[i])
            pc = pointCloud.generatedPointCloud(allLabels, faceOnly, allData, withoutFace, defs.StructureType.VOXEL, defs.DownsampleType.RANDOM)
            pointClouds.append(pc)
        return pointClouds


    def getRectangularPointCloud(self, width, height, addNoise=True, sparseVal = 1, resolution=240000, xmin=-1.4, xmax=1.4, ymin=-1.4, ymax = 1.4, zmin = -1.4, zmax = 1.4):
        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin

        #pc = np.zeros((int(xspan*resolution), int(yspan*resolution), int(zspan*resolution)))
        center = (int(xspan/2)*resolution, int(yspan/2)*resolution, int(zspan/2)*resolution)
        startPoint = (int(center[0] - size/2), int(center[1] - size/2), int(center[2] - size/2))
        endPoint = (int(center[0] + size/2), int(center[1] + size/2), int(center[2] + size/2))

        #make random noise to the top
        randNumTopPoints1D = random * min((startPoint[0] - 3), (startPoint[1] - 3), (startPoint[2]-3))
        randomTop = self.randomSampler.randomlySamplePoints(pc[0:(startPoint[0]-3), 0:(startPoint[1]-3), 0:startPoint[2]-3], randNumTopPoints1D)
        randomTopLabels = np.zeros(len(randomTop)).T
        print("Top noise shape: ", randomTop.shape, " and labels: ", randomTopLabels.shape)

        #make rectangle
        x, y, z = np.mgrid[startPoint[0]:endPoint[0]:sparseVal, startPoint[1]:endPoint[1]:sparseVal, startPoint[2]:endPoint[2]:sparseVal]
        rectanglePts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        rectangleLabels = np.ones(len(rectanglePts)).T
        print("Rectangle shape: ", rectanglePts.shape, " and labels: ", rectabgleLabels.shape)

        #make random noise to the top
        randNumBottomPoints1D = random * min((endPoint[0]+3), (endPoint[1]+3), (endPoint[2]+3))
        randomBottom = self.randomSampler.randomlySamplePoints(pc[0:(endPoint[0]+3), 0:(endPoint[1]+3), 0:endPoint[2]+3], randNumBottomPoints1D)
        randomBottomLabels = np.zeros(len(randomBottom)).T
        print("Bottom noise shape: ", randomBottom.shape, " and labels: ", randomBottomLabels.shape)

        allData = np.vstack(randomTop, rectanglePts, randomBottom)
        labels = np.vstack(randomTopLabels, rectangleLabels, randomBottomLabels)
        withoutFace = np.vstack(randomTop, randomBottom)
        return allData, labels, withoutFace, rectanglePts