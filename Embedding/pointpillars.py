# import numba
import numpy as np
from Visualization import visualizer
from DataTools import defs 

'''
This class holds all information needed for a pillar
x, y, z- location of the point
xc, yc, zc- Distance from the arithmetic mean of the pillar c the point belongs to in each dimension
xp, yp- Distance of the point from the center of the pillar in the x-y coordinate system
'''
class Pillar:
    def __init__(self, ID, center, maxPointsPerPillar, xmin, xmax, ymin, ymax, zmin, zmax):
        self.ID = ID
        self.isFace = False
        self.center = center
        self.normalizedCenter = [(self.center[0]-xmin)/(xmax-xmin), (self.center[1]-ymin)/(ymax-ymin)]
        self.picCenter = [(self.center[0])/(defs.ppDimensions[0]), (self.center[1])/(defs.ppDimensions[1])]
        #print("X min: ", xmin, " Y min: ", ymin)
        #print("Normalized center: ", self.normalizedCenter)
        self.D = np.empty((0,3))
        self.maxPointsPerPillar = maxPointsPerPillar
        self.isEmpty = True
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    #returns number of rows
    def getNumberOfEntries(self):
        return len(self.D)

    #compute Euclidean distance from a point to the center of the pillar
    def getDistanceFromCenter(self, point):
        x = self.center[0]
        y = self.center[1]
        return np.sqrt((x-point[0])**2 + (y-point[1])**2)

    #add a point to the pillar
    def addPoint(self, x, y, z, facePoint):
        self.D = np.vstack([self.D, [x,y,z]])
        self.isEmpty = False
        if facePoint:
            self.isFace = True

    def normalizeZeroToOne(self):
        self.D[:,0] = np.transpose(np.array([(x - self.xmin)/(self.xmax-self.xmin) for x in self.D[:,0]]))
        self.D[:,1] = np.transpose(np.array([(y - self.ymin)/(self.ymax-self.ymin) for y in self.D[:,1]]))
        self.D[:,2] = np.transpose(np.array([(z - self.zmin)/(self.zmax-self.zmin) for z in self.D[:,2]]))

    #Now that we have all points, compute the center and add the columns for c & p subscripts
    def finalizePillar(self):
        self.nonZero = len(self.D)
        if(not self.isEmpty):
            self.normalizeZeroToOne()
            self.computeCenterMean()
            self.addColumns()
            #randomly sample pillars if too many points, or zero pad if too few
            if self.getNumberOfEntries() > self.maxPointsPerPillar:
                self.randomlyDownsample()
            elif self.getNumberOfEntries() < self.maxPointsPerPillar:
                self.zeroPad()            
        else:
            #add zeros in unused columns to make the same dimensions
            if(len(self.D) == 0):
                self.D = np.zeros((self.maxPointsPerPillar, defs.num_features))
            else:
                self.D[:,3:7] = np.zeros((len(self.D), 5))
            self.zeroPad() 

    #compute x,y,z means, and make D a numpy array
    #@numba.jit(nopython=True)
    def computeCenterMean(self):
        #convert list to np array
        self.D = np.array(self.D)
        self.xMean = np.mean(self.D[:,0])
        self.yMean = np.mean(self.D[:,1])
        self.zMean = np.mean(self.D[:,2])

    #compute the distance from the mean and distance from the center for each point, and add those columns
    #@numba.jit(nopython=True)
    def addColumns(self):
        #Add 5 columns to the D matrix
        extraEmptyColumns = np.zeros((len(self.D), 5))
        self.D = np.append(self.D, extraEmptyColumns, axis=1)

        #compute distance from the arithmetic mean of the pillar 
        #xc
        self.D[:,3] = np.transpose(np.array([x - self.xMean for x in self.D[:,0]]))
        #yc
        self.D[:,4] = np.transpose(np.array([y - self.yMean for y in self.D[:,1]]))
        #zc
        self.D[:,5] = np.transpose(np.array([z - self.zMean for z in self.D[:,2]]))

        #compute distance from the center of the pillar
        #xp
        self.D[:,6] = np.transpose(np.array([x - self.center[0] for x in self.D[:,0]]))
        #yp
        self.D[:,7] = np.transpose(np.array([y - self.center[0] for y in self.D[:,1]]))

    #put rows of zeros to pad the difference to get max number of points allowed
    def zeroPad(self):
        numRows = self.maxPointsPerPillar - len(self.D)
        self.D = np.vstack([self.D, np.zeros((numRows, self.D.shape[1]))])
        #print("Padding: ", numRows)

    #Randomly pick indices to keep to make sure pillar only has max points allowed in it
    def randomlyDownsample(self):
        print("Downsampling from: ", len(self.D))
        randomIndices = np.random.choice(self.D.shape[0], size=self.maxPointsPerPillar, replace=False)
        self.D = self.D[randomIndices, :]

    def getVec(self):
        #print("Confirming D is dimensions 100x8: ", self.D.shape)
        return self.D

class PointPillars:
    def __init__(self, data, labels):
        print("Opening point pillars")
        self.data = data
        self.labels = labels
        self.pillarsDic = {}
        self.minY = min(self.data[:,0])
        self.maxY = max(self.data[:,0])
        self.minX = min(self.data[:,1])
        self.maxX = max(self.data[:,1])
        self.minZ = min(self.data[:,2])
        self.maxZ = max(self.data[:,2])
        self.Xspan = self.maxX - self.minX
        self.Yspan = self.maxY - self.minY
        self.visual = visualizer.Visualizer()

    #Intending to separate data into point pillars size (140x100)
    #@numba.jit(nopython=True)
    def buildPillars(self, pillarDimensions=defs.ppDimensions, maxPointsPerPillar=defs.max_points):
        #generate centroid points for each pillar
        #note- adding 1 to the dimensions so we can remove the first, and shift the span left
        self.pillars = np.empty((pillarDimensions[0], pillarDimensions[1]), dtype=Pillar)
        xshift = self.Xspan/(pillarDimensions[1] + 1)
        yshift = self.Yspan/(pillarDimensions[0] + 1)
        colVals = (np.arange(self.minX, self.maxX, xshift))[:-1] + (xshift/2.0)
        rowVals = (np.arange(self.minY, self.maxY, yshift))[:-1] + (yshift/2.0)
        IDcount = 1
        #print(rowVals)
        #print(colVals)
        for rowIdx in range(pillarDimensions[0]):
            for colIdx in range(pillarDimensions[1]):
                center = (rowVals[rowIdx], colVals[colIdx])
                self.pillars[rowIdx, colIdx] = Pillar(IDcount, center, maxPointsPerPillar, self.minX, self.maxX, self.minY, self.maxY, self.minZ, self.maxZ)
                self.pillarsDic[IDcount] = self.pillars[rowIdx, colIdx]
                IDcount = IDcount + 1

        #assign and add points to each pillar
        for i in range(len(self.data)):
            point = self.data[i]
            tempPillarDic = {}
            distances = []
            for pRow in self.pillars:
                for pillar in pRow:
                    #this could overwrite data, and that's fine as we just need the min matching value
                    distance = pillar.getDistanceFromCenter(point)
                    tempPillarDic[distance] = pillar.ID 
                    distances.append(distance)
            minDistance = min(np.array(distances))
            #print("Found min distance to ID: ", (tempPillarDic[minDistance]))
            #Add point to the pillar it matched with, and is the closest to
            facePoint = self.labels[i]
            self.pillarsDic[(tempPillarDic[minDistance])].addPoint(point[0], point[1], point[2], facePoint)

        pillarData = []
        countNonemptyPillars = 0
        labelData = np.zeros(self.pillars.shape[0] * self.pillars.shape[1])
        index = 0
        for pRows in self.pillars:
            for pillar in pRows:
                pillar.finalizePillar()
                pillarData.append(pillar.getVec())
                if pillar.isFace:
                    labelData[index] = 1
                if pillar.getNumberOfEntries() > 0:
                    countNonemptyPillars = countNonemptyPillars + 1
                index = index + 1

        pillarData = np.array(pillarData)
        print("Nonempty pillars: ", countNonemptyPillars)
        #self.visual.visualizePillars(self.pillars, (defs.ppDimensions[0]*10,defs.ppDimensions[1]*10), maxPointsPerPillar)
        print("Confirming pillar data shape is 1200 x 100 x 8: ", pillarData.shape)
        return pillarData, labelData

    def compareLabels(self, testLabels):
        count = 0
        countCorrect = 0
        countLabelledFaceIncorrectly = 0
        countMissedFace = 0
        for pRows in self.pillars:
            for pillar in pRows:
                
                if pillar.isFace:
                    if testLabels[count] == 1:
                        countCorrect = countCorrect + 1
                    else:
                        countMissedFace = countMissedFace + 1
                else:
                    if testLabels[count] == 0:
                        countCorrect == countCorrect + 1
                    else:
                        countLabelledFaceIncorrectly = countLabelledFaceIncorrectly + 1
                count = count + 1

        print("Total: ", count, " labelled correctly: ", countCorrect, " Missed face: ", countMissedFace, " Count labelled face incorrectly: ", countLabelledFaceIncorrectly)
        print("Percentages, correct: ", countCorrect/count, " Missed face: ", countMissedFace/count, " Labelled face incorrectly: ", countLabelledFaceIncorrectly/count)
        print("Ratio of incorrect labels, missed faces: ", countMissedFace/(countMissedFace + countLabelledFaceIncorrectly), " labelled face incorrectly: ", countLabelledFaceIncorrectly/(countMissedFace + countLabelledFaceIncorrectly))