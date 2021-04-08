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
    def __init__(self, ID, center, maxPointsPerPillar):
        self.ID = ID
        self.center = center
        self.D = np.empty((0,3))
        self.maxPointsPerPillar = maxPointsPerPillar
        self.isEmpty = True

    #returns number of rows
    def getNumberOfEntries(self):
        return len(self.D)

    #compute Euclidean distance from a point to the center of the pillar
    def getDistanceFromCenter(self, point):
        x = self.center[0]
        y = self.center[1]
        return np.sqrt((x-point[0])**2 + (y-point[1])**2)

    #add a point to the pillar
    def addPoint(self, x, y, z):
        self.D = np.vstack([self.D, [x,y,z]])
        self.isEmpty = False

    #Now that we have all points, compute the center and add the columns for c & p subscripts
    def finalizePillar(self):
        self.nonZero = len(self.D)
        if(not self.isEmpty):
            self.computeCenterMean()
            self.addColumns()
            #randomly sample pillars if too many points, or zero pad if too few
            if self.getNumberOfEntries() > self.maxPointsPerPillar:
                self.randomlyDownsample()
            elif self.getNumberOfEntries() < self.maxPointsPerPillar:
                self.zeroPad()            
        else:
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

class PointPillars:
    def __init__(self, data):
        print("Opening point pillars")
        self.data = data
        self.pillarsDic = {}
        self.minY = min(self.data[:,0])
        self.maxY = max(self.data[:,0])
        self.minX = min(self.data[:,1])
        self.maxX = max(self.data[:,1])
        self.Xspan = self.maxX - self.minX
        self.Yspan = self.maxY - self.minY
        self.visual = visualizer.Visualizer()

    #Intending to separate data into point pillars size (140x100)
    #@numba.jit(nopython=True)
    def buildPillars(self, pillarDimensions=defs.ppDimensions, maxPointsPerPillar=defs.maxParams):
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
                self.pillars[rowIdx, colIdx] = Pillar(IDcount, center, maxPointsPerPillar)
                self.pillarsDic[IDcount] = self.pillars[rowIdx, colIdx]
                IDcount = IDcount + 1

        #assign and add points to each pillar
        for point in self.data:
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
            self.pillarsDic[(tempPillarDic[minDistance])].addPoint(point[0], point[1], point[2])

        countNonemptyPillars = 0
        for pRows in self.pillars:
            for pillar in pRows:
                pillar.finalizePillar()
                if pillar.getNumberOfEntries() > 0:
                    countNonemptyPillars = countNonemptyPillars + 1

        print("Nonempty pillars: ", countNonemptyPillars)
        #self.visual.visualizePillars(self.pillars, (300,400), maxPointsPerPillar, self.minX, self.maxX, self.minY, self.maxY)
