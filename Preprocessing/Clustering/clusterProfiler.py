
#This stores information on number of points, the width, and the height
#you would expect to get for a certain face. Width and height are in meters.
class FaceProfile:
    def __init__(self, numPoints, width, height):
        self.numPoints = numPoints
        self.width = width
        self.height = height

    def lessThanFace(self, otherFace):
        if otherFace.numPoints < self.numPoints:
            print("Num points too few: ", otherFace.numPoints)
            return True
        if otherFace.width < self.width:
            print("Width too small: ", otherFace.width)
            return True
        if otherFace.height < self.height:
            print("Height too small: ", otherFace.height)
            return True
        return False

    def greaterThanFace(self, otherFace):
        if otherFace.numPoints > self.numPoints:
            print("Num points too many: ", otherFace.numPoints)
            return True
        if otherFace.width > self.width:
            print("Width too big: ", otherFace.width)
            return True
        if otherFace.height > self.height:
            print("Height too big: ", otherFace.height)
            return True
        return False    

class FaceRange:
    def __init__(self, minFace, maxFace):
        self.widthRange = maxFace.width - minFace.width
        self.heightRange = maxFace.height - minFace.height
        self.numPointsRange = maxFace.numPoints - minFace.numPoints

class ClusterProfiler:
    def __init__(self):
        #Note- google search says average width: 0.1524,
        # and average height: 0.1778. The following was calculated
       # from our data (without downsampling)
        #self.averageFace = FaceProfile(28237, 0.1907, 0.2487)
        #calculated min points: 39877, lowering for a buffer
        #self.minFace = FaceProfile(30000, 0.1, 0.2)
        #calculated max points: 63339, increasing for a buffer
        #self.maxFace = FaceProfile(80000, 0.4, 0.5)
        # from our data (with downsampling)
        self.averageFace = FaceProfile(200, 0.1907, 0.2487)
        #calculated min points: 39877, lowering for a buffer
        self.minFace = FaceProfile(100, 0.1, 0.2)
        #calculated max points: 63339, increasing for a buffer
        self.maxFace = FaceProfile(300, 0.4, 0.6)        
        #Use FaceRange to compute stats on face
        self.faceRange = FaceRange(self.minFace, self.maxFace)

    def scoreCluster(self, cluster, threshold):
        score = 0
        clusterFace = self.createFaceProfileFromCluster(cluster)
        if self.minFace.lessThanFace(clusterFace):
            score = -1
            print("Eliminating cluster for being too small")
        elif self.maxFace.greaterThanFace(clusterFace):
            score = -1
            print("Eliminating cluster for being too big")
        else:
            pointsScore = abs(self.averageFace.numPoints - clusterFace.numPoints)/(0.5 * self.faceRange.numPointsRange)
            widthScore = abs(self.averageFace.width - clusterFace.width)/(0.5 * self.faceRange.widthRange)
            heightScore = abs(self.averageFace.height - clusterFace.height)/(0.5 * self.faceRange.heightRange)
            score = pointsScore + widthScore + heightScore
            print("Rated this cluster with point score: ", pointsScore, " width score: ", widthScore, " height score: ", heightScore, " total score: ", score)
        return score

    def createFaceProfileFromCluster(self, cluster):
        #count number of points
        pointCount = len(cluster[:,1])
        #get distance range in height and width
        w = max(cluster[:,1]) - min(cluster[:,1])
        #get max number of points in height and width
        h = max(cluster[:,2]) - min(cluster[:,2])
        #compare to assign height and width
        faceWidth = min(w,h)
        faceHeight = max(w,h)
        #create face
        return FaceProfile(pointCount, faceWidth, faceHeight)