from DataTools import pointCloud
import math

class DataAugmentation:
    def __init__(self, pointClouds, rotationDegree, translationLength):
        self.originalPointClouds = pointClouds
        self.allPointClouds = pointClouds
        self.rotationDegree = rotationDegree
        self.translationLength = translationLength

    def augmentAllData(self):
        #create mixture of face and backgrounds, then augment
        newPointCloudCount = 0
        for background in self.originalPointClouds:
            for face in self.originalPointClouds:
                #only create a new point cloud if the existing does not exist
                if background == face:
                    self.fullyAutmentPointCloud(background)
                else:
                    pc = pointCloud.augmentedPointCloud(background.allWithoutFace, face.faceseg)
                    self.fullyAutmentPointCloud(pc)
                    newPointCloudCount = newPointCloudCount + 1
        return self.allPointClouds

    def fullyAugmentPointCloud(self, pc):
        #translate in a spiral form
        b = 0.306 #radians- angle that makes the line tan to the spiral
        a = 1 #distance from one loop to another loop in the spiral
        
        spiralRange = 30
        for translation in range(spiralRange):
            angle = angle + (4*pi)/spiralRange
            rt = math.exp(b * angle)
            x = rt * math.cos(angle) + pc.center.x
            y = rt * math.sin(angle) + pc.center.y
            center = np.array([x,y])

            '''
            #rotate
            for rotation in range(10):
                #geo3DRot = 

                #scale
                for scale in range(5):

                    #flop
                    for flop in range(2):
            
                        #save
            '''
