from DataTools import pointCloud
import math
import numpy as np

class DataAugmentation:
    def __init__(self, pointClouds):
        self.originalPointClouds = pointClouds
        self.allPointClouds = pointClouds
        #self.rotationDegree = rotationDegree
        #self.translationLength = translationLength

    def augmentAllData(self):
        #create mixture of face and backgrounds, then augment
        newPointCloudCount = 0
        for background in self.originalPointClouds:
            for face in self.originalPointClouds:
                #only create a new point cloud if the existing does not exist
                if background == face:
                    self.fullyAugmentPointCloud(background)
                else:
                    pc = pointCloud.augmentedPointCloud(background.allWithoutFace, face.faceseg)
                    self.allPointClouds.append(pc)
                    self.fullyAugmentPointCloud(pc)
                    newPointCloudCount = newPointCloudCount + 1
        return self.allPointClouds

    def fullyAugmentPointCloud(self, pc):
        #translate in a spiral form
        b = 0.306 #radians- angle that makes the line tan to the spiral
        a = 1 #distance from one loop to another loop in the spiral
        
        face = pc.faceseg
        background = pc.allWithoutFace

        scaleRanges = [0.94, 0.96, 0.98, 1.02, 1.04, 1.06]
        spiralRange = 30
        for translation in range(spiralRange):
            angle = translation * (4*math.pi)/spiralRange
            rt = math.exp(b * angle)
            x = rt * math.cos(angle) + pc.getFaceCenter()[0]
            y = rt * math.sin(angle) + pc.getFaceCenter()[1]
            center = np.array([y,x])

            xdiff = int(pc.getFaceCenter()[1] - x)
            ydiff = int(pc.getFaceCenter()[0] - y)
            translationMatrix = np.array([ydiff, xdiff, 0])
            translatedPc = pc.face3d.translate(translationMatrix)

            #rotate
            for rotation in range(10):
                angle = 2 * np.pi * (rotation/10.0)
                rotated_Points = self.rotate_point(translatedPc, math.radians(angle),1,1,0)

                #scale (3 smaller 3 bigger) 2% increments
                for scaleVal in range(len(scaleRanges)):
                    scaledPoints = scale(scaleRanges[scaleVal], center, scaleRanges[scaleVal], center)

                    #flop
                    for flop in range(2):
                        floppedPoints = scaledPoints
                        if flop == 0: #generate flopped, otherwise save as is
                            for i in range(len(floppedPoints.shape[1])):
                                floppedPoints[:,i] = x + (x - floppedPoints[:,i])
                            
                        #save face with background in new point cloud object
                        self.allPointClouds.append(pointCloud.augmentedPointCloud(background, floppedPoints))
            
    def rotate_point (self, point, rotation_angle,x,y,z):
        point = np.array(point)
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                    [-sin_theta, cos_theta, 0],
                                    [x, y, z]])
        rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
        return rotated_point