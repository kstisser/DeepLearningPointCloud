import matplotlib.pyplot as plt
import numpy as np
import open3d
# import cv2 as cv

class Visualizer:
    def __init__(self):
        print("Opening visualizer")

    def visualizeFaceAndAllPlot(self, facePoints, allPoints):
        fig = plt.figure(figsize=[10,20])
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(allPoints[:,0], allPoints[:,1], allPoints[:,2], alpha=0.1, marker='o')
        ax.view_init(-140,120)

        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(facePoints[:,0], facePoints[:,1], facePoints[:,2], alpha=0.1, marker='o')
        ax.view_init(-140,120) 

        labeled_data = np.concatenate((facePoints,allPoints),axis = 0)
        labels = np.concatenate((np.ones(facePoints.shape[0]), np.ones(allPoints.shape[0])*2))

        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(labeled_data[:,0], labeled_data[:,1], labeled_data[:,2], c = labels, alpha=1, marker='.')
        ax.view_init(-140, 120)

        face_set = set([tuple(p) for p in facePoints])      ## Convert the list of points into a set of points. Sets can contain tuples but not lists or np.arrays so se must typecast it.
        samp_labels = [tuple(p) in face_set for p in allPoints] ## Construct a list using list builder notation: this should be read for p in all_samp, put (tuple(p) in face_set) into the list. 
                                                  ## Note that (tuple(p) in face_set) is a true or false value. 
        
        ax = fig.add_subplot(224, projection='3d')
        ax.scatter(allPoints[:,0], allPoints[:,1], allPoints[:,2], c = samp_labels, alpha=1, marker='.')
        ax.view_init(-140, 120)

        plt.show()   

    def visualizePillars(self, pillars, imgShape, maxPointsPerPillar, xmin, xmax, ymin, ymax):
        #generate a 3 channel image for visualization
        img = np.zeros((imgShape[0],imgShape[1],3), np.uint8)               
        buffer = 3
        for pRows in pillars:
            for pillar in pRows:
                centerX = (pillar.center[0] - xmin)/(xmax-xmin) * imgShape[0]
                centerY = (pillar.center[1] - ymin)/(ymax-ymin) * imgShape[1]
                #print("Centerx: ", centerX, " centery: ", centerY)
                upperLeft = (int(max(centerX-buffer,0)), int(max(centerY-buffer,0)))
                lowerRight = (int(min(centerX+buffer, imgShape[0]-1)), int(min(centerY+buffer, imgShape[1]-1)))
                #print("Upper left: ", upperLeft)
                #print("Lower right: ", lowerRight)   
                #print("Center: ", pillar.center)              
                if pillar.isEmpty:
                    print("is empty")
                    #color empty pillars green                   
                    ## img = cv.rectangle(img, upperLeft, lowerRight, (0,200,0), -1)
                else:
                    #get filled ratio
                    unfilledRatio = 1.0 - float(float(pillar.nonZero)/float(maxPointsPerPillar))
                    colorVal = int(unfilledRatio * 200 + 54)
                    #print("Colorval: ", colorVal)
                    ##img = cv.rectangle(img, upperLeft, lowerRight, (0,0,colorVal), -1)
        # print("Image size: ", img.shape)
        # cv.imshow('Point Pillars',img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def visualizeClusters(self, dataToPredict, labels):
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(dataToPredict[:,0], dataToPredict[:,1], dataToPredict[:,2], c = labels, alpha=1, marker='.')
        ax.view_init(-140, 120)
        plt.show()
