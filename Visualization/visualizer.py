import matplotlib.pyplot as plt
import numpy as np
import open3d

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