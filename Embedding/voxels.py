import numpy as np

class Voxels:
    def __init__(self, voxelSize):
        self.voxelSize= voxelSize

    def voxelate(self, pointCloud):
        max_X = np.max(pointCloud, axis=0)
        min_X = np.min(pointCloud, axis=0)
        print("Min X: ", min_X, ", Max X: ", max_X)
        
        max_y = np.max(pointCloud, axis=1)
        min_y = np.min(pointCloud, axis=1)
        print("Min Y: ", min_y, ", Max Y: ", max_y)

        max_z = np.max(pointCloud, axis=2)
        min_z = np.min(pointCloud, axis=2)
        print("Min z: ", min_z, ", Max Z: ", max_z)        
        return pointCloud