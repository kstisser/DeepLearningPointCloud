import numpy as np

class RandomSampler:
    def __init__(self):
        print("Opened random sampler")

    def randomlySamplePoints(self, points, downsampleNumber):
        rows = points.shape[0]
        randomIndices = (np.random.choice(rows, size=downsampleNumber, replace=False))
        downsampled = points[randomIndices,:]   

        print("Downsampled shape: ", downsampled.shape)
        return downsampled
