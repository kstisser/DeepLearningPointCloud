from enum import Enum

#parameters for the program
maxParams = 20
ppDimensions = (30,40)

#point pillar net
nb_channels = 64
max_points = 100
max_pillars = 1400
batch_size = 4
num_features = 8

#point pillar model
detectionMethod = DetectionMethod.BINARY
decay_rate = 
learning_rate = 0.001
iters_to_decay = 
total_training_epochs = 

#detection head
nb_anchors = 4 #TODO- fix this, and understand what it is for detection head
nb_classes = 2 #face or not face

class DetectionMethod(Enum):
    BINARY = 1
    DETECTIONHEAD = 2

class StructureType(Enum):
    VOXEL = 1
    POINTPILLAR = 2
    PARABOLA = 3

class DownsampleType(Enum):
    NODOWNSAMPLE = 1
    RANDOM = 2

class EmbeddingType(Enum):
    VOXELS = 1
    POINTPILLARS = 2
    PARABOLAS = 3

class ClusterType(Enum):
    SKDBSCAN = 1
    O3DDBSCAN = 2

class FaceBoundingBox:
    def __init__(self):
        self.width = 0.25
        self.height = 0.35
        self.depth = self.width





