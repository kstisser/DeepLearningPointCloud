from enum import Enum



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


#parameters for the program
maxParams = 20
ppDimensions = (30,40)

#point pillar net
nb_channels = 64
max_points = 20
max_pillars = 1200#ppDimensions[0] * ppDimensions[1]
batch_size = 1
num_features = 8

#point pillar model
detectionMethod = DetectionMethod.BINARY
decay_rate = 0.02
learning_rate = 0.001
iters_to_decay = 30
total_training_epochs = 10

#detection head
nb_anchors = 4 #TODO- fix this, and understand what it is for detection head
nb_classes = 2 #face or not face

#loss parameters
alpha = 0.25
gamma = 2.0
focal_weight = 3.0
loc_weight = 2.0
size_weight = 2.0
angle_weight = 1.0
heading_weight = 0.2
class_weight = 0.5

#Image dimensions for pillar net
x_min = 0.0
x_max = 80.64
x_step = 0.16

y_min = -40.32
y_max = 40.32
y_step = 0.16

z_min = -1.0
z_max = 3.0

# derived parameters
Xn_f = float(x_max - x_min) / x_step
Yn_f = float(y_max - y_min) / y_step
#Xn = int(Xn_f)
#Yn = int(Yn_f)
Xn = int(ppDimensions[1])
Yn = int(ppDimensions[0])