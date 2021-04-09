import tensorflow as tf
import numpy as np
from DataTools import defs

class PointPillarFeatureNet:
    def __init__(self):
        self.max_pillars = defs.max_pillars
        self.max_points = defs.max_points
        self.num_features = defs.num_features
        self.batch_size = defs.batch_size
        self.nb_channels = defs.nb_channels
        self.input_shape = (self.max_pillars, self.max_points, self.num_features)
        self.image_size  = tuple([defs.Xn, defs.Yn])

    #expects input layer to be (Batch size, max # pillars, max # points per pillar, # features in Point Pillar)
    # ie (4,12000,100,9)
    def feedForward(self):
        input_pillars = tf.keras.layers.Input(self.input_shape, self.batch_size, name="pillars/input")
        input_indices = tf.keras.layers.Input((self.max_pillars, 3), batch_size=self.batch_size, name="pillars/indices", dtype=tf.int32)
        #linear (conv2d 64?)
        x = tf.keras.layers.Conv2D(self.nb_channels, (1, 1), activation='linear', use_bias=False, name="pillars/conv2d")(input_pillars)
        x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm", fused=True, epsilon=1e-3, momentum=0.99)(x)
        x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
        x = tf.keras.layers.MaxPool2D((1, self.max_points), name="pillars/maxpooling2d")(x)

        if tf.keras.backend.image_data_format() == "channels_first":
            reshape_shape = (self.nb_channels, self.max_pillars)
        else:
            reshape_shape = (self.max_pillars, self.nb_channels)

        x = tf.keras.layers.Reshape(reshape_shape, name="pillars/reshape")(x)
        pillars = tf.keras.layers.Lambda(lambda inp: tf.scatter_nd(inp[0], inp[1],
                                                                (self.batch_size,) + self.image_size + (self.nb_channels,)),
                                        name="pillars/scatter_nd")([input_indices, x])
    
        #Batch normalization

        #relu

        #max pooling
        return pillars


    def correct_batch_indices(self, tensor):
        array = np.zeros((self.batch_size, self.max_pillars, 3), dtype=np.float32)
        for i in range(self.batch_size):
            array[i, :, 0] = i
        return tensor + tf.constant(array, dtype=tf.float32)