import tensorflow as tf
import numpy as np

class PointPillarFeatureNet:
    def __init__(self, data):
        self.data = data
        self.max_pillars = 14000
        self.max_points_per_pillar = 100
        self.num_features = 8
        self.batch_size = 4
        self.input_shape = (self.max_pillars, self.max_points_per_pillar, self.num_features)

    #expects input layer to be (Batch size, max # pillars, max # points per pillar, # features in Point Pillar)
    # ie (4,12000,100,9)
    def feedForward(self):
        input_pillars = tf.keras.layers.Input(self.input_shape, self.batch_size, name="pillars/input")
        input_indices = tf.keras.layers.Input((self.max_pillars, 3), batch_size=self.batch_size, name="pillars/indices", dtype=tf.float32)
        #linear (conv2d 64?)

        #Batch normalization

        #relu

        #max pooling


    def correct_batch_indices(self, tensor):
        array = np.zeros((self.batch_size, self.max_pillars, 3), dtype=np.float32)
        for i in range(self.batch_size):
            array[i, :, 0] = i
        return tensor + tf.constant(array, dtype=tf.float32)