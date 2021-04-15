import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DataTools import defs
from tensorflow.keras import regularizers

class Model:
    def __init__(self):
        print("Starting model")

    def train(self, trainData, trainLabels, testData, testLabels):
        print("Incoming training data shape: ", trainData.shape)
        print("Incoming training label length: ", (trainLabels.shape))
        print("Incoming testing data shape: ", testData.shape)
        print("Incoming testing label length: ", (testLabels.shape))        

        rate = 0.3
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(defs.max_pillars, defs.max_points, defs.num_features)),
            tf.keras.layers.Dropout(rate),       
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1200)
        ])  
        '''tf.keras.layers.Conv2D(defs.nb_channels, (1, 1), activation='linear', use_bias=False, name="pillars/conv2d"),
            tf.keras.layers.BatchNormalization(name="pillars/batchnorm", fused=True, epsilon=1e-3, momentum=0.99),
            tf.keras.layers.Activation("relu", name="pillars/relu"),
            #tf.keras.layers.MaxPool2D((1, defs.max_points), name="pillars/maxpooling2d"),  '''   
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy", "mae"])   

        history = model.fit(trainData, trainLabels, epochs=10, 
                   validation_data=(testData, testLabels), batch_size=5, verbose=True)  

        model.evaluate(testData, testLabels, verbose=1)           

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss vs. epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.show()            