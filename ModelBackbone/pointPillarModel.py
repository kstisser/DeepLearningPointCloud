import numpy as np
from DataTools import defs
class PointPillarModel:
    def __init__(self, modelFileLocation, logDir):
        self.modelFileLocation = modelFileLocation
        self.logDir = logDir

    def createModelBackbone(self, pillars):
        # 2d cnn backbone

        # Block1(S, 4, C)
        x = pillars
        for n in range(4):
            S = (2, 2) if n == 0 else (1, 1)
            x = tf.keras.layers.Conv2D(defs.nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                    name="cnn/block1/conv2d%i" % n)(x)
            x = tf.keras.layers.BatchNormalization(name="cnn/block1/bn%i" % n, fused=True)(x)
        x1 = x

        # Block2(2S, 6, 2C)
        for n in range(6):
            S = (2, 2) if n == 0 else (1, 1)
            x = tf.keras.layers.Conv2D(2 * defs.nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                    name="cnn/block2/conv2d%i" % n)(x)
            x = tf.keras.layers.BatchNormalization(name="cnn/block2/bn%i" % n, fused=True)(x)
        x2 = x

        # Block3(4S, 6, 4C)
        for n in range(6):
            S = (2, 2) if n == 0 else (1, 1)
            x = tf.keras.layers.Conv2D(2 * defs.nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                    name="cnn/block3/conv2d%i" % n)(x)
            x = tf.keras.layers.BatchNormalization(name="cnn/block3/bn%i" % n, fused=True)(x)
        x3 = x

        # Up1 (S, S, 2C)
        up1 = tf.keras.layers.Conv2DTranspose(2 * defs.nb_channels, (3, 3), strides=(1, 1), padding="same", activation="relu",
                                            name="cnn/up1/conv2dt")(x1)
        up1 = tf.keras.layers.BatchNormalization(name="cnn/up1/bn", fused=True)(up1)

        # Up2 (2S, S, 2C)
        up2 = tf.keras.layers.Conv2DTranspose(2 * defs.nb_channels, (3, 3), strides=(2, 2), padding="same", activation="relu",
                                            name="cnn/up2/conv2dt")(x2)
        up2 = tf.keras.layers.BatchNormalization(name="cnn/up2/bn", fused=True)(up2)

        # Up3 (4S, S, 2C)
        up3 = tf.keras.layers.Conv2DTranspose(2 * defs.nb_channels, (3, 3), strides=(4, 4), padding="same", activation="relu",
                                            name="cnn/up3/conv2dt")(x3)
        up3 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(up3)

        # Concat
        concat = tf.keras.layers.Concatenate(name="cnn/concatenate")([up1, up2, up3])
#conv layer over this- same size
#single 1x1 or just this
#dice + bin crossentropy
        if defs.detectionMethod == DetectionMethod.DETECTIONHEAD:
            # Detection head
            occ = tf.keras.layers.Conv2D(defs.nb_anchors, (1, 1), name="occupancy/conv2d", activation="sigmoid")(concat)

            loc = tf.keras.layers.Conv2D(defs.nb_anchors * 3, (1, 1), name="loc/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
            loc = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="loc/reshape")(loc)

            size = tf.keras.layers.Conv2D(defs.nb_anchors * 3, (1, 1), name="size/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
            size = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="size/reshape")(size)

            angle = tf.keras.layers.Conv2D(defs.nb_anchors, (1, 1), name="angle/conv2d")(concat)

            heading = tf.keras.layers.Conv2D(defs.nb_anchors, (1, 1), name="heading/conv2d", activation="sigmoid")(concat)

            clf = tf.keras.layers.Conv2D(defs.nb_anchors * defs.nb_classes, (1, 1), name="clf/conv2d")(concat)
            clf = tf.keras.layers.Reshape(tuple(i // 2 for i in image_size) + (defs.nb_anchors, defs.nb_classes), name="clf/reshape")(clf)

            pillar_net = tf.keras.models.Model([input_pillars, input_indices], [occ, loc, size, angle, heading, clf])
        elif defs.detectionMethod == DetectionMethod.BINARY:
            #What do do here? 

            pillar_net = Dense((20*30*40), activation = sigmoid)

        #????????Should I be loading weights?
        pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))

        #loss
        loss = K.binary_crossentropy(y_true, y_pred)
        masked_loss = tf.boolean_mask(loss, self.mask)
        return self.heading_weight * tf.reduce_mean(masked_loss)

        #optimizer
        optimizer = tf.keras.optimizers.Adam(lr=defs.learning_rate, decay=defs.decay_rate)
        #compile
        pillar_net.compile(optimizer, loss=loss.losses())

    epoch_to_decay = int(
        np.round(defs.iters_to_decay / defs.batch_size * int(np.ceil(float(len(label_files)) / params.batch_size))))
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=self.logDir),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.logDir, "model.h5"),
                                           monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % epoch_to_decay == 0) and (epoch != 0)) else lr, verbose=True),
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
    ]

    training_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files[:-validation_len], label_files[:-validation_len], calibration_files[:-validation_len])
    validation_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files[-validation_len:], label_files[-validation_len:], calibration_files[-validation_len:])


    #     print(pillar_net.summary())    
    # 
    # Train and save
    try:
        pillar_net.fit(training_gen,
                       validation_data = validation_gen,
                       steps_per_epoch=len(training_gen),
                       callbacks=callbacks,
                       use_multiprocessing=True,
                       epochs=int(defs.total_training_epochs),
                       workers=6)
    except KeyboardInterrupt:
        model_str = "interrupted_%s.h5" % time.strftime("%Y%m%d-%H%M%S")
        pillar_net.save(os.path.join(self.logDir, model_str))
        print("Interrupt. Saving output to %s" % os.path.join(os.getcwd(), self.logDir[1:], model_str))   
    