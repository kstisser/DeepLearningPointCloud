
class PointPillarModel:
    def __init__(self, params: Params):
        self.params = params

    def createModelBackbone(self, pillars):
        # 2d cnn backbone

        # Block1(S, 4, C)
        x = pillars
        for n in range(4):
            S = (2, 2) if n == 0 else (1, 1)
            x = tf.keras.layers.Conv2D(nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                    name="cnn/block1/conv2d%i" % n)(x)
            x = tf.keras.layers.BatchNormalization(name="cnn/block1/bn%i" % n, fused=True)(x)
        x1 = x

        # Block2(2S, 6, 2C)
        for n in range(6):
            S = (2, 2) if n == 0 else (1, 1)
            x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                    name="cnn/block2/conv2d%i" % n)(x)
            x = tf.keras.layers.BatchNormalization(name="cnn/block2/bn%i" % n, fused=True)(x)
        x2 = x

        # Block3(4S, 6, 4C)
        for n in range(6):
            S = (2, 2) if n == 0 else (1, 1)
            x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                    name="cnn/block3/conv2d%i" % n)(x)
            x = tf.keras.layers.BatchNormalization(name="cnn/block3/bn%i" % n, fused=True)(x)
        x3 = x

        # Up1 (S, S, 2C)
        up1 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(1, 1), padding="same", activation="relu",
                                            name="cnn/up1/conv2dt")(x1)
        up1 = tf.keras.layers.BatchNormalization(name="cnn/up1/bn", fused=True)(up1)

        # Up2 (2S, S, 2C)
        up2 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(2, 2), padding="same", activation="relu",
                                            name="cnn/up2/conv2dt")(x2)
        up2 = tf.keras.layers.BatchNormalization(name="cnn/up2/bn", fused=True)(up2)

        # Up3 (4S, S, 2C)
        up3 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(4, 4), padding="same", activation="relu",
                                            name="cnn/up3/conv2dt")(x3)
        up3 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(up3)

        # Concat
        concat = tf.keras.layers.Concatenate(name="cnn/concatenate")([up1, up2, up3])

        # Detection head
        occ = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="occupancy/conv2d", activation="sigmoid")(concat)

        loc = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="loc/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
        loc = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="loc/reshape")(loc)

        size = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="size/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
        size = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="size/reshape")(size)

        angle = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="angle/conv2d")(concat)

        heading = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="heading/conv2d", activation="sigmoid")(concat)

        clf = tf.keras.layers.Conv2D(nb_anchors * nb_classes, (1, 1), name="clf/conv2d")(concat)
        clf = tf.keras.layers.Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="clf/reshape")(clf)

        pillar_net = tf.keras.models.Model([input_pillars, input_indices], [occ, loc, size, angle, heading, clf])
    #     print(pillar_net.summary())        
    