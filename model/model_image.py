from tensorflow import keras
from tensorflow.keras import layers


class ImageModel(keras.Model):
    def __init__(self, num_classes=10):

        super(ImageModel, self).__init__()

        self.base = keras.Sequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
            ]
        )

        self.fcn = layers.Dense(32, activation="relu")
        self.out = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        base_out = self.base(inputs)
        fcn_out = self.fcn(base_out)
        return self.out(fcn_out)
