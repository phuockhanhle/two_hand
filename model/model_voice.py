from tensorflow import keras
from tensorflow.keras import layers


class VoiceModel(keras.Model):
    def __init__(self, num_classes=10):

        super(VoiceModel, self).__init__()

        self.base = keras.Sequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
                # layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                # layers.BatchNormalization(),
                layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
                # layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                # layers.BatchNormalization(),
                layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
                # layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                # layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                # layers.BatchNormalization(),
                layers.Dropout(0.5),
            ]
        )

        self.out = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        base_out = self.base(inputs)
        return self.out(base_out)
