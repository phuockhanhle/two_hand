from tensorflow import keras
from tensorflow.keras import layers


"""
class ImageModel
    base and out  
class VoiceModel
    base and out 

class MixMode 
    init 
    from pretrained ImageModel, VoiceModel
    Freeze base of 2 model component
    train on Mix data, validate on absent data 
    
"""


class MixedModel(keras.Model):
    def __init__(self, num_classes, input_image_shape, input_voice_shape):

        super(MixedModel, self).__init__()

        self.base_image = keras.Sequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
            ]
        )

        self.fcn_voice_to_image = layers.Dense(64, activation="relu")

        self.fcn_image = layers.Dense(32, activation="relu")
        self.dropout_image = layers.Dropout(0.5)
        self.image_out = layers.Dense(num_classes, activation="softmax")

        self.base_voice = keras.Sequential(
            [
                layers.Conv2D(256, kernel_size=(4, 4), activation="relu"),
                layers.MaxPooling2D(pool_size=(4, 4)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
            ]
        )

        self.fcn_image_to_voice = layers.Dense(64, activation="relu")

        self.fcn_voice = layers.Dense(32, activation="relu")
        self.voice_out = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        voice = inputs[0]
        image = inputs[1]

        base_image_out = self.base_image(image)
        base_voice_out = self.base_voice(voice)

        fcn_image_to_voice_out = self.fcn_image_to_voice(base_image_out)
        fcn_voice_out = self.fcn_voice(base_voice_out + fcn_image_to_voice_out)

        fcn_voice_to_image_out = self.fcn_voice_to_image(base_voice_out)
        fcn_image_out = self.fcn_image(base_image_out + fcn_voice_to_image_out)

        return self.voice_out(fcn_voice_out), self.image_out(fcn_image_out)


def create_model_image():
    num_classes = 10
    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    return model


def optimize_model(model):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


def train(model, x_train, y_train, batch_size, epochs, validation_split=0.1):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def evaluate(model, x_test, y_test):
    print("shape y test ", y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def create_model_voice(input_shape):
    num_classes = 10

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(256, kernel_size=(4, 4), activation="relu"),
            layers.MaxPooling2D(pool_size=(4, 4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    return model

def debug():
    return None