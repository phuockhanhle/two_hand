from tensorflow import keras
from tensorflow.keras import layers


class MixedModel(keras.Model):
    def __init__(self, image_model, voice_model, num_classes=10):

        super(MixedModel, self).__init__()

        self.base_image = image_model.base
        self.base_image.trainable = False
        self.base_voice = voice_model.base
        self.base_voice.trainable = False

        self.fcn_image = layers.Dense(32, activation="relu")
        self.dropout_image = layers.Dropout(0.5)
        self.image_out = layers.Dense(num_classes, activation="softmax")

        self.fcn_voice = layers.Dense(32, activation="relu")
        self.voice_out = layers.Dense(num_classes, activation="softmax")

        self.fcn_voice_to_image = layers.Dense(64, activation="relu")
        self.fcn_image_to_voice = layers.Dense(64, activation="relu")

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