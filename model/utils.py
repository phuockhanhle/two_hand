def optimize_model(model):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


def train(*, model, x_train, y_train,
          x_val=None, y_val=None,
          batch_size=32, epochs=5):
    if x_val is None or y_val is None:
        model.fit(x_train, y_train, batch_size=batch_size,
                  epochs=epochs)
    else:
        model.fit(x_train, y_train, batch_size=batch_size,
                  epochs=epochs, validation_data=(x_val, y_val))


def evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    voice_score, image_score = None, None
    if "output_1_accuracy" in score.keys():
        print(f"voice accuracy :{score['output_1_accuracy']}")
        voice_score = score['output_1_accuracy']
    if "output_2_accuracy" in score.keys():
        print(f"image accuracy :{score['output_2_accuracy']}")
        image_score = score['output_2_accuracy']
    return voice_score, image_score
