def optimize_model(model):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


def train(*, model, x_train, y_train,
          x_val=None, y_val=None,
          batch_size=32, epochs=5, validation_split=0.1):
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def evaluate(model, x_test, y_test):
    print("shape y test ", y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
