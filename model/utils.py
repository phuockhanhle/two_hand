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
    model.evaluate(x_test, y_test, verbose=0)
