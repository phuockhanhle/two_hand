def optimize_model(model):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


def train(model, x_train, y_train, batch_size, epochs, validation_split=0.1):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def evaluate(model, x_test, y_test):
    print("shape y test ", y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
