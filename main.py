import numpy as np

import model as md
import preprocess
import wave
import pylab


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


if __name__ == '__main__':
    # x_train_image, x_test_image, y_train_image, y_test_image = preprocess.build_dataset_image()
    # x_train_voice, x_test_voice, y_train_voice, y_test_voice = preprocess.build_dataset_voice(mfcc_list=None, digit=None, path_prepro="./data/prepro_dataset")

    x_train_image, x_test_image, x_train_voice, x_test_voice, y_train_voice, y_test_voice = preprocess.build_dataset_mixed(num_classes=10)


    # model = md.create_model_image()

    model = md.MixedModel(num_classes=10, input_image_shape=(28, 28, 1), input_voice_shape=(20, 33, 1))
    # model.summary()

    md.optimize_model(model)

    # md.train(model, x_train, y_train, 64, 10)
    # model.fit(x_train_image, y_train_image, batch_size=64, epochs=3)
    model.fit([x_train_image, x_train_voice], [y_train_voice, y_train_voice], batch_size=64, epochs=15)

    print("validation")
    # model.evaluate([np.zeros(x_test_image.shape), x_test_voice], [y_test_voice, y_test_voice])
    model.evaluate([x_test_image, np.zeros(x_test_voice.shape)], [y_test_voice, y_test_voice])

    # md.evaluate(model, x_test, y_test)
    #
    # md.debug()



    # mfcc, digit = preprocess.get_dataset_voice("./data/free-spoken-digit-dataset-master/recordings")

    # x_train, x_test, y_train, y_test = preprocess.build_dataset_voice(mfcc_list=None, digit=None, path_prepro="./data/prepro_dataset")
    #
    # model = md.create_model_voice(x_train[0].shape)
    #
    # md.optimize_model(model)
    #
    # md.train(model, x_train, y_train, 64, 10)
    #
    # md.evaluate(model, x_test, y_test)

    # md.debug()


