import os
from tensorflow import keras
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def build_dataset_image():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


def get_dataset_voice(DATASET_PATH):
    mfcc_list, digit = [], []
    for idx, file_name in enumerate(tqdm(os.listdir(DATASET_PATH))):
        name = file_name.split('_')
        digit.append(name[0])
        wav, sr = librosa.load(os.path.join(DATASET_PATH, file_name))
        mfcc = librosa.feature.mfcc(y=wav)
        mfcc_list.append(mfcc)

    return mfcc_list, digit


def get_max_shape(mfcc_list):
    shape_col = np.max([mfcc.shape[1] for mfcc in mfcc_list])
    shape_row = np.max([mfcc.shape[0] for mfcc in mfcc_list])

    return shape_row, shape_col


def padding(mfcc, max_shape_col, max_shape_roz=0):
    mfcc = np.hstack((mfcc, np.zeros((mfcc.shape[0], max_shape_col - mfcc.shape[1]))))
    # np.hstack((mfcc, np.zeros((mfcc.shape[1], max_shape_roz - mfcc.shape[0]))))
    return mfcc


def build_dataset_voice(mfcc_list, digit, path_prepro):
    data_dict = {
        'x_train': None,
        'x_test': None,
        'y_train': None,
        'y_test': None,
    }

    if not os.path.exists(os.path.join(path_prepro, 'x_train.npy')):
        shape_row, shape_col = get_max_shape(mfcc_list)

        x, y = [], []

        for idx, mfcc in enumerate(tqdm(mfcc_list)):
            mfcc = padding(mfcc, shape_col)
            x.append(mfcc)
            y.append(digit[idx])

        x = np.array(x)
        x = np.expand_dims(x, -1)
        y = keras.utils.to_categorical(np.array(y), 10)

        data_dict['x_train'], data_dict["x_test"], data_dict['y_train'], data_dict["y_test"] = train_test_split(
            x, y, test_size=0.33, random_state=42)

        for name, value in data_dict.items():
            np.save(os.path.join(path_prepro, name), value)
    else:

        for name in data_dict.keys():
            data_dict[name] = np.load(os.path.join(path_prepro, f"{name}.npy"))

    return data_dict['x_train'], data_dict["x_test"], data_dict['y_train'], data_dict["y_test"]


def build_dataset_mixed(num_classes):
    x_train_by_class, x_test_by_class = get_image_data_by_class(num_classes)
    x_train_voice, x_test_voice, y_train_voice, y_test_voice = build_dataset_voice(mfcc_list=None,
                                                                                   digit=None,
                                                                                   path_prepro="./data/prepro_dataset")

    nb_samples_train_per_digit = [np.sum(np.where(y_train_voice == 1)[1] == digit) for digit in range(num_classes)]
    x_train_image_for_mix = []
    for digit, nb_samples in enumerate(nb_samples_train_per_digit):
        x_train_image_for_mix.extend(x_train_by_class[digit][:nb_samples])

    nb_samples_test_per_digit = [np.sum(np.where(y_test_voice == 1)[1] == digit) for digit in range(num_classes)]
    x_test_image_for_mix = []
    for digit, nb_samples in enumerate(nb_samples_test_per_digit):
        x_test_image_for_mix.extend(x_test_by_class[digit][:nb_samples])

    print(len(x_train_image_for_mix))

    return np.stack(x_train_image_for_mix), np.stack(x_test_image_for_mix), x_train_voice, x_test_voice, y_train_voice, y_test_voice


def get_image_data_by_class(num_class):
    x_train_by_class = [[] for _ in range(num_class)]
    x_test_by_class = [[] for _ in range(num_class)]

    x_train_image, x_test_image, y_train_image, y_test_image = build_dataset_image()

    for idx, y in enumerate(y_train_image[:]):
        digit = np.where(y == 1)[0][0]
        x_train_by_class[digit].append(x_train_image[idx])

    for idx, y in enumerate(y_test_image[:]):
        digit = np.where(y == 1)[0][0]
        x_test_by_class[digit].append(x_test_image[idx])

    return x_train_by_class, x_test_by_class
