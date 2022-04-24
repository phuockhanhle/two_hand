import os
import wave

from scipy.io import wavfile
from tensorflow import keras
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


def build_dataset_image():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    # x_train = x_train.astype("float32")
    x_test = x_test.astype("float32") / 255
    # x_test = x_test.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train_cat, y_test_cat


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = np.frombuffer(frames, np.int16)
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


def convert_wax_to_image(INPUT_DIR, OUTPUT_DIR):
    for idx, filename in tqdm(enumerate(os.listdir(INPUT_DIR))):
        # if idx == 10:
        #     return
        if "wav" in filename:
            file_path = os.path.join(INPUT_DIR, filename)
            file_stem = Path(file_path).stem
            file_dist_path = os.path.join(OUTPUT_DIR, file_stem) + '.png'
            frame_rate, data = wavfile.read(file_path)
            plt.ioff()
            plt.axis('off')
            plt.specgram(data, Fs=frame_rate)
            plt.savefig(file_dist_path)
            plt.clf()


def build_dataset_voice(source_dir):
    X = []
    y = []
    for idx, file_name in tqdm(enumerate(os.listdir(source_dir))):
        file_path = os.path.join(source_dir, file_name)
        label = int(file_name.split('_')[0])
        data = cv2.imread(file_path)
        data = cv2.resize(data, (256, 256))
        X.append(data)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    X = X.astype("float32") / 255
    y = keras.utils.to_categorical(y, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    return X_train, X_test, y_train, y_test


def build_dataset_mixed(num_classes):
    x_train_by_class, x_test_by_class = get_image_data_by_class(num_classes)
    x_train_voice, x_test_voice, y_train_voice, y_test_voice = build_dataset_voice(source_dir='./data/prepro_dataset')

    x_train_image_for_mix = get_image_by_labels(x_train_by_class, y_train_voice, num_classes)
    x_test_image_for_mix = get_image_by_labels(x_test_by_class, y_test_voice, num_classes)

    return np.stack(x_train_image_for_mix), np.stack(x_test_image_for_mix), x_train_voice, x_test_voice, y_train_voice, y_test_voice


def get_image_by_labels(x_by_class, labels, num_classes):
    x = []
    list_index = [0 for _ in range(num_classes)]
    for y in labels:
        digit = np.where(y == 1)[0][0]
        idx = list_index[digit]
        x.append(x_by_class[digit][idx])
        list_index[digit] += 1
    return x


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
