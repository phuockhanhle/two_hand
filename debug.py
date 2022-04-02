import preprocess
import matplotlib.pyplot as plt

x_train_image, x_test_image, x_train_voice, x_test_voice, y_train_voice, y_test_voice = preprocess.build_dataset_mixed(
    num_classes=10)

for i in range(5):
    print(y_train_voice[i])
    plt.imshow(x_train_image[i])
    plt.show()
