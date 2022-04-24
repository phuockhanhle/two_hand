import numpy as np
import model.model_voice as model_voice
import model.model_image as model_image
import model.model_mix as model_mix
import model.utils as model_utils
import preprocess

if __name__ == '__main__':
    # Train image model
    x_train_image, x_test_image, y_train_image, y_test_image = preprocess.build_dataset_image()
    image_model = model_image.ImageModel()
    model_utils.optimize_model(image_model)
    model_utils.train(model=image_model, x_train=x_train_image, y_train=y_train_image,
                      batch_size=64, epochs=5)
    print("validation of image model only")
    image_model.evaluate(x_test_image, y_test_image)

    # Train voice model
    x_train_voice, x_test_voice, y_train_voice, y_test_voice = preprocess.build_dataset_voice("./data/prepro_dataset")
    voice_model = model_voice.VoiceModel()
    model_utils.optimize_model(voice_model)
    model_utils.train(model=voice_model, x_train=x_train_voice, y_train=y_train_voice,
                      batch_size=32, epochs=20)
    print("validation of voice model only")
    voice_model.evaluate(x_test_voice, y_test_voice)

    # Mixed model
    x_train_image, x_test_image, = preprocess.build_dataset_mixed(num_classes=10,
                                                                  y_train_voice=y_train_voice,
                                                                  y_test_voice=y_test_voice,
                                                                  x_train_image=x_train_image,
                                                                  x_test_image=x_test_image,
                                                                  y_train_image=y_train_image,
                                                                  y_test_image=y_test_image)

    mix_model = model_mix.MixedModel(image_model, voice_model)
    model_utils.optimize_model(mix_model)
    model_utils.train(model=mix_model, x_train=[x_train_voice, x_train_image],
                      y_train=[y_train_voice, y_train_voice],
                      batch_size=32, epochs=10)
    print("validation have image to predict voice")
    mix_model.evaluate([np.zeros(x_test_voice.shape), x_test_image], [y_test_voice, y_test_voice])
    print("validation have voice to predict image")
    mix_model.evaluate([x_test_voice, np.zeros(x_test_image.shape)], [y_test_voice, y_test_voice])
    print("validation have both image and voice")
    mix_model.evaluate([x_test_voice, x_test_image], [y_test_voice, y_test_voice])
    print(f"output 1 acc : voice acc ||| output 2 acc : image acc")

    # print("save model")
    # image_model.save("./data/pretrain_model/image_model")
    # voice_model.save("./data/pretrain_model/voice_model")
    # mix_model.save("./data/pretrain_model/mix_model")
