import tensorflow as tf
import os
from classification_models.keras import Classifiers
from common2 import train
import sys
from contextlib import redirect_stdout

#Returns network, preprocess_input and layers to finetune for the given network
def get_network(name):
    if name == "ResNet50V2":
        return tf.keras.applications.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input, 3
    elif name == "ResNet50":
        return tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input, 2
    elif name == "ResNet101":
        return tf.keras.applications.ResNet101, tf.keras.applications.resnet.preprocess_input, 2
    elif name == "EfficientNetB0":
        return tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB1":
        return tf.keras.applications.EfficientNetB1, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB2":
        return tf.keras.applications.EfficientNetB2, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB3":
        return tf.keras.applications.EfficientNetB3, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB4":
        return tf.keras.applications.EfficientNetB4, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB5":
        return tf.keras.applications.EfficientNetB5, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB6":
        return tf.keras.applications.EfficientNetB6, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetB7":
        return tf.keras.applications.EfficientNetB7, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == "EfficientNetV2B0":
        return tf.keras.applications.EfficientNetV2B0, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == "EfficientNetV2B1":
        return tf.keras.applications.EfficientNetV2B1, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == "EfficientNetV2B2":
        return tf.keras.applications.EfficientNetV2B2, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == "EfficientNetV2B3":
        return tf.keras.applications.EfficientNetV2B3, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == "VGG16":
        return tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input, 1
    elif name == "DenseNet169":
        return tf.keras.applications.DenseNet169, tf.keras.applications.densenet.preprocess_input, 3
    elif name == "ResNet34":
        return Classifiers.get('resnet34'), 3 # "Keras Zoo" models requires a ImageNet weight file located at /weights
    elif name == "InceptionV3":
        return tf.keras.applications.InceptionV3 , tf.keras.applications.inception_v3.preprocess_input, 3
    else:
        print("Network not found")

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory " , path ,  " Created")
    else:
        print("Directory " , path ,  " already exists")




tf.keras.backend.set_image_data_format('channels_last')
PATH = "samples" #Head path to dataset samples
SAVE_PATH = "res/"
samples = [512]

networks_to_train = ["VGG16", "ResNet50V2"]

for network_name in networks_to_train:
    SAVE_PATH = SAVE_PATH + network_name
    network, preprocess_input, layers_to_fine_tune = get_network(network_name)

    create_path(SAVE_PATH)

    for sample in samples:
        SAVE_PATH_FOR_SAMPLE = SAVE_PATH + "/{}".format(sample)

        create_path(SAVE_PATH_FOR_SAMPLE)

        #Redirect stdout to file to save training data
        with open(SAVE_PATH_FOR_SAMPLE + "/log.log", 'w') as f, redirect_stdout(f):
            train(
                network = network,
                preprocess_input = preprocess_input,
                PATH = PATH + "/{}".format(sample),
                SAVE_PATH = SAVE_PATH_FOR_SAMPLE,
                layers_to_fine_tune = layers_to_fine_tune
            )


        print("Finished " + SAVE_PATH_FOR_SAMPLE)


