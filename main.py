import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from classification_models.keras import Classifiers
from common2 import train
import sys
from contextlib import redirect_stdout

def get_network(name):
    if name == "ResNet50":
        return tf.keras.applications.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input
    elif name == "ResNet18":
        return Classifiers.get('resnet18') # "Keras Zoo" models requires a ImageNet weight file on the device
    elif name == "InceptionV3":
        return tf.keras.applications.InceptionV3 , tf.keras.applications.inception_v3.preprocess_input
    else:
        print("Network not found")

def create_path(path)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory " , path ,  " Created")
    else:
        print("Directory " , path ,  " already exists")


tf.keras.backend.set_image_data_format('channels_last')
PATH = "samples" #Head path to dataset samples
samples = [512,1024]

networks_to_train = ["ResNet50", "InceptionV3"]

for network_name in networks_to_train:
    SAVE_PATH = network_name
    network, preprocess_input = get_network(network_name)

    create_path(SAVE_PATH

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
                layers_to_fine_tune = 3
            )


        print("Finished " + SAVE_PATH_FOR_SAMPLE)


