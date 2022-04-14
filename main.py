import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from classification_models.keras import Classifiers
from common2 import train
import sys
from contextlib import redirect_stdout

def getNetwork(name):
    if name == "ResNet50":
        return tf.keras.applications.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input
    elif name == "ResNet18":
        return Classifiers.get('resnet18') # "Keras Zoo" models requires a ImageNet weight file on the device
    elif name == "InceptionV3":
        return tf.keras.applications.InceptionV3 , tf.keras.applications.inception_v3.preprocess_input
    else:
        print("Network not found")



tf.keras.backend.set_image_data_format('channels_last')
PATH = "samples"
samples = [512,1024]

networks_to_train = ["ResNet50", "InceptionV3"]

for network_name in networks_to_train:
    SAVE_PATH = network_name
    network, preprocess_input = getNetwork(network_name)

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        print("Directory " , SAVE_PATH ,  " Created")
    else:
        print("Directory " , SAVE_PATH ,  " already exists")

    for sample in samples:
        SAVE_PATH_FOR_SAMPLE = SAVE_PATH + "/{}".format(sample)

        if not os.path.exists(SAVE_PATH_FOR_SAMPLE):
            os.mkdir(SAVE_PATH_FOR_SAMPLE)
            print("Directory " , SAVE_PATH_FOR_SAMPLE ,  " Created")
        else:
            print("Directory " , SAVE_PATH_FOR_SAMPLE,  " already exists")

        #Redirect stdout to file for the training
        with open(SAVE_PATH_FOR_SAMPLE + "/log.log", 'w') as f, redirect_stdout(f):
            #sys.stdout = open(SAVE_PATH_FOR_SAMPLE + "/log.log", 'w')
            train(
                network = network,
                preprocess_input = preprocess_input,
                PATH = PATH + "/{}".format(sample),
                SAVE_PATH = SAVE_PATH_FOR_SAMPLE,
                layers_to_fine_tune = 3
            )


        print("Finished " + SAVE_PATH_FOR_SAMPLE)


