import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from classification_models.keras import Classifiers
from common2 import train
import sys
from contextlib import redirect_stdout

tf.keras.backend.set_image_data_format('channels_last')

#ResNet18, preprocess_input = Classifiers.get('resnet18') #For keras zoo models

network = tf.keras.applications.ResNet50V2
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input #Preprocess input into correct format (i.e. [0,255] -> [-1,1])

PATH = "samples"
SAVE_PATH = "testtt"
samples = [512,1024,2048,4096,8192]

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

