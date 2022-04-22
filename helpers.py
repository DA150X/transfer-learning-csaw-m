import os
import tensorflow as tf


# Returns network, preprocess_input and layers to finetune for the given network
def get_network(name):
    if name == 'ResNet50V2':
        return tf.keras.applications.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input, 3
    elif name == 'ResNet50':
        return tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input, 2
    elif name == 'ResNet152V2':
        return tf.keras.applications.resnet_v2.ResNet152V2, tf.keras.applications.resnet_v2.preprocess_input, 3
    elif name == 'ResNet101':
        return tf.keras.applications.ResNet101, tf.keras.applications.resnet.preprocess_input, 2
    elif name == 'EfficientNetB0':
        return tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB1':
        return tf.keras.applications.EfficientNetB1, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB2':
        return tf.keras.applications.EfficientNetB2, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB3':
        return tf.keras.applications.EfficientNetB3, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB4':
        return tf.keras.applications.EfficientNetB4, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB5':
        return tf.keras.applications.EfficientNetB5, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB6':
        return tf.keras.applications.EfficientNetB6, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetB7':
        return tf.keras.applications.EfficientNetB7, tf.keras.applications.efficientnet.preprocess_input, 4
    elif name == 'EfficientNetV2B0':
        return tf.keras.applications.EfficientNetV2B0, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'EfficientNetV2B1':
        return tf.keras.applications.EfficientNetV2B1, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'EfficientNetV2B2':
        return tf.keras.applications.EfficientNetV2B2, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'EfficientNetV2B3':
        return tf.keras.applications.EfficientNetV2B3, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'EfficientNetV2S':
        return tf.keras.applications.EfficientNetV2S, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'EfficientNetV2M':
        return tf.keras.applications.EfficientNetV2M, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'EfficientNetV2L':
        return tf.keras.applications.EfficientNetV2L, tf.keras.applications.efficientnet_v2.preprocess_input, 4
    elif name == 'VGG16':
        return tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input, 1
    elif name == 'DenseNet169':
        return tf.keras.applications.DenseNet169, tf.keras.applications.densenet.preprocess_input, 3
    # elif name == 'ResNet34':
    #     return Classifiers.get('resnet34'), 3 # 'Keras Zoo' models requires a ImageNet weight file located at /weights
    elif name == 'InceptionV3':
        return tf.keras.applications.InceptionV3, tf.keras.applications.inception_v3.preprocess_input, 3
    elif name == 'InceptionResNetV2':
        return tf.keras.applications.InceptionResNetV2, tf.keras.applications.inception_resnet_v2.preprocess_input, 4
    else:
        print('Network not found')


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('Directory ', path, ' Created')
    else:
        print('Directory ', path, ' already exists')
