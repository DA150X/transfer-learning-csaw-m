import os
import tensorflow as tf
from pathlib import Path
from common import train
from focal_loss import BinaryFocalLoss
from contextlib import redirect_stdout
from helpers import get_network, create_path_if_not_exists


tf.keras.backend.set_image_data_format('channels_last')

HOME = str(Path.home())
SAMPLE_PATH = f'{HOME}/da150x/samples/'
SAVE_PATH = f'{HOME}/da150x/results/'
samples = [
    # x1
    '500x1',
    '1000x1',
    '3000x1',
    '5000x1',
    '7000x1',
    '9523x1',

    # x10
    '500x10',
    '1000x10',
    '3000x10',
    '5000x10',
    '7000x10',
    '9523x10',

    # x20
    '500x20',
    '1000x20',
    '3000x20',
    '5000x20',
    '7000x20',
    '9523x20',

    # x30
    '500x30',
    '1000x30',
    '3000x30',
    '5000x30',
    '7000x30',
    '9523x30',
]
batch_sizes = [32]
learning_rates = [0.01]
initial_epochs = 10
fine_tune_epochs = 10
networks_to_train = [
    'ResNet50V2',
    'EfficientNetB0',
    'EfficientNetV2B0',

    # nice to have
    # 'DenseNet169',
    # 'InceptionV3',
    # 'EfficientNetV2S',
    # 'EfficientNetV2B3',
    # 'EfficientNetB7',
    # 'ResNet101V2'
]
loss_functions = [
    {'slug': 'BinCrossEnt', 'func': tf.keras.losses.BinaryCrossentropy(from_logits=True)},
]

for sample in samples:
    for learning_rate in learning_rates:
        for network_name in networks_to_train:
            network, preprocess_input, layers_to_fine_tune = get_network(network_name)

            for loss_function in loss_functions:
                loss_function_func = loss_function['func']
                loss_function_slug = loss_function['slug']

                for batch_size in batch_sizes:
                    config_name = f'#N-{network_name}#S-{sample}-#B-{batch_size}-#LR{learning_rate}#LOSS-{loss_function_slug}'
                    config_params = {
                        'network_name': network_name,
                        'sample': sample,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'loss_function': loss_function_slug,
                    }
                    output_path = f'{SAVE_PATH}/{config_name}/'

                    create_path_if_not_exists(output_path)

                    with open(output_path + '/output.log', 'w') as f, redirect_stdout(f):
                        print(f'running {sample}')
                        train(
                            network=network,
                            preprocess_input=preprocess_input,
                            src_path=f'{SAMPLE_PATH}/{sample}',
                            save_path=output_path,
                            layers_to_fine_tune=layers_to_fine_tune,
                            base_learning_rate=learning_rate,
                            BATCH_SIZE=batch_size,
                            initial_epochs=initial_epochs,
                            fine_tune_epochs=fine_tune_epochs,
                            save_root=SAVE_PATH,
                            config_params=config_params,
                            loss_function=loss_function_func,
                        )

                    print('Finished ' + output_path)
