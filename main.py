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
    # 100 images
    '100x1-If_cancer',
    '100x1-If_interval_cancer',
    '100x1-If_large_invasive_cancer',
    '100x1-If_composite',
    '100x2-If_cancer',
    '100x2-If_interval_cancer',
    '100x2-If_large_invasive_cancer',
    '100x2-If_composite',
    '100x3-If_cancer',
    '100x3-If_interval_cancer',
    '100x3-If_large_invasive_cancer',
    '100x3-If_composite',

    # x3
    '500x3',
    '1000x3',
    '3000x3',
    '5000x3',
    '7000x3',
    '9523x3',

    # x1
    '500x1-If_interval_cancer',
    '500x1-If_large_invasive_cancer',
    '500x1-If_composite',
    '1000x1-If_interval_cancer',
    '1000x1-If_large_invasive_cancer',
    '1000x1-If_composite',
    '3000x1-If_interval_cancer',
    '3000x1-If_large_invasive_cancer',
    '3000x1-If_composite',
    '5000x1-If_interval_cancer',
    '5000x1-If_large_invasive_cancer',
    '5000x1-If_composite',
    '7000x1-If_interval_cancer',
    '7000x1-If_large_invasive_cancer',
    '7000x1-If_composite',
    '9523x1-If_interval_cancer',
    '9523x1-If_large_invasive_cancer',
    '9523x1-If_composite',

    # x2
    '500x2-If_interval_cancer',
    '500x2-If_large_invasive_cancer',
    '500x2-If_composite',
    '1000x2-If_interval_cancer',
    '1000x2-If_large_invasive_cancer',
    '1000x2-If_composite',
    '3000x2-If_interval_cancer',
    '3000x2-If_large_invasive_cancer',
    '3000x2-If_composite',
    '5000x2-If_interval_cancer',
    '5000x2-If_large_invasive_cancer',
    '5000x2-If_composite',
    '7000x2-If_interval_cancer',
    '7000x2-If_large_invasive_cancer',
    '7000x2-If_composite',
    '9523x2-If_interval_cancer',
    '9523x2-If_large_invasive_cancer',
    '9523x2-If_composite',

    # x3
    '500x3-If_interval_cancer',
    '500x3-If_large_invasive_cancer',
    '500x3-If_composite',
    '1000x3-If_interval_cancer',
    '1000x3-If_large_invasive_cancer',
    '1000x3-If_composite',
    '3000x3-If_interval_cancer',
    '3000x3-If_large_invasive_cancer',
    '3000x3-If_composite',
    '5000x3-If_interval_cancer',
    '5000x3-If_large_invasive_cancer',
    '5000x3-If_composite',
    '7000x3-If_interval_cancer',
    '7000x3-If_large_invasive_cancer',
    '7000x3-If_composite',
    '9523x3-If_interval_cancer',
    '9523x3-If_large_invasive_cancer',
    '9523x3-If_composite',

    # x1
    '500x1-If_cancer',
    '1000x1-If_cancer',
    '3000x1-If_cancer',
    '5000x1-If_cancer',
    '7000x1-If_cancer',
    '9523x1-If_cancer',

    # x2
    '500x2-If_cancer',
    '1000x2-If_cancer',
    '3000x2-If_cancer',
    '5000x2-If_cancer',
    '7000x2-If_cancer',
    '9523x2-If_cancer',

    # x3
    '500x3-If_cancer',
    '1000x3-If_cancer',
    '3000x3-If_cancer',
    '5000x3-If_cancer',
    '7000x3-If_cancer',
    '9523x3-If_cancer',
]
batch_sizes = [32]
learning_rates = [0.01]
initial_epochs = 10
fine_tune_epochs = 10
networks_to_train = [
    'ResNet50V2',
    'EfficientNetV2B0',
    'DenseNet169',

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
            network, preprocess_input, layers_to_fine_tune, fine_tune_learning_rate_multiplier = get_network(network_name)

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
                            fine_tune_learning_rate_multiplier=fine_tune_learning_rate_multiplier,
                        )

                    print('Finished ' + output_path)
