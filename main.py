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
# samples = ['3000x10', '6000x10', '9000x10', '3000x100', '6000x100', '9000x100']
samples = ['9000x2']
batch_sizes = [32]
learning_rates = [0.01, 0.001]
initial_epochs = 7
fine_tune_epochs = 7
networks_to_train = ['ResNet50V2', 'EfficientNetB0', 'DenseNet169', 'InceptionV3']
loss_functions = [
    {'slug': 'FocalLoss', 'func': BinaryFocalLoss(gamma=2)},
    {'slug': 'Hinge', 'func': tf.keras.losses.Hinge(reduction=tf.keras.losses.Reduction.AUTO, name='hinge')},
    {'slug': 'Huber', 'func': tf.keras.losses.Huber(delta=1.0, reduction='auto', name='huber_loss')},
    {'slug': 'BinCrossEnt', 'func': tf.keras.losses.BinaryCrossentropy(from_logits=True)},
    {'slug': 'KLDivergence', 'func': tf.keras.losses.KLDivergence(reduction='auto', name='kl_divergence')},
]


for sample in samples:
    for network_name in networks_to_train:
        network, preprocess_input, layers_to_fine_tune = get_network(network_name)

        for loss_function in loss_functions:
            loss_function_func = loss_function['func']
            loss_function_slug = loss_function['slug']

            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
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
