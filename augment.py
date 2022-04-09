# python -m pip install Augmentor
# unzip data.zip
# mv data data_augmented

import Augmentor

def train(path):
    p = Augmentor.Pipeline(path)

    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

    p.sample(100000)


train('data_augmented/train/0_not_cancer')
train('data_augmented/train/1_cancer')


def validate(path):
    p = Augmentor.Pipeline(path)

    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

    p.sample(5000)


validate('data_augmented/validation/0_not_cancer')
validate('data_augmented/validation/1_cancer')

# mv data_augmented/train/0_not_cancer/output 0_not_cancer
# mv data_augmented/train/1_cancer/output 1_cancer
# rm -r data_augmented/train/0_not_cancer
# rm -r data_augmented/train/1_cancer
# mv 0_not_cancer data_augmented/train/0_not_cancer
# mv 1_cancer data_augmented/train/1_cancer

# mv data_augmented/validation/0_not_cancer/output 0_not_cancer
# mv data_augmented/validation/1_cancer/output 1_cancer
# rm -r data_augmented/validation/0_not_cancer
# rm -r data_augmented/validation/1_cancer
# mv 0_not_cancer data_augmented/validation/0_not_cancer
# mv 1_cancer data_augmented/validation/1_cancer
