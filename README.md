# Transfer Learning on CSAW-M

- [Transfer Learning on CSAW-M](#transfer-learning-on-csaw-m)
  - [Outline](#outline)
    - [Augment](#augment)
    - [Train](#train)

## Outline

### Augment

Before

```
path/to/csaw-m
├── cross_validation
├── images
│   └── preprocessed
│       ├── test
│       └── train
└── labels
    ├── CSAW-M_test.csv
    └── CSAW-M_train.csv
```

`$ augment.py [path_to_data] [sample_size] [scale_factor]`

`$ augment.py path/to/csaw-m 10,100,200,500,1000... 10`

After

```
sample_1
├── description.txt
├── train
│   ├── 0_not_cancer
│   └── 1_cancer
├── validation
│   ├── 0_not_cancer
│   └── 1_cancer
└── test
    ├── 0_not_cancer
    └── 1_cancer

sample_2
├── description.txt
├── train
│   ├── 0_not_cancer
│   └── 1_cancer
├── validation
│   ├── 0_not_cancer
│   └── 1_cancer
└── test
    ├── 0_not_cancer
    └── 1_cancer
```

### Train

```
$ train_resnet50v2.py sample_1 && train_resnet50v2.py sample_2

> sample_1.result
> └── all the results... tables, images etc.
```

- `train_resnet50v2.py`
- `train_resnet151v2.py`
- `common.py`
- etc.

```py
from common import train()

SAVE_PATH = "resnet50v2"

BATCH_SIZE = 64
IMG_SIZE = (224,224)

initial_epochs = 10
fine_tune_epochs = 10
layers_to_fine_tune = 3

preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
network = tf.keras.applications.ResNet50V2

train(
    network,
    initial_epochs=initial_epochs,
    fine_tune_epochs=fine_tune_epochs,
    layers_to_fine_tune=layers_to_fine_tune
)
```
