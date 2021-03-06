import csv
import sys
import shutil
import random
import pathlib
import argparse
import Augmentor


class Image:
    def __init__(self, label, src_path, filename):
        self.label = label
        self.filename = filename
        self.src_path = src_path

    def write_to_dir(self, dir):
        if not pathlib.Path(f'{dir}/{self.label}/').exists():
            pathlib.Path(f'{dir}/{self.label}/').mkdir(parents=True)

        target = f'{dir}/{self.label}/{self.filename}'
        shutil.copyfile(self.src_path, target)


class Dataset:
    def __init__(self, path, label_src):
        self.train_images = []
        self.test_images = []
        self.label_src = label_src

        labels_path = f'{path}/labels/CSAW-M_train.csv'
        with open(labels_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            count = 0
            for row in reader:
                filename = row['Filename']
                cancer = bool(int(row[label_src]))

                label = '0_not_cancer'
                if cancer:
                    label = '1_cancer'

                img = Image(label, f'{path}/images/preprocessed/train/{filename}', filename)
                self.train_images.append(img)

        labels_path = f'{path}/labels/CSAW-M_test.csv'
        with open(labels_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            count = 0
            for row in reader:
                filename = row['Filename']
                cancer = bool(int(row[label_src]))

                label = '0_not_cancer'
                if cancer:
                    label = '1_cancer'

                img = Image(label, f'{path}/images/preprocessed/test/{filename}', filename)
                self.test_images.append(img)


def get_argument_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        prog='augment',
        description='Augment the CSAW-dataset',
        formatter_class=argparse.HelpFormatter,
    )

    parser.add_argument(
        'path_to_csaw_m',
        type=str,
        help='the path to the CSAW-M dataset',
    )

    parser.add_argument(
        'output',
        type=str,
        help='output directory for the samples',
    )

    parser.add_argument(
        'sample_sizes',
        type=str,
        help='comma separated list of sample sizes to pick',
    )

    parser.add_argument(
        'scale_factor',
        type=int,
        help='factor to scale the samples by',
    )

    parser.add_argument(
        'label',
        type=str,
        choices=['If_cancer', 'If_interval_cancer', 'If_large_invasive_cancer', 'If_composite'],
        help='label to select cancer/not cancer from',
    )

    parser.add_argument(
        '--ignore-output-exists',
        action='store_true',
        default=False,
        help='Use this flag to ignore the overwrite/abort prompt if output directory already exists',
    )

    return parser


def fail(msg):
    print(msg)
    sys.exit(1)


def parse_input(args):
    path_to_csaw_m = args.path_to_csaw_m
    output = args.output
    sample_sizes = args.sample_sizes.split(',')
    scale_factor = args.scale_factor
    label = args.label

    for i, sample in enumerate(sample_sizes):
        try:
            sample_sizes[i] = int(sample)
        except ValueError:
            fail(f'bad sample size: "{sample}"')

    return path_to_csaw_m, output, sample_sizes, scale_factor, label


def write_images_to_temporary_directory(images, output_dir):
    tmpdir = output_dir + '/tmp'

    if pathlib.Path(tmpdir).exists():
        shutil.rmtree(tmpdir)

    pathlib.Path(tmpdir).mkdir()

    for image in images:
        image.write_to_dir(tmpdir)


def dataset_valid(dataset):
    num_cancer = 0
    num_not_cancer = 0
    for image in dataset:
        if image.label == '1_cancer':
            num_cancer += 1
        else:
            num_not_cancer += 1

    return num_cancer != 0 and num_not_cancer != 0


def generate_samples(dataset, size, scale_factor, output, label):
    print(f'scaling {size} by {scale_factor} with label {label}')
    found_valid = False

    train_subset = None
    validation_subset = None
    test_subset = None

    while not found_valid:
        train_subset = random.sample(dataset.train_images, size)
        test_subset = dataset.test_images

        # remove subset
        index = int((len(train_subset) * (20/100)))
        validation_subset = train_subset[:index]
        train_subset = train_subset[index:]

        if dataset_valid(validation_subset) and dataset_valid(train_subset):
            found_valid = True
        else:
            print('invalid datasets, generating new...')


    print('train_subset', len(train_subset))
    print('validation_subset', len(validation_subset))
    print('test_subset', len(test_subset))

    augment_dataset(train_subset, scale_factor, output, f'{size}x{scale_factor}-{label}/train')
    augment_dataset(validation_subset, scale_factor, output, f'{size}x{scale_factor}-{label}/validation')
    for image in test_subset:
        image.write_to_dir(f'{output}/{size}x{scale_factor}-{label}/test')


def augment_dataset(dataset, scale_factor, output_dir, output_name):
    write_images_to_temporary_directory(dataset, output_dir)
    num_cancer = len(list(pathlib.Path(output_dir + '/tmp/1_cancer').glob('*')))
    num_not_cancer = len(list(pathlib.Path(output_dir + '/tmp/0_not_cancer').glob('*')))

    # ensure the dataset becomes balanced
    target_number = num_not_cancer * scale_factor
    # num_cancer_target = num_cancer * scale_factor
    # num_not_cancer_target = num_not_cancer * scale_factor

    p = Augmentor.Pipeline(output_dir + '/tmp/1_cancer')
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.random_contrast(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.sample(target_number)

    p = Augmentor.Pipeline(output_dir + '/tmp/0_not_cancer')
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.random_contrast(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.sample(target_number)

    shutil.move(output_dir + '/tmp/1_cancer/output', f'{output_dir}/{output_name}/1_cancer')
    shutil.move(output_dir + '/tmp/0_not_cancer/output', f'{output_dir}/{output_name}/0_not_cancer')
    shutil.rmtree(output_dir + '/tmp')


def main():
    parser = get_argument_parser()
    args = parser.parse_args(sys.argv[1:])

    path_to_csaw_m, output, sample_sizes, scale_factor, label = parse_input(args)

    print('Input parameters')
    print(f'path to csaw-m:   {path_to_csaw_m}')
    print(f'output directory: {output}')
    print(f'sample sizes:     {sample_sizes}')
    print(f'scale factor:     {scale_factor}')
    print(f'label:            {label}')

    dataset = Dataset(path_to_csaw_m, label)

    while pathlib.Path(output).exists() and not args.ignore_output_exists:
        print(f'Output dir {output} already exists. Would you like to delete, or abort')
        answer = input('Type one of [O]verwrite, or [A]bort: ')
        if answer.upper() == 'O':
            shutil.rmtree(output)
        elif answer.upper() == 'A':
            exit(0)

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    for size in sample_sizes:
        generate_samples(dataset, size, scale_factor, output, label)


if __name__ == '__main__':
    main()
