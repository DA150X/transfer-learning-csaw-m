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
    def __init__(self, path):
        self.train_images = []
        self.test_images = []

        labels_path = f'{path}/labels/CSAW-M_train.csv'
        with open(labels_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            count = 0
            for row in reader:
                filename = row['Filename']
                cancer = bool(int(row['If_cancer']))

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
                cancer = bool(int(row['If_cancer']))

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

    for i, sample in enumerate(sample_sizes):
        try:
            sample_sizes[i] = int(sample)
        except ValueError:
            fail(f'bad sample size: "{sample}"')

    return path_to_csaw_m, output, sample_sizes, scale_factor


def write_images_to_temporary_directory(images, output_dir):
    tmpdir = output_dir + '/tmp'

    if pathlib.Path(tmpdir).exists():
        shutil.rmtree(tmpdir)

    pathlib.Path(tmpdir).mkdir()

    for image in images:
        image.write_to_dir(tmpdir)


def generate_samples(dataset, size, scale_factor, output):
    print(f'scaling {size} by {scale_factor}')
    train_subset = random.sample(dataset.train_images, size)
    test_subset = dataset.test_images

    # remove subset
    index = int((len(train_subset) * (20/100)))
    validation_subset = train_subset[:index]
    train_subset = train_subset[index:]

    print('train_subset', len(train_subset))
    print('validation_subset', len(validation_subset))
    print('test_subset', len(test_subset))

    augment_dataset(train_subset, scale_factor, output, f'{size}x{scale_factor}/train')
    augment_dataset(validation_subset, scale_factor, output, f'{size}x{scale_factor}/validation')
    for image in test_subset:
        image.write_to_dir(f'{output}/{size}x{scale_factor}/test')


def augment_dataset(dataset, scale_factor, output_dir, output_name):
    write_images_to_temporary_directory(dataset, output_dir)
    num_cancer = len(list(pathlib.Path(output_dir + '/tmp/1_cancer').glob('*')))
    num_not_cancer = len(list(pathlib.Path(output_dir + '/tmp/0_not_cancer').glob('*')))
    num_cancer_target = num_cancer * scale_factor
    num_not_cancer_target = num_not_cancer * scale_factor

    p = Augmentor.Pipeline(output_dir + '/tmp/1_cancer')
    p.rotate(probability=0.7, max_left_rotation=17, max_right_rotation=17)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.random_contrast(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.sample(num_cancer_target)

    p = Augmentor.Pipeline(output_dir + '/tmp/0_not_cancer')
    p.rotate(probability=0.7, max_left_rotation=17, max_right_rotation=17)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.random_contrast(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.sample(num_not_cancer_target)

    shutil.move(output_dir + '/tmp/1_cancer/output', f'{output_dir}/{output_name}/1_cancer')
    shutil.move(output_dir + '/tmp/0_not_cancer/output', f'{output_dir}/{output_name}/0_not_cancer')
    shutil.rmtree(output_dir + '/tmp')


def main():
    parser = get_argument_parser()
    args = parser.parse_args(sys.argv[1:])

    path_to_csaw_m, output, sample_sizes, scale_factor = parse_input(args)

    print('Input parameters')
    print(f'path to csaw-m:   {path_to_csaw_m}')
    print(f'output directory: {output}')
    print(f'sample sizes:     {sample_sizes}')
    print(f'scale factor:     {scale_factor}')

    dataset = Dataset(path_to_csaw_m)

    while pathlib.Path(output).exists() and not args.ignore_output_exists:
        print(f'Output dir {output} already exists. Would you like to delete, or abort')
        answer = input('Type one of [O]verwrite, or [A]bort: ')
        if answer.upper() == 'O':
            shutil.rmtree(output)
        elif answer.upper() == 'A':
            exit(0)

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    for size in sample_sizes:
        generate_samples(dataset, size, scale_factor, output)


if __name__ == '__main__':
    main()
