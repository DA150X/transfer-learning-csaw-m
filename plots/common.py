import sys
import shutil
import argparse
from beeprint import pp
from pathlib import Path
from decimal import Decimal


def get_argument_parser(name):
    parser = argparse.ArgumentParser(
        prog=name,
        formatter_class=argparse.HelpFormatter,
    )

    parser.add_argument(
        'path_to_csv',
        type=str,
        help='the path to the csv files',
    )

    parser.add_argument(
        'path_to_output',
        type=str,
        help='the path to write output files to',
    )

    return parser


def get_sample_sizes(path_to_results, metric):
    filename_metric = f'{path_to_results}/{metric}'
    results = read_csv_results(filename_metric, validation=True)

    sample_sizes = []
    for row in results:
        sample_size = row['sample'].split('x')[0]
        if sample_size not in sample_sizes:
            sample_sizes.append(sample_size)

    sample_sizes.sort(key=lambda x: int(x))
    return sample_sizes


def get_labels(path_to_results, metric):
    filename_metric = f'{path_to_results}/{metric}'
    results = read_csv_results(filename_metric, validation=True)

    labels = []
    for row in results:
        label = row['label']
        if label not in labels:
            labels.append(label)

    return labels


def get_results_for_sample_size_and_label(path_to_results, sample_size, label, metric, validation):
    filename_metric = f'{path_to_results}/{metric}'
    results = read_csv_results(filename_metric, validation=validation)

    values = {}
    epochs = {}
    for row in results:
        if row['sample_size'] != sample_size:
            continue
        if row['label'] != label:
            continue
        name = row['config_name']
        if name not in values:
            values[name] = []
        if name not in epochs:
            epochs[name] = []

        values[name].append(row['value'])
        epochs[name].append(row['epoch'])

    for name in values:
        values[name] = values[name][:-1]
        epochs[name] = epochs[name][:-1]

    return values, epochs



def get_test_results(path_to_results, metric):
    filename_metric = f'{path_to_results}/{metric}'
    results = read_csv_results(filename_metric, validation=True)

    before_scores = get_before_scores(path_to_results, metric)

    before_and_after = {}
    for row in results:
        networkname = row['network_name']
        sample = row['sample']
        selector = f'{networkname} {sample}'

        # establish data type, and last will always be after training and fine-tuning
        before_and_after[selector] = {'before': before_scores[row['config_name']], 'after': row['value']}

    return before_and_after


def get_before_scores(path_to_results, metric):
    before_scores = {}
    for config in get_all_configs(path_to_results, metric):
        logfile = f'{path_to_results}/{config}/output.log'
        with open(logfile, 'r') as log:
            lines = log.readlines()
            for line in lines:
                if f'initial {metric}' in line:
                    value = Decimal(line.split(':')[1])
                    before_scores[config] = value
    return before_scores


def get_all_configs(path_to_results, metric):
    filename_metric = f'{path_to_results}/{metric}'
    results = read_csv_results(filename_metric, validation=False)
    configs = []
    for row in results:
        if row['config_name'] not in configs:
            configs.append(row['config_name'])
    return configs


def read_csv_results(path_to_csv, validation):
    if validation:
        path_to_csv += '_val'
    path_to_csv += '.csv'

    rows = []
    with open(path_to_csv, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            cols = line.split(',')
            network_name = cols[0]
            sample = cols[1]
            batch_size = int(cols[2])
            learning_rate = Decimal(cols[3])
            loss_function = cols[4]
            epoch = int(cols[5])
            value = Decimal(cols[6])
            sample_size = sample.split('x')[0]
            scale_factor_and_label = sample.split('x')[1]

            if scale_factor_and_label.isdigit():
                label = 'If_cancer'
            else:
                label = scale_factor_and_label.split('-')[1]

            rows.append({
                'config_name': f'#N-{network_name}#S-{sample}-#B-{batch_size}-#LR{learning_rate}#LOSS-{loss_function}',
                'network_name': network_name,
                'sample': sample,
                'epoch': epoch,
                'value': value,
                'sample_size': sample_size,
                'label': label,
            })
    return rows


def ensure_outputdir_and_write_chart(output_dir, plt, filename, dpi=100):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    plt.savefig(f'{output_dir}/{filename}', dpi=dpi)
    plt.close()
