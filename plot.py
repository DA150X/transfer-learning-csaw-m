import sys
import argparse
import beeprint
from pathlib import Path
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def get_argument_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        prog='plot',
        description='Plot data',
        formatter_class=argparse.HelpFormatter,
    )

    parser.add_argument(
        'path_to_csv',
        type=str,
        help='the path to the csv files',
    )

    return parser


def main():
    parser = get_argument_parser()
    args = parser.parse_args(sys.argv[1:])

    group_by_network(args)
    # group_by_scale_factor(args)
    # group_by_sample_size(args)
    # run_all(args)


def run_all(args):
    metrics = [
        'acc',
        'acc_val',
        'loss',
        'loss_val',
        'auc',
        'auc_val',
        'f1',
        'f1_val',
    ]

    for metric in metrics:
        filename_csv = f'{args.path_to_csv}/{metric}.csv'
        output_path = f'{args.path_to_csv}/{metric}.png'

        plots = []
        plt.figure(figsize=(8, 8))
        plt.legend(loc='upper left')
        plt.ylabel(metric)
        plt.title(f'a chart with {metric} performance for various configurations')
        plt.xlabel('epoch')

        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            values = {}
            epochs = {}
            last_value = []

            for line in lines[1:]:
                cols = line.split(',')
                network_name = cols[0]
                sample = cols[1]
                batch_size = int(cols[2])
                learning_rate = Decimal(cols[3])
                loss_function = cols[4]
                epoch = int(cols[5])
                value = Decimal(cols[6])

                main_label = f'#N-{network_name}#S-{sample}-#B-{batch_size}-#LR{learning_rate}#LOSS-{loss_function}'
                if main_label not in values:
                    values[main_label] = []
                if main_label not in epochs:
                    epochs[main_label] = []

                values[main_label].append(value)
                epochs[main_label].append(epoch)

            for label in values:
                last_value.append({'label': label, 'value': values[label][len(values[label]) - 1]})

            last_value.sort(key=lambda x: x['value'], reverse=True)
            for label in last_value:
                label = label['label']
                p, = plt.plot(epochs[label], values[label], label=label)
                plots.append(p)

        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(handles=plots, title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
        plt.tight_layout()
        plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
        plt.savefig(output_path)
        plt.close()


def group_by_sample_size(args):
    metrics = [
        'acc',
        'acc_val',
        'loss',
        'loss_val',
        'auc',
        'auc_val',
        'f1',
        'f1_val',
    ]

    for metric in metrics:
        filename_csv = f'{args.path_to_csv}/{metric}.csv'
        output_path = f'{args.path_to_csv}/group_by_sample_size/{metric}.png'

        samples = []
        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                cols = line.split(',')
                sample = cols[1].split('x')[0]
                if sample not in samples:
                    samples.append(sample)


        plots = []
        plt.figure(figsize=(8, 8))
        plt.legend(loc='upper left')
        plt.ylabel(metric)
        plt.title(f'a chart with {metric} performance for various configurations')
        plt.xlabel('epoch')

        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            values = {}
            epochs = {}
            last_value = []

            for line in lines[1:]:
                cols = line.split(',')
                network_name = cols[0]
                sample = cols[1]
                sample_group = cols[1].split('x')[0]
                scale_factor = cols[1].split('x')[1]
                # if scale_factor != '1':
                #     continue
                batch_size = int(cols[2])
                learning_rate = Decimal(cols[3])
                loss_function = cols[4]
                epoch = int(cols[5])
                value = Decimal(cols[6])

                main_label = f'{sample_group}'
                if main_label not in values:
                    values[main_label] = {}
                if main_label not in epochs:
                    epochs[main_label] = {}

                if epoch not in values[main_label]:
                    values[main_label][epoch] = []
                if epoch not in epochs[main_label]:
                    epochs[main_label][epoch] = []

                values[main_label][epoch].append(value)
                epochs[main_label][epoch].append(epoch)

            for label in values:
                new = []
                for epoch in values[label]:
                    new.append(Decimal(sum(values[label][epoch]) / len(values[label][epoch])))
                values[label] = new

            for label in epochs:
                new = []
                for epoch in epochs[label]:
                    new.append(epochs[label][epoch][0])
                epochs[label] = new

            beeprint.pp(values)
            beeprint.pp(epochs)

            for label in values:
                last_value.append({'label': label, 'value': values[label][len(values[label]) - 2]})

            last_value.sort(key=lambda x: x['value'], reverse=True)
            for label in last_value:
                label = label['label']
                p, = plt.plot(epochs[label], values[label], label=label)
                plots.append(p)

            fontP = FontProperties()
            fontP.set_size('xx-small')
            plt.legend(handles=plots, title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
            plt.tight_layout()
            plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
            Path(output_path).parent.mkdir(exist_ok=True)
            plt.savefig(output_path)
            plt.close()


def group_by_scale_factor(args):
    metrics = [
        'acc',
        'acc_val',
        'loss',
        'loss_val',
        'auc',
        'auc_val',
        'f1',
        'f1_val',
    ]

    for metric in metrics:
        filename_csv = f'{args.path_to_csv}/{metric}.csv'
        output_path = f'{args.path_to_csv}/group_by_scale_factor/{metric}.png'

        samples = []
        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                cols = line.split(',')
                sample = cols[1].split('x')[0]
                if sample not in samples:
                    samples.append(sample)


        plots = []
        plt.figure(figsize=(8, 8))
        plt.legend(loc='upper left')
        plt.ylabel(metric)
        plt.title(f'a chart with {metric} performance for various configurations')
        plt.xlabel('epoch')

        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            values = {}
            epochs = {}
            last_value = []

            for line in lines[1:]:
                cols = line.split(',')
                network_name = cols[0]
                sample = cols[1]
                sample_group = cols[1].split('x')[0]
                scale_factor = cols[1].split('x')[1]
                # if scale_factor != '1':
                #     continue
                batch_size = int(cols[2])
                learning_rate = Decimal(cols[3])
                loss_function = cols[4]
                epoch = int(cols[5])
                value = Decimal(cols[6])

                main_label = f'{scale_factor}'
                if main_label not in values:
                    values[main_label] = {}
                if main_label not in epochs:
                    epochs[main_label] = {}

                if epoch not in values[main_label]:
                    values[main_label][epoch] = []
                if epoch not in epochs[main_label]:
                    epochs[main_label][epoch] = []

                values[main_label][epoch].append(value)
                epochs[main_label][epoch].append(epoch)

            for label in values:
                new = []
                for epoch in values[label]:
                    new.append(Decimal(sum(values[label][epoch]) / len(values[label][epoch])))
                values[label] = new

            for label in epochs:
                new = []
                for epoch in epochs[label]:
                    new.append(epochs[label][epoch][0])
                epochs[label] = new

            beeprint.pp(values)
            beeprint.pp(epochs)

            for label in values:
                last_value.append({'label': label, 'value': values[label][len(values[label]) - 2]})

            last_value.sort(key=lambda x: x['value'], reverse=True)
            for label in last_value:
                label = label['label']
                p, = plt.plot(epochs[label], values[label], label=label)
                plots.append(p)

            fontP = FontProperties()
            fontP.set_size('xx-small')
            plt.legend(handles=plots, title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
            plt.tight_layout()
            plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
            Path(output_path).parent.mkdir(exist_ok=True)
            plt.savefig(output_path)
            plt.close()


def group_by_network(args):
    metrics = [
        'acc',
        'acc_val',
        'loss',
        'loss_val',
        'auc',
        'auc_val',
        'f1',
        'f1_val',
    ]

    for metric in metrics:
        filename_csv = f'{args.path_to_csv}/{metric}.csv'
        output_path = f'{args.path_to_csv}/group_by_network/{metric}.png'

        samples = []
        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                cols = line.split(',')
                sample = cols[1].split('x')[0]
                if sample not in samples:
                    samples.append(sample)


        plots = []
        plt.figure(figsize=(8, 8))
        plt.legend(loc='upper left')
        plt.ylabel(metric)
        plt.title(f'a chart with {metric} performance for various configurations')
        plt.xlabel('epoch')

        with open(filename_csv, 'r') as file:
            lines = file.readlines()
            values = {}
            epochs = {}
            last_value = []

            for line in lines[1:]:
                cols = line.split(',')
                network_name = cols[0]
                sample = cols[1]
                sample_group = cols[1].split('x')[0]
                scale_factor = cols[1].split('x')[1]
                # if scale_factor != '1':
                #     continue
                batch_size = int(cols[2])
                learning_rate = Decimal(cols[3])
                loss_function = cols[4]
                epoch = int(cols[5])
                value = Decimal(cols[6])

                main_label = f'{network_name}'
                if main_label not in values:
                    values[main_label] = {}
                if main_label not in epochs:
                    epochs[main_label] = {}

                if epoch not in values[main_label]:
                    values[main_label][epoch] = []
                if epoch not in epochs[main_label]:
                    epochs[main_label][epoch] = []

                values[main_label][epoch].append(value)
                epochs[main_label][epoch].append(epoch)

            for label in values:
                new = []
                for epoch in values[label]:
                    new.append(Decimal(sum(values[label][epoch]) / len(values[label][epoch])))
                values[label] = new

            for label in epochs:
                new = []
                for epoch in epochs[label]:
                    new.append(epochs[label][epoch][0])
                epochs[label] = new

            beeprint.pp(values)
            beeprint.pp(epochs)

            for label in values:
                last_value.append({'label': label, 'value': values[label][len(values[label]) - 2]})

            last_value.sort(key=lambda x: x['value'], reverse=True)
            for label in last_value:
                label = label['label']
                p, = plt.plot(epochs[label], values[label], label=label)
                plots.append(p)

            fontP = FontProperties()
            fontP.set_size('xx-small')
            plt.legend(handles=plots, title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
            plt.tight_layout()
            plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
            Path(output_path).parent.mkdir(exist_ok=True)
            plt.savefig(output_path)
            plt.close()


if __name__ == '__main__':
    main()
