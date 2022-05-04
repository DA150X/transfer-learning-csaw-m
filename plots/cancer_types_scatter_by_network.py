import sys
import numpy as np
from beeprint import pp
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from common import (
    get_argument_parser,
    ensure_outputdir_and_write_chart,
    get_sample_sizes,
    get_labels,
    get_networks,
    get_test_results_for_label_and_network,
)


def main():
    parser = get_argument_parser('cancer_types_scatter_by_network')
    args = parser.parse_args(sys.argv[1:])

    metrics = [
        'auc',
        'loss',
    ]
    for metric in metrics:
        create_chart_for_metric(metric, args)


def create_chart_for_metric(metric, args):
    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)
    networks = get_networks(args.path_to_csv, metric)

    pylab.rcParams.update({
        'legend.title_fontsize': 'xx-large',
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'x-large',
        'font.size': 18,
    })

    colors = {
        'DenseNet169': '#003f5c',
        'EfficientNetV2B0': '#bc5090',
        'ResNet50V2': '#ffa600',
    }

    for network in networks:
        plt.figure(figsize=(14, 10))
        fig, ax = plt.subplots(figsize=(14, 10))

        plt.title(make_title(metric, network))
        plt.ylabel(make_y_axis_label(metric))
        plt.xlabel('Cancer Label')

        plt.xticks([0, 1, 2, 3], [
            '${{If\_cancer}}$',
            '${{If\_composite}}$',
            '${{If\_interval\_cancer}}$',
            '${{If\_large\_invasive\_cancer}}$',
        ])

        inc = 0
        for label in labels:
            x = []
            y = []
            z = []
            values = get_test_results_for_label_and_network(args.path_to_csv, metric, label, network)
            for sample_size, vals in values.items():
                value = []
                for scale_factor, metric_value in vals.items():
                    value.append(metric_value)

                avg = float(sum(value) / len(value))
                x.append(inc)
                y.append(avg)
                z.append(200 + int(sample_size) / 4)

            sc = plt.scatter(
                x,
                y,
                s=z,
                alpha=0.3,
                label=network,
                color=colors[network],
            )
            inc += 1


        plt.ylim([min(plt.ylim()) - 0.2 * max(plt.ylim()), max(plt.ylim()) + 0.2 * max(plt.ylim())])
        plt.xlim([min(plt.xlim()) - 0.2 * max(plt.xlim()), max(plt.xlim()) + 0.2 * max(plt.xlim())])

        filename = f'{network}_{metric}'
        ensure_outputdir_and_write_chart(args.path_to_output + '/cancer_types_scatter', plt, filename, dpi=300)


def make_y_axis_label(metric):
    if metric == 'auc':
        metric = 'AUC'
    elif metric == 'loss':
        metric = 'Loss'
    return r' ${{{metric}}}$'.format(metric=metric)


def make_title(metric, network):
    string = ''
    if metric == 'auc':
        metric = 'AUC'
    elif metric == 'loss':
        metric = 'Loss'

    string += r' $\bf{{{metric}}}$ for each cancer label with $\bf{{{network}}}$'.format(metric=metric,network=network)
    return string


if __name__ == '__main__':
    main()
