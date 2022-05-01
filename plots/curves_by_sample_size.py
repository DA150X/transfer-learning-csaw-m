import sys
import numpy as np
from beeprint import pp
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
    get_results_for_sample_size_and_label,
)


def main():
    parser = get_argument_parser('validation_curves_by_sample_size')
    args = parser.parse_args(sys.argv[1:])

    metrics = [
        'acc',
        'loss',
        'f1',
        'auc',
    ]
    for metric in metrics:
        create_chart_for_metric(metric, args)


def create_chart_for_metric(metric, args):
    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)
    validation_options = [True, False]

    pylab.rcParams.update({
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
    })

    # https://learnui.design/tools/data-color-picker.html#divergent
    colors = {
        '100': '#00876c',
        '500': '#63b179',
        '1000': '#aed987',
        '3000': '#ffff9d',
        '5000': '#fcc267',
        '7000': '#ef8250',
        '9523': '#d43d51',
        'fine-tune': '#444',
    }

    for validation in validation_options:
        for label in labels:
            plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots(figsize=(10, 10))

            plt.title(make_title(metric, label, validation))
            plt.ylabel(make_y_axis_label(metric, label, validation))
            plt.xlabel('Epoch')
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))

            for sample_size in sample_sizes:
                values, epochs = get_results_for_sample_size_and_label(args.path_to_csv, sample_size, label, metric, validation=True)
                for name in values:
                    plt.plot(epochs[name], values[name], label=name, color=colors[sample_size], alpha=0.5)

            plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
            plt.plot([9, 9], plt.ylim(), alpha=.5, color=colors['fine-tune'])  # fine-tune line

            custom_lines = [
                Line2D([0], [0], color=colors['100'], lw=4),
                Line2D([0], [0], color=colors['500'], lw=4),
                Line2D([0], [0], color=colors['1000'], lw=4),
                Line2D([0], [0], color=colors['3000'], lw=4),
                Line2D([0], [0], color=colors['5000'], lw=4),
                Line2D([0], [0], color=colors['7000'], lw=4),
                Line2D([0], [0], color=colors['9523'], lw=4),
                Line2D([0], [0], color=colors['fine-tune'], lw=4),
            ]
            ax.legend(custom_lines, ['100', '500', '1000', '3000', '5000', '7000', '9523', 'Start Fine Tuning'], loc='upper left', title='Legend', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()

            if validation:
                filename = f'validation_{label}_{metric}'
            else:
                filename = f'train_{label}_{metric}'
            ensure_outputdir_and_write_chart(args.path_to_output + '/curves_by_sample_size', plt, filename)


def make_y_axis_label(metric, label, validation):
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'f_1'
    elif metric == 'auc':
        metric = 'AUC'
    return r' ${{{metric}}}$'.format(metric=metric)


def make_title(metric, label, validation):
    string = ''
    if validation:
        string += 'Validation'
    else:
        string += 'Training'

    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'F_1'
    elif metric == 'auc':
        metric = 'AUC'

    label = label.replace('_', '\_')
    string += r' ${{{metric}}}$ for the $\bf{{{label}}}$ label'.format(metric=metric, label=label)
    return string


if __name__ == '__main__':
    main()