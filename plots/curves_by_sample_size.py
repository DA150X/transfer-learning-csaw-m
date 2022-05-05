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
    parser = get_argument_parser('curves_by_sample_size')
    args = parser.parse_args(sys.argv[1:])

    metrics = [
        'acc',
        'loss',
        'f1',
        'auc',
    ]
    for metric in metrics:
        create_chart_for_metric(metric, args)

    make_legend(args)


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
        'font.family': 'Georgia',
    })

    # https://learnui.design/tools/data-color-picker.html#divergent
    colors = {
        '100': '#d43d51',
        '500': '#ef8250',
        '1000': '#fcc267',
        '3000': '#C5C98E',  # custom
        '5000': '#aed987',
        '7000': '#63b179',
        '9523': '#00876c',
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
                values, epochs = get_results_for_sample_size_and_label(args.path_to_csv, sample_size, label, metric, validation=validation)
                for name in values:
                    plt.plot(epochs[name], values[name], label=name, color=colors[sample_size], alpha=0.5)

            plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
            plt.plot([10, 10], plt.ylim(), alpha=.5, color=colors['fine-tune'])  # fine-tune line

            if validation:
                filename = f'validation_{label}_{metric}'
            else:
                filename = f'train_{label}_{metric}'
            ensure_outputdir_and_write_chart(args.path_to_output + '/curves_by_sample_size', plt, filename, dpi=300)


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


def make_legend(args):
    pylab.rcParams.update({
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'font.family': 'Georgia',
    })

    # https://learnui.design/tools/data-color-picker.html#divergent
    colors = {
        '100': '#d43d51',
        '500': '#ef8250',
        '1000': '#fcc267',
        '3000': '#C5C98E',  # custom
        '5000': '#aed987',
        '7000': '#63b179',
        '9523': '#00876c',
        'fine-tune': '#444',
    }

    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(10, 10))

    custom_lines = [
        Line2D([0], [0], color=colors['100'], lw=4),
        Line2D([0], [0], color=colors['500'], lw=4),
        Line2D([0], [0], color=colors['1000'], lw=4),
        Line2D([0], [0], color=colors['3000'], lw=4),
        Line2D([0], [0], color=colors['5000'], lw=4),
        Line2D([0], [0], color=colors['7000'], lw=4),
        Line2D([0], [0], color=colors['9523'], lw=4),
        Line2D([0], [0], color=colors['fine-tune'], lw=4, alpha=0.5),
    ]
    legend = ax.legend(custom_lines, ['100', '500', '1000', '3000', '5000', '7000', '9523', 'Start Fine Tuning'], loc='upper left', title='Legend')
    plt.tight_layout()

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = args.path_to_output + '/curves_by_sample_size/legend.png'
    fig.savefig(filename, dpi=300, bbox_inches=bbox, pad_inches=0)


if __name__ == '__main__':
    main()
