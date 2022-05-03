import sys
import numpy as np
from beeprint import pp
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.lines import Line2D
from common import (
    get_argument_parser,
    ensure_outputdir_and_write_chart,
    get_test_results_for_label_and_sample_size,
    get_labels,
    get_sample_sizes,
)


def main():
    parser = get_argument_parser('learning_improvements')
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
        '3000': '#C5C98E',  # custom
        '5000': '#fcc267',
        '7000': '#ef8250',
        '9523': '#d43d51',
        'fine-tune': '#444',
    }
    for label in labels:
        plt.figure(figsize=(15, 10))
        fig, ax = plt.subplots(figsize=(14, 9))

        plt.title(make_title(metric, label))
        plt.ylabel(make_y_axis_label(metric, label))
        plt.xlabel('Sample size')

        plt.legend(loc='upper left')
        plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
        plt.tight_layout()
        plt.xticks([0, 1, 2, 3, 4, 5, 6], ['100', '500', '1000', '3000', '5000', '7000', '9523'])

        inc = 0
        for sample_size in sample_sizes:
            values = get_test_results_for_label_and_sample_size(args.path_to_csv, metric, label, sample_size)
            before_values = [values[key]['before'] for key in values]
            after_values = [values[key]['after'] for key in values]

            if len(before_values) == 0:
                continue
            if len(after_values) == 0:
                continue

            before = sum(before_values) / len(before_values)
            after = sum(after_values) / len(after_values)

            x = inc
            y = before
            dx = 0
            dy = after - before
            plt.arrow(
                x,
                y,
                dx,
                dy,
                width=.1,
                head_width=.3,
                head_length=.05,
                length_includes_head=False,
                color=colors[sample_size]
            )
            plt.annotate(
                f'{after - before:+.2}',
                xy=(inc - .3, y - Decimal(.1)),
                fontsize=15,
            )
            inc += 1

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
            ax.legend(custom_lines, ['100', '500', '1000', '3000', '5000', '7000', '9523', 'Start Fine Tuning'], loc='upper left', title='Legend', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()

        filename = f'{label}_{metric}'
        ensure_outputdir_and_write_chart(args.path_to_output + '/learning_improvements', plt, filename, dpi=300)


def make_y_axis_label(metric, label):
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'f_1'
    elif metric == 'auc':
        metric = 'AUC'
    return r' ${{{metric}}}$'.format(metric=metric)


def make_title(metric, label):
    string = 'Test '

    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'F_1  score'
    elif metric == 'auc':
        metric = 'AUC'

    label = label.replace('_', '\_')
    string += r'${{{metric}}}$ before and after training for the $\bf{{{label}}}$ label'.format(metric=metric, label=label)
    return string


if __name__ == '__main__':
    main()
