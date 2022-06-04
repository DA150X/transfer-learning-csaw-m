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
    get_test_results_for_label_network_scale_factor_and_sample_size,
    get_scale_factors,
)


def main():
    parser = get_argument_parser('scale_factor')
    args = parser.parse_args(sys.argv[1:])

    metric = 'auc'
    label = 'If_cancer'

    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)
    networks = get_networks(args.path_to_csv, metric)
    scale_factors = get_scale_factors(args.path_to_csv, metric)

    pylab.rcParams.update({
        'legend.title_fontsize': 'xx-large',
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'font.size': 18,
        'font.family': 'Georgia',
    })

    # https://learnui.design/tools/data-color-picker.html#divergent
    colors = {
        '9523': '#00876c',
        '7000': '#63b179',
        '5000': '#aed987',
        '3000': '#C5C98E',  # custom
        '1000': '#fcc267',
        '500': '#ef8250',
        '100': '#d43d51',
        'fine-tune': '#444',
    }

    xvals = [1, 2, 3]

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    plt.ylabel(make_y_axis_label(metric))
    plt.xlabel('Scale factor')
    plt.xticks([1, 2, 3], ['1x', '2x', '3x'])

    for sample_size in sample_sizes:
        y = []
        x = []
        inc = 0
        for scale_factor in scale_factors:
            for network in networks:
                vals = []
                values = get_test_results_for_label_network_scale_factor_and_sample_size(
                    args.path_to_csv,
                    metric,
                    label,
                    network,
                    scale_factor,
                    sample_size
                )
                after = values['after']
                if after is not None:
                    vals.append(after)

            avg = sum(vals) / len(vals)
            x.append(xvals[inc])
            y.append(avg)
            inc += 1

        plt.plot(
            x,
            y,
            label=sample_size,
            color=colors[sample_size],
            linewidth=3,
            marker='o',
            markersize=18,
        )

    plt.legend(loc='lower right', title='Sample size')
    plt.ylim([min(plt.ylim()) - 0.05 * max(plt.ylim()), max(plt.ylim()) + 0.05 * max(plt.ylim())])
    plt.xlim(0.9, 3.8)
    plt.tight_layout()

    filename = f'if_cancer_{metric}'
    ensure_outputdir_and_write_chart(args.path_to_output + '/scale_factor', plt, filename, dpi=300)


def make_y_axis_label(metric):
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'f_1'
    elif metric == 'auc':
        metric = 'AUC'
    return r' ${{{metric}}}$'.format(metric=metric)


if __name__ == '__main__':
    main()
