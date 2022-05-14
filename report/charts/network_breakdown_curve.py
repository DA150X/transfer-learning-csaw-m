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
    get_test_results_for_label_network_and_sample_size,
)


def main():
    parser = get_argument_parser('network_breakdown_curve')
    args = parser.parse_args(sys.argv[1:])

    metric = 'auc'

    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)
    networks = get_networks(args.path_to_csv, metric)

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

    colors = {
        'DenseNet169': '#003f5c',
        'EfficientNetV2B0': '#bc5090',
        'ResNet50V2': '#ffa600',
    }

    xvals = [100, 500, 1000, 3000, 5000, 7000, 9523]

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    plt.title(make_title(metric))
    plt.ylabel(make_y_axis_label(metric))
    plt.xlabel('Sample size')
    plt.xticks([100, 3000, 5000, 7000, 9523], ['100', '3000', '5000', '7000', '9523'])

    x_all = []
    y_all = []

    for network in networks:
        inc = 0
        x = []
        y = []
        for sample_size in sample_sizes:
            avg = []
            for label in labels:
                values = get_test_results_for_label_network_and_sample_size(args.path_to_csv, metric, label, network, sample_size)
                for scale_factor, value in values.items():
                    avg.append(value)

            avg = sum(avg) / len(avg)

            x.append(xvals[inc])
            y.append(avg)
            x_all.append(xvals[inc])
            y_all.append(float(avg))
            inc += 1

        plt.plot(
            x,
            y,
            label=network,
            color=colors[network],
            linewidth=3
        )

    # calculate the trendline
    z = np.polyfit(x_all, y_all, 1)
    p = np.poly1d(z)
    plt.plot(
        x_all,
        p(x_all),
        linestyle='dotted',
        linewidth=3,
        color='#444444',
        alpha=0.7,
        label='Trendline'
    )

    plt.legend(loc='lower right', title='Legend')
    plt.ylim([min(plt.ylim()) - 0.25 * max(plt.ylim()), max(plt.ylim()) + 0.25 * max(plt.ylim())])
    plt.tight_layout()

    filename = f'all_{metric}'
    ensure_outputdir_and_write_chart(args.path_to_output + '/network_breakdown_curve', plt, filename, dpi=300)


def make_y_axis_label(metric):
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'f_1'
    elif metric == 'auc':
        metric = 'AUC'
    return r' ${{{metric}}}$'.format(metric=metric)


def make_title(metric):
    string = ''
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'F_1'
    elif metric == 'auc':
        metric = 'AUC'

    string += r'Average $\bf{{{metric}}}$ for each network accross all cancer labels'.format(metric=metric)
    return string


if __name__ == '__main__':
    main()
