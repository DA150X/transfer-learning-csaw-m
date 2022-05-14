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
    parser = get_argument_parser('sample_size_scatter')
    args = parser.parse_args(sys.argv[1:])

    metrics = [
        'auc',
        'loss',
    ]
    for metric in metrics:
        create_chart_for_metric(metric, args)

    write_legend(args)


def create_chart_for_metric(metric, args):
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

    for label in labels:
        plt.figure(figsize=(14, 10))
        fig, ax = plt.subplots(figsize=(14, 10))

        plt.title(make_title(metric, label))
        plt.ylabel(make_y_axis_label(metric, label))
        plt.xlabel('Sample size')
        plt.xticks([100, 3000, 5000, 7000, 9523], ['100', '3000', '5000', '7000', '9523'])

        x_all = []
        y_all = []
        for network in networks:
            inc = 0
            x = []
            y = []
            z = []
            for sample_size in sample_sizes:
                values = get_test_results_for_label_network_and_sample_size(args.path_to_csv, metric, label, network, sample_size)
                for scale_factor, value in values.items():
                    if label != 'If_cancer':
                        if scale_factor != '1':
                            print(network, label, sample_size, scale_factor)
                            continue
                    scale_factor = int(scale_factor)
                    x.append(xvals[inc])
                    y.append(value)
                    x_all.append(xvals[inc])
                    y_all.append(float(value))
                    z.append(100 + (350 * scale_factor))
                inc += 1

            sc = plt.scatter(
                x,
                y,
                s=z,
                alpha=0.5,
                label=network,
                color=colors[network],
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
        )

        plt.ylim([min(plt.ylim()) - 0.2 * max(plt.ylim()), max(plt.ylim()) + 0.2 * max(plt.ylim())])
        # plt.xlim([min(plt.xlim()) - 0.2 * max(plt.xlim()), max(plt.xlim()) + 0.2 * max(plt.xlim())])
        plt.tight_layout()

        filename = f'{label}_{metric}'
        ensure_outputdir_and_write_chart(args.path_to_output + '/sample_size_scatter', plt, filename, dpi=300)


def write_legend(args):
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
        'legend.edgecolor': '#ffffff',
    })

    colors = {
        'DenseNet169': '#003f5c',
        'EfficientNetV2B0': '#bc5090',
        'ResNet50V2': '#ffa600',
    }

    for label in labels:
        plt.figure(figsize=(14, 10))
        fig, ax = plt.subplots(figsize=(14, 10))
        x_all = []
        y_all = []
        for network in networks:
            inc = 0
            x = []
            y = []
            z = []
            for sample_size in sample_sizes:
                values = get_test_results_for_label_network_and_sample_size(args.path_to_csv, metric, label, network, sample_size)
                for scale_factor, value in values.items():
                    scale_factor = int(scale_factor)
                    x.append(inc)
                    y.append(value)
                    x_all.append(inc)
                    y_all.append(float(value))
                    z.append(100 + (350 * scale_factor))
                inc += 1

            sc = plt.scatter(
                x,
                y,
                s=z,
                alpha=0.5,
                label=network,
                color=colors[network],
            )
            legend1 = ax.legend(
                title='Network',
                loc='upper left',
                labelspacing=2,
                borderpad=.2,
                framealpha=1,
                frameon=True,
            )
            ax.add_artist(legend1)
            kw = dict(
                prop='sizes',
                num=3,
                color='grey',
                fmt='{x:.4g}',
                func=lambda s: (s / 250) - 1,
            )
            legend2 = ax.legend(
                *sc.legend_elements(**kw),
                loc='lower right',
                title='Dataset scale factor',
                labelspacing=1.4,
                borderpad=1.2,
                frameon=True,
                framealpha=1,
            )

        fig = legend1.figure
        fig.canvas.draw()
        bbox = legend1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        filename = args.path_to_output + '/sample_size_scatter/legend_1.png'
        fig.savefig(filename, dpi='figure', bbox_inches=bbox, pad_inches=0)

        fig = legend2.figure
        fig.canvas.draw()
        bbox = legend2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        filename = args.path_to_output + '/sample_size_scatter/legend_2.png'
        fig.savefig(filename, dpi='figure', bbox_inches=bbox, pad_inches=0)


def make_y_axis_label(metric, label):
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'f_1'
    elif metric == 'auc':
        metric = 'AUC'
    return r' ${{{metric}}}$'.format(metric=metric)


def make_title(metric, label):
    string = ''
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'F_1'
    elif metric == 'auc':
        metric = 'AUC'

    label = label.replace('_', '\_')
    string += r' $\bf{{{metric}}}$ for the $\bf{{{label}}}$ label'.format(metric=metric, label=label)
    return string


if __name__ == '__main__':
    main()
