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
    parser = get_argument_parser('sample_size_scatter_all')
    args = parser.parse_args(sys.argv[1:])

    metric = 'auc'
    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)
    networks = get_networks(args.path_to_csv, metric)

    pylab.rcParams.update({
        'legend.title_fontsize': 'x-large',
        'legend.fontsize': 'x-large',
        'figure.titlesize': 'xx-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'font.size': 15,
        'font.family': 'Georgia',
    })

    colors = {
        'DenseNet169': '#003f5c',
        'EfficientNetV2B0': '#bc5090',
        'ResNet50V2': '#ffa600',
    }

    xvals = [100, 500, 1000, 3000, 5000, 7000, 9523]

    plt.figure(figsize=(25, 25))

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.suptitle(r'$\bf{{AUC}}$ score for the four cancer labels')

    pltnum = 0
    for label in labels:
        if pltnum == 0:
            ax = axs[0, 0]
        elif pltnum == 1:
            ax = axs[0, 1]
        elif pltnum == 2:
            ax = axs[1, 0]
        elif pltnum == 3:
            ax = axs[1, 1]
        pltnum += 1

        ax.set_title(label.replace('_', '  '))
        ax.set_ylabel(make_y_axis_label(metric, label))
        ax.set_xlabel('Sample size')
        ax.set_xticks([100, 3000, 5000, 7000, 9523], ['100', '3000', '5000', '7000', '9523'])

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

            sc = ax.scatter(
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
        ax.plot(
            x_all,
            p(x_all),
            linestyle='solid',
            linewidth=3,
            color='#444444',
            alpha=0.6,
        )

        ax.set_ylim([0.35, 0.8])

        if pltnum != 1:
            continue
        legend1 = fig.legend(
            title='Network',
            loc='lower left',
            labelspacing=2,
            borderpad=1,
            framealpha=1,
            frameon=False,
            ncol=3,
        )
        ax.add_artist(legend1)
        kw = dict(
            prop='sizes',
            num=3,
            color='grey',
            fmt='{x:.4g}',
            func=lambda s: (s / 250) - 1,
        )
        legend2 = fig.legend(
            *sc.legend_elements(**kw),
            loc='lower right',
            title='Dataset scale factor',
            labelspacing=1.4,
            borderpad=1,
            frameon=False,
            framealpha=1,
            ncol=3,
        )

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.20)
    filename = f'all_{metric}'
    ensure_outputdir_and_write_chart(args.path_to_output + '/sample_size_scatter', plt, filename, dpi=300)


def make_y_axis_label(metric, label):
    if metric == 'acc':
        metric = 'accuracy'
    elif metric == 'f1':
        metric = 'f_1'
    elif metric == 'auc':
        metric = 'AUC'
    return r' ${{{metric}}}$'.format(metric=metric)


if __name__ == '__main__':
    main()
