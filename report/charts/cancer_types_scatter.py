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
    parser = get_argument_parser('cancer_types_scatter')
    args = parser.parse_args(sys.argv[1:])

    metric = 'auc'

    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)
    networks = get_networks(args.path_to_csv, metric)

    pylab.rcParams.update({
        'legend.title_fontsize': 'large',
        'legend.fontsize': 'large',
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

    plt.figure(figsize=(25, 25))

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.suptitle(r'$\bf{{AUC}}$ score for each network and cancer label')

    legend_labels = []

    pltnum = 0
    for network in ['All Networks'] + networks:
        if pltnum == 0:
            ax = axs[0, 0]
        elif pltnum == 1:
            ax = axs[0, 1]
        elif pltnum == 2:
            ax = axs[1, 0]
        elif pltnum == 3:
            ax = axs[1, 1]
        pltnum += 1

        ax.set_title(network, fontweight='bold')
        ax.set_ylabel(make_y_axis_label(metric))
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')

        ax.set_xticks([0, 1, 2, 3], [
            'If cancer',
            'If composite',
            'If interval cancer',
            'If large invasive cancer',
        ])

        inc = 0
        for label in labels:
            if network != 'All Networks':
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

                sc = ax.scatter(
                    x,
                    y,
                    s=z,
                    alpha=0.3,
                    label=network,
                    color=colors[network],
                )
                inc += 1
            else:
                for n in networks:
                    x = []
                    y = []
                    z = []
                    values = get_test_results_for_label_and_network(args.path_to_csv, metric, label, n)
                    for sample_size, vals in values.items():
                        value = []
                        for scale_factor, metric_value in vals.items():
                            value.append(metric_value)

                        avg = float(sum(value) / len(value))
                        x.append(inc)
                        y.append(avg)
                        z.append(200 + int(sample_size) / 4)

                    sc = ax.scatter(
                        x,
                        y,
                        s=z,
                        alpha=0.3,
                        label=n if n not in legend_labels else "",
                        color=colors[n],
                    )
                    if n not in legend_labels:
                        legend_labels.append(n)
                inc += 1

        ax.set_ylim([0.35, 0.8])
        ax.set_xlim([-0.5, 3.5])

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
            func=lambda s: -200 + s * 4 - 300,
        )
        legend2 = fig.legend(
            *sc.legend_elements(**kw),
            loc='lower right',
            title='Sample Size',
            labelspacing=1.4,
            borderpad=1,
            frameon=False,
            framealpha=1,
            ncol=4,
        )

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.20)
    filename = f'{metric}'
    ensure_outputdir_and_write_chart(args.path_to_output + '/cancer_types_scatter', plt, filename, dpi=300)


def make_y_axis_label(metric):
    if metric == 'auc':
        metric = 'AUC'
    elif metric == 'loss':
        metric = 'Loss'
    return r' ${{{metric}}}$'.format(metric=metric)


if __name__ == '__main__':
    main()
