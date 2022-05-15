import sys
import numpy as np
from beeprint import pp
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline
from common import (
    get_argument_parser,
    ensure_outputdir_and_write_chart,
)


def main():
    parser = get_argument_parser('auc_explanation')
    args = parser.parse_args(sys.argv[1:])

    pylab.rcParams.update({
        'legend.title_fontsize': 'xx-large',
        'legend.fontsize': 'xx-large',
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large',
        'font.size': 18,
    })

    colors = {
        'DenseNet169': '#003f5c',
        'EfficientNetV2B0': '#bc5090',
        'ResNet50V2': '#ffa600',
    }

    plt.figure(figsize=(14, 14))
    fig, ax = plt.subplots(figsize=(14, 14))

    plt.title('AUC - ROC Curve')
    plt.ylabel('(TPR) True Positive Rate')
    plt.xlabel('(FPR) False Positive Rate')

    plt.yticks([0.5, 1])
    plt.xticks([0, 0.5, 1])
    plt.tight_layout()

    plt.ylim([0, 1])
    plt.xlim([0, 1])

    x = np.linspace(0, 1, 1000)
    y = x
    plt.plot(x, y, '#444', linestyle='dashed', linewidth=6, alpha=0.7)

    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    y = [0, 0.4, 0.45, 0.55, 0.80, 0.86, 0.89, 0.95, 0.99, 1, 1]
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(min(x), max(x), 1000)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, '#003f5c', linestyle='solid', linewidth=6, alpha=0.9)
    plt.fill_between(
        x=X_,
        y1=Y_,
        color='#003f5c',
        alpha=0.2,
    )

    plt.text(
        0.75,
        0.25,
        'AUC',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=50,
    )
    plt.text(
        0.25,
        0.75,
        'ROC',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=50,
    )

    plt.setp(ax.spines.values(), linewidth=4)
    ax.spines['bottom'].set_color('#000')
    ax.spines['top'].set_color('#000')
    ax.spines['right'].set_color('#000')
    ax.spines['left'].set_color('#000')
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)

    filename = f'auc.png'
    ensure_outputdir_and_write_chart(args.path_to_output + '/auc_explanation', plt, filename, dpi=300)


if __name__ == '__main__':
    main()
