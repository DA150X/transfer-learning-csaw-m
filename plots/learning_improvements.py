import sys
import numpy as np
from beeprint import pp
import matplotlib.pyplot as plt
from common import (
    get_argument_parser,
    ensure_outputdir_and_write_chart,
    get_test_results,
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
    values = get_test_results(args.path_to_csv, metric)

    # values = filter_by(
    #     values,
    #     lambda selector: 'EfficientNetV2B0' in selector,
    #     lambda selector: selector.replace('EfficientNetV2B0', '').strip()
    # )
    values = filter_by(
        values,
        lambda selector: 'If_cancer' not in selector and 'If_composite' not in selector and 'If_large_invasive_cancer' not in selector and 'If_interval_cancer' not in selector,
        lambda selector: selector,
    )
    values = group_by(
        values,
        lambda selector: selector.split('x')[0].split(' ')[1],
    )

    plt.figure(figsize=(8, 8))
    plt.legend(loc='upper left')
    plt.title(f'some graph')
    plt.ylabel('accuracy')
    plt.xlabel('network')
    plt.ylim([0, 1])
    plt.gca().get_xaxis().set_visible(False)

    inc = 0
    for selector, value in values.items():
        print(selector, value)
        x = inc
        y = value['before']
        dx = 0
        dy = value['after'] - value['before']
        plt.arrow(x, y, dx, dy, width=.01, head_width=.1, head_length=.01, length_includes_head=True, color='black')
        plt.annotate(
            selector,
            xy=(inc, 0.1),
        )
        inc += 1

    ensure_outputdir_and_write_chart(args.path_to_output + '/learning_improvements', plt, metric)


def filter_by(values, filter_func, new_selector_func):
    out = {}
    keys = values.keys()
    keys = list(filter(filter_func, keys))
    for key in keys:
        new_key = new_selector_func(key)
        out[new_key] = values[key]
    return out

def group_by(values, group_func):
    out = {}
    for key, value in values.items():
        new_key = group_func(key)
        if new_key not in out:
            out[new_key] = []
        out[new_key].append(value)

    for group, values in out.items():
        before = 0
        after = 0
        for val in values:
            before += val['before']
            after += val['after']
        out[group] = {
            'before': before / len(values),
            'after': after / len(values),
        }

    return out


if __name__ == '__main__':
    main()
