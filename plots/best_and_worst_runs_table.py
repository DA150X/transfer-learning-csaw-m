import sys
import numpy as np
from beeprint import pp
from pathlib import Path
from decimal import Decimal
from common import (
    get_argument_parser,
    ensure_outputdir_and_write_chart,
    get_test_results_for_label_network_scale_factor_and_sample_size,
    get_labels,
    get_sample_sizes,
    get_networks,
    get_scale_factors,
)


def main():
    parser = get_argument_parser('best_runs_table')
    args = parser.parse_args(sys.argv[1:])

    output_filename_top = f'{args.path_to_output}/table-10.tex'
    if Path(output_filename_top).exists():
        Path(output_filename_top).unlink()

    with open(output_filename_top, 'w') as out:
        out.write('% this table is autogenerated \n')

    output_filename_bottom = f'{args.path_to_output}/table-11.tex'
    if Path(output_filename_bottom).exists():
        Path(output_filename_bottom).unlink()

    with open(output_filename_bottom, 'w') as out:
        out.write('% this table is autogenerated \n')

    create_chart_for_metric('auc', args)


def create_chart_for_metric(metric, args):
    output_filename_top = f'{args.path_to_output}/table-10.tex'
    output_filename_bottom = f'{args.path_to_output}/table-11.tex'
    sample_sizes = get_sample_sizes(args.path_to_csv, metric)
    networks = get_networks(args.path_to_csv, metric)
    scale_factors = get_scale_factors(args.path_to_csv, metric)
    labels = get_labels(args.path_to_csv, metric)

    scores = []
    for label in labels:
        if label != 'If_cancer':
            scale_factors_to_use = ['1']
        else:
            scale_factors_to_use = scale_factors
        for network in networks:
            for scale_factor in scale_factors_to_use:
                for sample_size in sample_sizes:
                    value = get_test_results_for_label_network_scale_factor_and_sample_size(
                        args.path_to_csv,
                        metric,
                        label,
                        network,
                        scale_factor,
                        sample_size
                    )
                    after = value['after']
                    if after is None:
                        continue

                    scores.append({
                        'auc': after,
                        'sample_size': sample_size,
                        'label': label.replace('_', ' '),
                        'network': network,
                        'scale_factor': scale_factor,
                    })

    scores.sort(key=lambda x: x['auc'], reverse=True)
    top_ten = scores[:10]
    scores.sort(key=lambda x: x['auc'])
    bottom_ten = scores[:10]

    with open(output_filename_top, 'a') as out:
        out.write(f'\\begin{{table}}[H]\n')
        out.write(f'\\begin{{center}}\n')
        out.write(f'\\caption{{The \\textbf{{best}} AUC performances recorded, and which data-set that result was achieved on}}\n')
        out.write(f'\\label{{tab:best_runs_table}}\n')
        out.write(f'\\\scriptsize\n')
        out.write(f'\\begin{{tabular}}{{l|c|c|l|l|c}}\n')
        out.write(f'& \\textbf{{AUC Score}} & \\textbf{{Sample Size}} & \\textbf{{Label}} & \\textbf{{Network}} & \\textbf{{Scale factor}}\\\\\n')
        out.write(f'\\hline\n')
        out.write(f'\\hline\n')
        out.write(f'\\noalign{{\\vskip 3pt}}\n')

    inc = 0
    for score in top_ten:
        inc += 1
        with open(output_filename_top, 'a') as out:
            out.write(f'\\textbf{{{inc}.}}')
            out.write(f' & {round(score["auc"], 2)}')
            out.write(f' & {score["sample_size"]}')
            out.write(f' & \\tt{{{score["label"]}}}')
            out.write(f' & \\tt{{{score["network"]}}}')
            out.write(f' & \\tt{{{score["scale_factor"]}x}}')

            out.write(f'\\\\\n')

    with open(output_filename_top, 'a') as out:
        out.write(f'\\noalign{{\\vskip 6pt}}\n')
        out.write(f'\\hline\n')
        out.write(f'\\noalign{{\\vskip 12pt}}\n')
        out.write(f'\\end{{tabular}}\n')
        out.write(f'\\end{{center}}\n')
        out.write(f'\\end{{table}}\n')
        out.write('\n')
        out.write('\n')

    with open(output_filename_bottom, 'a') as out:
        out.write(f'\\begin{{table}}[H]\n')
        out.write(f'\\begin{{center}}\n')
        out.write(f'\\caption{{The \\textbf{{worst}} AUC performances recorded, and which data-set that result was achieved on}}\n')
        out.write(f'\\label{{tab:worst_runs_table}}\n')
        out.write(f'\\\scriptsize\n')
        out.write(f'\\begin{{tabular}}{{l|c|c|l|l|c}}\n')
        out.write(f'& \\textbf{{AUC Score}} & \\textbf{{Sample Size}} & \\textbf{{Label}} & \\textbf{{Network}} & \\textbf{{Scale factor}}\\\\\n')
        out.write(f'\\hline\n')
        out.write(f'\\hline\n')
        out.write(f'\\noalign{{\\vskip 3pt}}\n')

    inc = 0
    for score in bottom_ten:
        inc += 1
        with open(output_filename_bottom, 'a') as out:
            out.write(f'\\textbf{{{inc}.}}')
            out.write(f' & {round(score["auc"], 2)}')
            out.write(f' & {score["sample_size"]}')
            out.write(f' & \\tt{{{score["label"]}}}')
            out.write(f' & \\tt{{{score["network"]}}}')
            out.write(f' & \\tt{{{score["scale_factor"]}x}}')

            out.write(f'\\\\\n')

    with open(output_filename_bottom, 'a') as out:
        out.write(f'\\noalign{{\\vskip 6pt}}\n')
        out.write(f'\\hline\n')
        out.write(f'\\noalign{{\\vskip 12pt}}\n')
        out.write(f'\\end{{tabular}}\n')
        out.write(f'\\end{{center}}\n')
        out.write(f'\\end{{table}}\n')
        out.write('\n')
        out.write('\n')


if __name__ == '__main__':
    main()
