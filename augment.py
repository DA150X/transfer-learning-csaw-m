import sys
import argparse


def get_argument_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        prog='augment',
        description='Augment the CSAW-dataset',
        formatter_class=argparse.HelpFormatter,
    )

    parser.add_argument(
        'path_to_csaw_m',
        type=str,
        help='the path to the CSAW-M dataset',
    )

    parser.add_argument(
        'output',
        type=str,
        help='output directory for the samples',
    )

    parser.add_argument(
        'sample_sizes',
        type=str,
        help='comma separated list of sample sizes to pick',
    )

    parser.add_argument(
        'scale_factor',
        type=int,
        help='factor to scale the samples by',
    )

    return parser


def fail(msg):
    print(msg)
    sys.exit(1)


def parse_input(args):
    path_to_csaw_m = args.path_to_csaw_m
    output = args.output
    sample_sizes = args.sample_sizes.split(',')
    scale_factor = args.scale_factor

    for i, sample in enumerate(sample_sizes):
        try:
            sample_sizes[i] = int(sample)
        except ValueError:
            fail(f'bad sample size: "{sample}"')

    return path_to_csaw_m, output, sample_sizes, scale_factor


def main():
    parser = get_argument_parser()
    args = parser.parse_args(sys.argv[1:])

    path_to_csaw_m, output, sample_sizes, scale_factor = parse_input(args)

    print('Input parameters')
    print(f'path to csaw-m:   {path_to_csaw_m}')
    print(f'output directory: {output}')
    print(f'sample sizes:     {sample_sizes}')
    print(f'scale factor:     {scale_factor}')


if __name__ == '__main__':
    main()
