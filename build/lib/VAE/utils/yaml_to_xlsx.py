"""Collect configurations in Excel file."""

import argparse

from VAE.utils.collection import TrainerConfigCollection


def main(args):
    collection = TrainerConfigCollection(path=args.path, filemask=args.filemask, recursive=args.recursive)
    collection.to_excel(filename=args.output, column_width=17)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect configurations in Excel file.")
    parser.add_argument('path', help="Path to the configuration files.")
    parser.add_argument('--filemask',
                        default='*.yaml',
                        help="Filemask of the configuration files (default: %(default)s)")
    parser.add_argument('--recursive', default=True, help="Recursive search in subdirectories (default: %(default)s)")
    parser.add_argument('--output',
                        default='trainer_configs.xlsx',
                        help="Output name of Excel file (default: %(default)s)")
    args = parser.parse_args()
    main(args)
