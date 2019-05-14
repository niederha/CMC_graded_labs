"""Parse command line arguments"""

import argparse


def save_plots():
    """Parse args"""
    parser = argparse.ArgumentParser(description=(
        "CMC lab"
    ))
    parser.add_argument(
        "--save", "-s",
        help="Save figures",
        dest="save",
        action="store_true",
        default=False
    )
    return parser.parse_args().save

