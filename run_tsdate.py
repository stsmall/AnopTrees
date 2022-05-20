# -*-coding:utf-8 -*-
"""
@File    :  run_tsdate.py
@Time    :  2022/05/19 21:04:12
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  modified from awohns/unified_genealogy_paper
@Notes   :  None
@Usage   :  python run_tsdate.py
"""
import sys
import os
import argparse
import tsdate
import tskit
import tsinfer


def parse_args(args_in):
    """Parse args."""
    versions = f"tskit version: {tskit.__version__}, tsinfer version: {tsdate.__version__}"
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument("--verbosity", "-v", action="count", default=0)
    parser.add_argument("input", help="The input tree sequence file name")
    parser.add_argument("output", help="The path to write the output tree sequence file to")
    parser.add_argument("Ne", type=float, help="Effective population size")
    parser.add_argument("--mutation-rate", default=1e-8, type=float, help="Mutation rate")
    parser.add_argument("--recombination-rate", default=1e-8, type=float, help="recombination rate")
    parser.add_argument("--threads", default=1, type=int, help="")
    parser.add_argument("-V", "--version", action="version", version=versions)
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =================================================================
    #  Gather args
    # =================================================================
    args = parse_args(sys.argv[1:])
    print(args.mutation_rate)
    # =================================================================
    #  Main executions
    # =================================================================
    if not os.path.isfile(args.input):
        raise ValueError("No input tree sequence file")
    input_ts = tskit.load(args.input)
    input_ts_pre = tsdate.preprocess_ts(input_ts)
    prior = tsdate.build_prior_grid(input_ts_pre, Ne = args.Ne, approximate_priors=True)
    ts = tsdate.date(input_ts_pre,
                     mutation_rate=args.mutation_rate,
                     recombination_rate=args.recombination_rate,
                     priors=prior,
                     num_threads=args.threads,
                     ignore_oldest_root=True,
                     progress=True)
    ts.dump(args.output)


if __name__ == "__main__":
    main()
