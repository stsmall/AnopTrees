# -*-coding:utf-8 -*-
"""
@File    :  run_tsinfer.py
@Time    :  2022/05/19 20:22:31
@Author  :  Scott T Small 
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  modified from awohns/unified_genealogy_paper
@Notes   :  modified tsinfer CLI
@Usage   :  python run_tsinfer.py
"""

import sys
import argparse
import os
import tskit
import tsinfer
import tsdate

def generate_ancestors(samples_fn, num_threads, prefix):
    sample_data = tsinfer.load(samples_fn)
    anc = tsinfer.generate_ancestors(
        sample_data,
        num_threads=num_threads,
        path=f"{prefix}.ancestors",
        progress_monitor=True,
    )
    return anc


def match_ancestors(samples_fn, anc, num_threads, precision, r_prob, m_prob, prefix):
    sample_data = tsinfer.load(samples_fn)
    inferred_anc_ts = tsinfer.match_ancestors(
        sample_data,
        anc,
        num_threads=num_threads,
        precision=precision,
        recombination=r_prob,
        mismatch=m_prob,
        progress_monitor=True,
    )
    inferred_anc_ts.dump(f"{prefix}.atrees")
    return inferred_anc_ts


def match_samples(samples_fn, inferred_anc_ts, num_threads, r_prob, m_prob, precision, prefix):
    sample_data = tsinfer.load(samples_fn)
    inferred_ts = tsinfer.match_samples(
        sample_data,
        inferred_anc_ts,
        num_threads=num_threads,
        recombination=r_prob,
        mismatch=m_prob,
        precision=precision,
        progress_monitor=True,
        simplify=False,
    )
    ts_path = f"{prefix}.no_simplify.trees"
    inferred_ts.dump(ts_path)
    return inferred_ts


def reinfer_after_dating(samples_fn, dated_trees):
    """Reinfer trees after adding dates using tsdate.py
    
    follows the example : https://github.com/tskit-dev/tsdate/issues/191

    Parameters
    ----------
    samples_fn : _type_
        _description_
    dated_trees : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    dated = tskit.load(f"{dated_trees}.dated.trees").simplify()
    sites_time = tsdate.sites_time_from_ts(dated) 
    dated_samples = tsdate.add_sampledata_times(samples_fn, sites_time)   
    return dated_samples


def parse_args(args_in):
    """Parse args."""
    versions = f"tskit version: {tskit.__version__}, tsinfer version: {tsinfer.__version__}"    
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description="", 
                                     prog=sys.argv[0],
                                     formatter_class=prog)  
    parser.add_argument("--verbosity", "-v", action="count", default=0)
    parser.add_argument("samples", 
                        help="The samples file name, as saved by tsinfer.SampleData.initialise()")
    parser.add_argument("prefix", help="The prefix of the output filename")
    parser.add_argument("-t", "--threads", default=1, type=int,
                        help="The number of worker threads to use")
    parser.add_argument("-s", "--step", default="infer", choices=["GA", "MA", "MS"],
                        help="Which step of the algorithm to run:"
                        "generate ancestors (GA), match ancestors"
                        "(MA), or match samples (MS) or all three (infer)")
    parser.add_argument("--rho", default=0.01, help="")
    parser.add_argument("--mismatch_ma", default=1, help="")
    parser.add_argument("--mismatch_ms", default=1, help="")
    parser.add_argument("-p", "--precision", default=10, type=int,
                        help="The precision parameter to pass to the function")
    parser.add_argument("--reinfer", action="store_true", help="reinfer on a dated tree")
    parser.add_argument("-V", "--version", action="version", version=versions)
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =================================================================
    #  Gather args
    # =================================================================
    args = parse_args(sys.argv[1:])
    r_prob = args.rho
    ma_prob = args.mismatch_ma
    ms_prob = args.mismatch_ms
    prefix = args.prefix
    threads = args.threads
    samples = args.samples
    # =================================================================
    #  Main executions
    # =================================================================
    if not os.path.isfile(samples):
        raise ValueError("No samples file")

    if args.reinfer:
        if not os.path.isfile(f"{prefix}.dated.trees"):
            raise ValueError("only can reinfer on dated trees")
        samples = reinfer_after_dating(samples, prefix)
        prefix = f"{prefix}-reinfer"

    if args.step == "infer":
        anc = generate_ancestors(
            samples,
            threads,
            prefix)
        inferred_anc_ts = match_ancestors(
            samples, 
            anc, 
            threads, 
            args.precision, 
            r_prob, 
            ma_prob, 
            prefix)
        match_samples(
            samples, 
            inferred_anc_ts, 
            threads, 
            r_prob, 
            ms_prob, 
            args.precision, 
            prefix)
    
    if args.step == "GA":
        anc = generate_ancestors(
            samples,
            threads,
            prefix)
    if args.step == "MA":
        anc = tsinfer.load(f"{args.prefix}.truncated.ancestors")
        inferred_anc_ts = match_ancestors(
            samples,
            anc,
            threads,
            args.precision,
            r_prob,
            ma_prob,
            prefix)
    if args.step == "MS":
        anc = tsinfer.load(f"{args.prefix}.truncated.ancestors")
        inferred_anc_ts = tskit.load(f"{args.prefix}.atrees")
        match_samples(
            samples,
            inferred_anc_ts,
            threads,
            r_prob,
            ms_prob,
            args.precision,
            prefix,
        )


if __name__ == "__main__":
    main()
