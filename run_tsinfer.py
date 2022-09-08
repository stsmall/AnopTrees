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
import msprime as msp
import daiquiri
import logging
daiquiri.setup(level=logging.INFO)


def truncate_anc(anc, low: float = 0, high: float = .5):
    """truncate ancestors

    in undated tree interval is from 0-1

    Parameters
    ----------
    anc : _type_
        _description_
    low : float, optional
        _description_, by default 0
    high : float, optional
        _description_, by default .5

    Returns
    -------
    _type_
        _description_
    """
    return anc.truncate_ancestors(low, high)


def generate_ancestors(samples_fn, num_threads, prefix, truncate=False):
    """_summary_

    Parameters
    ----------
    samples_fn : _type_
        _description_
    num_threads : _type_
        _description_
    prefix : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    sample_data = tsinfer.load(samples_fn)
    anc_data= tsinfer.generate_ancestors(sample_data,
                                      num_threads=num_threads,
                                      path=f"{prefix}.ancestors",
                                      progress_monitor=True)
    if truncate:
        anc_data = truncate_anc(anc_data)
        anc_data.dump(f"{prefix}.trunc.ancestors")
    return anc_data


def match_ancestors(samples_fn, anc, num_threads, r_prob, m_prob, prefix):
    """_summary_

    Parameters
    ----------
    samples_fn : _type_
        _description_
    anc : _type_
        _description_
    num_threads : _type_
        _description_
    r_prob : _type_
        _description_
    m_prob : _type_
        _description_
    prefix : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    sample_data = tsinfer.load(samples_fn)
    inferred_anc_ts = tsinfer.match_ancestors(
        sample_data,
        anc,
        num_threads=num_threads,
        recombination_rate=r_prob,
        mismatch_ratio=m_prob,
        progress_monitor=True
    )
    inferred_anc_ts.dump(f"{prefix}.atrees")
    return inferred_anc_ts


def match_samples(samples_fn, inferred_anc_ts, num_threads, r_prob, m_prob, prefix):
    """_summary_

    Parameters
    ----------
    samples_fn : _type_
        _description_
    inferred_anc_ts : _type_
        _description_
    num_threads : _type_
        _description_
    r_prob : _type_
        _description_
    m_prob : _type_
        _description_
    prefix : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    sample_data = tsinfer.load(samples_fn)
    inferred_ts = tsinfer.match_samples(
        sample_data,
        inferred_anc_ts,
        num_threads=num_threads,
        recombination_rate=r_prob,
        mismatch_ratio=m_prob,
        progress_monitor=True,
        simplify=False,
    )
    ts_path = f"{prefix}.no_simplify.trees"
    inferred_ts.dump(ts_path)


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
    return tsdate.add_sampledata_times(samples_fn, sites_time)


def parse_args(args_in):
    """Parse args."""
    versions = f"tskit version: {tskit.__version__}, tsinfer version: {tsinfer.__version__}"
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description="",
                                     prog=sys.argv[0],
                                     formatter_class=prog)
    parser.add_argument("samples",
                        help="The samples file name, as saved by tsinfer.SampleData.initialise()")
    parser.add_argument("prefix", help="The prefix of the output filename")
    parser.add_argument("-t", "--threads", default=1, type=int,
                        help="The number of worker threads to use")
    parser.add_argument("-s", "--step", default="infer", choices=["GA", "MA", "MS"],
                        help="Which step of the algorithm to run:"
                        "generate ancestors (GA), match ancestors"
                        "(MA), or match samples (MS) or all three (infer)")
    parser.add_argument("--recombination_rate", type=str, default='1e-8', help="file or float")
    parser.add_argument("--mismatch_ma", type=float, default=1, help="")
    parser.add_argument("--mismatch_ms", type=float, default=1, help="")
    parser.add_argument("--reinfer", action="store_true", help="reinfer on a dated tree")
    parser.add_argument("--truncate", action="store_true", help="truncate ancestors")
    parser.add_argument("-V", "--version", action="version", version=versions)
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =================================================================
    #  Gather args
    # =================================================================
    args = parse_args(sys.argv[1:])
    rec_rate = args.recombination_rate
    if rec_rate[0].isdigit():
        recombination_rate = float(rec_rate)
    else: # is file
        recombination_rate = msp.RateMap.read_hapmap(rec_rate, has_header=True)
    mismatch_ma = args.mismatch_ma
    mismatch_ms = args.mismatch_ms
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
            samples_fn=samples,
            num_threads=threads,
            prefix=prefix,
            truncate=args.truncate)
        if args.truncate:
            prefix = f"{prefix}.trunc"
        inferred_anc_ts = match_ancestors(
            samples_fn=samples,
            anc=anc,
            num_threads=threads,
            r_prob=recombination_rate,
            m_prob=mismatch_ma,
            prefix=prefix
            )
        match_samples(
            samples_fn=samples,
            inferred_anc_ts=inferred_anc_ts,
            num_threads=threads,
            r_prob=recombination_rate,
            m_prob=mismatch_ms,
            prefix=prefix)

    if args.step == "GA":
        anc = generate_ancestors(
            samples_fn=samples,
            num_threads=threads,
            prefix=prefix,
            truncate=args.truncate)
    if args.step == "MA":
        if not os.path.isfile(f"{prefix}.ancestors"):
            raise ValueError("No anc file")
        anc = tsinfer.load(f"{prefix}.ancestors")
        inferred_anc_ts = match_ancestors(
            samples_fn=samples,
            anc=anc,
            num_threads=threads,
            r_prob=recombination_rate,
            m_prob=mismatch_ma,
            prefix=prefix
            )
    if args.step == "MS":
        if not os.path.isfile(f"{prefix}.atrees"):
            raise ValueError("No atrees file")
        inferred_anc_ts = tskit.load(f"{prefix}.atrees")
        match_samples(
            samples_fn=samples,
            inferred_anc_ts=inferred_anc_ts,
            num_threads=threads,
            r_prob=recombination_rate,
            m_prob=mismatch_ms,
            prefix=prefix)


if __name__ == "__main__":
    main()
