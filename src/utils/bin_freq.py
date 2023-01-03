# -*-coding:utf-8 -*-
"""
@File    :  bin_freq.py
@Time    :  2022/05/27 23:47:30
@Author  :  Scott T Small 
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  adapted from github.com/awohns/unified_genealogy_paper/all-data/bin_missing.py
@Notes   :  Sample data files with missing data create ancestors at many different time points,
often only one ancestor in each time point, which can cause difficulties parallelising
the inference. This script takes a sampledata file (usually containing missing data),
calculates the times-as-freq values, then bins them into frequency bands.
@Usage   :  python bin_missing.py infile.samples outfile.bin.samples [--decimal 3]
"""
import argparse
import numpy as np
import tsinfer
import tskit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="A tsinfer sample file ending in '.samples")
    parser.add_argument("output_file", help="A tsinfer sample file ending in '.samples")
    parser.add_argument("--decimal", type=int, default=3, help="what decimal to round to")
    args = parser.parse_args()

    sd = tsinfer.load(args.input_file).copy(path=args.output_file)
    times = sd.sites_time[:]
    for j, variant in enumerate(sd.variants()):
        time = variant.site.time
        if tskit.is_unknown_time(time):
            counts = tsinfer.allele_counts(variant.genotypes)
            # Non-variable sites have no obvious freq-as-time values
            assert counts.known != counts.derived
            assert counts.known != counts.ancestral
            assert counts.known > 0
            # Time = freq of *all* derived alleles. Note that if n_alleles > 2 this
            # may not be sensible: https://github.com/tskit-dev/tsinfer/issues/228
            times[variant.site.id] = counts.derived / counts.known
    times_f = np.around(times, args.decimal)
    times_f[times_f == 0] = 1/sd.num_samples
    sd.sites_time[:] = times_f
    print(
        "Number of samples:",
        sd.num_samples,
        ". Number of discrete times:",
        len(np.unique(sd.sites_time[:])),
    )
    sd.finalise()
