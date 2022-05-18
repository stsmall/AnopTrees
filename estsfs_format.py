# -*-coding:utf-8 -*-
"""
@File    :   estsfs_format.py
@Time    :   2022/05/18 13:03:07
@Author  :   Scott T Small
@Version :   1.0
@Contact :   stsmall@gmail.com
@License :   Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :   Use the est-sfs program to infer ancestral states and uSFS.
@Notes   :   1) requires the allele counts from output of vcftools --counts
            2) count files must be gzipped
@Usage   :   python estsfs_format.py -i ingroup -o outgroup1 outgroup2
"""


import argparse
import contextlib
import gzip
import sys
from collections import defaultdict


def count_allele(counts_line, ingroup):
    """count alleles from vcftools count format.

    Parameters
    ----------
    counts_line : _type_
        _description_
    ingroup : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    bp_order = ["A", "C", "G", "T"]
    anc_list = [0, 0, 0, 0]  # A C G T

    ref, ref_count = counts_line[4].split(":")
    ref_count = int(ref_count)
    if len(ref) == 1 and ref_count > 0:
        bp_ix = bp_order.index(ref)
        if ingroup:
            anc_list[bp_ix] += ref_count
        else:
            anc_list[bp_ix] = 1
    with contextlib.suppress(IndexError):
        alt, alt_count = counts_line[5].split(":")
        alt_count = int(alt_count)
        if len(alt) == 1 and alt_count > 0:
            with contextlib.suppress(ValueError):
                bp_ix = bp_order.index(alt)
                if ingroup:
                    anc_list[bp_ix] += alt_count
                elif sum(anc_list) == 0:
                    anc_list[bp_ix] = 1
    return anc_list


def count_allele_tab(counts_line):
    """ counts alleles in tab format

    Parameters
    ----------
    counts_line : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    bp_order = ["A", "C", "G", "T"]
    anc_list = [0, 0, 0, 0]  # A C G T
    ref = counts_line[3]
    with contextlib.suppress(ValueError):
        bp_ix = bp_order.index(ref)
        anc_list[bp_ix] = 1
    return anc_list


def estsfs_format(file_ingroup, file_outgroup):
    """Read in allele counts for est-sfs input.

    Parameters
    ----------
    fileIngroup : str
        ingroup counts
    fileOutgroup : list
        outgroup counts

    Returns
    -------
    anc_dict : dict
        dictionary of anc sites
    hap_list : list
        returns the max of the haplist

    """
    anc_dict = defaultdict(list)
    ingroup = file_ingroup
    outgroups = file_outgroup
    # get ingroup counts
    with gzip.open(ingroup, 'r') as counts:
        line = next(counts)  # skip header
        for line in counts:
            line = line.decode()
            line = line.split()
            chrom = line[0]
            pos = line[1]
            site = f'{chrom}_{pos}'
            anc_counts = count_allele(line, ingroup=True)
            anc_dict[site].append(anc_counts)
    # get outgroup counts
    for file in outgroups:
        tab = False
        if file.endswith(".tab.gz"):
            tab = True
        with gzip.open(file, 'r') as counts:
            line = next(counts)  # skip header
            for line in counts:
                line = line.decode()
                line = line.split()
                chrom = line[0]
                pos = line[1]
                site = f'{chrom}_{pos}'
                if site in anc_dict:
                    if tab:
                        anc_counts = count_allele_tab(line)
                    else:
                        anc_counts = count_allele(line, ingroup=False)
                    anc_dict[site].append(anc_counts)

    return anc_dict


def estsfs_infiles(anc_dict, n_outgroup):
    """Run est-sfs.

    Parameters
    ----------
    anc_dict : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # create input file
    bases = ["A", "G", "C", "T"]
    counts = []
    first = next(iter(anc_dict.keys()))
    chrom = first.split("_")[0]
    with open(f"{chrom}.pos.txt", 'w') as out:
        with open(f"{chrom}.est.infile", 'w') as est:
            for key in anc_dict:
                chrom, pos = key.split("_")
                counts = [",".join(map(str, x)) for x in anc_dict[key]]
                while len(counts) < (n_outgroup + 1):
                    counts.append('0,0,0,0')
                est.write(f'{" ".join(counts)}\n')
                anc_counts = anc_dict[key][0]
                maj_ix = anc_counts.index(max(anc_counts))
                maj_allele = bases[maj_ix]
                out.write(f"{chrom}\t{pos}\t{maj_allele}\n")
    # create config file
    n_outgroups = len(counts) - 1
    with open(f"{chrom}.config.file", 'w') as config:
        config.write(f'n_outgroup={n_outgroups}\nmodel 1\nnrandom 1')


def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument('-i', "--ingroup", type=str, required=True,
                        help="ingroup/focalgroup counts")
    parser.add_argument('-o', "--outgroup", type=str, nargs='+', required=True,
                        help="outgroup counts")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =========================================================================
    #  Gather args
    # =========================================================================
    args = parse_args(sys.argv[1:])

    file_ingroup = args.ingroup
    file_outgroup = args.outgroup

    # =========================================================================
    #  Main executions
    # =========================================================================
    anc_dict = estsfs_format(file_ingroup, file_outgroup)
    estsfs_infiles(anc_dict, len(file_outgroup))


if __name__ == "__main__":
    main()
