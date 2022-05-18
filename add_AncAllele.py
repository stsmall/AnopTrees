# -*-coding:utf-8 -*-
"""
@File    :   add_AncAllele.py
@Time    :   2022/05/18 13:16:01
@Author  :   Scott T Small
@Version :   1.0
@Contact :   stsmall@gmail.com
@License :   Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :   None
@Notes   :   None
@Usage   :   python add_AncAllele.py
"""

import argparse
import sys
import gzip


def read_estsfs(est_file):
    """Loads est-sfs File in dictionary"""
    est_dt = {}
    with open(est_file, 'r') as est:
        for line in est:
            lin = line.split()
            chr_pos = f"{lin[0]}_{lin[1]}"
            est_dt[chr_pos] = lin[2:]  # major probMajor
    return est_dt


def add_aa(est_dt, vcf_infile):
    """Add anc allele, AA, to vcf.

    Parameters
    ----------
    est_dt : _type_
        _description_
    vcf_infile : _type_
        _description_
    """
    with open(f"{vcf_infile}.derived", 'w') as f:
        with gzip.open(vcf_infile, 'r') as vcf:
            for line in vcf:
                line = line.decode()
                if line.startswith("#"):
                    f.write(line)
                else:
                    lin = line.split()
                    chrom = lin[0]
                    pos = lin[1]
                    ref = lin[3]
                    fields = lin[7].split(";")
                    # find derived
                    chrom_pos = f"{chrom}_{pos}"
                    try:
                        AA, AAprob = est_dt[chrom_pos]
                    except KeyError:
                        AA, AAprob = [ref, 0.0]
                    if len(fields) == 1 and "." in fields:
                        lin[7] = f"AA={AA};AAprob={AAprob}"
                    else:
                        fields.insert(0, f"AA={AA};AAprob={AAprob}")
                        lin[7] = ";".join(fields)
                    f.write("{}\n".format("\t".join(lin)))
    return None


def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument('-v', "--vcfFile", type=str, required=True, help="")
    parser.add_argument('-e', "--estFile", type=str, required=True, help="")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =================================================================
    #  Gather args
    # =================================================================
    args = parse_args(sys.argv[1:])
    # =================================================================
    #  Main executions
    # =================================================================
    est_dt = read_estsfs(args.estFile)
    add_aa(est_dt, args.vcfFile)


if __name__ == "__main__":
    main()
