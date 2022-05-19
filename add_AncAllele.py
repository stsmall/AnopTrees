# -*-coding:utf-8 -*-
"""
@File    :   add_AncAllele.py
@Time    :   2022/05/18 13:16:01
@Author  :   Scott T Small
@Version :   1.0
@Contact :   stsmall@gmail.com
@License :   Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :   Adds the AA and AAProb fields to VCF from est-sfs output
@Notes   :   there can be 5 categories of AAProb
             min : min was ancestral
             not : neither ref/alt are anc
             dbl : the root had two diff alleles (not common)
             maj : the prob of maj/min was equal, default to maj
             NA : missing from est-sfs output, default to maj
@Usage   :   python add_AncAllele.py -v VCF -e est-sfs.outfile
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
    flipped = 0
    nomatch = 0
    bases = "ACGT"
    node_bases = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT",
                  "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    outfile_name = vcf_infile.removesuffix(".vcf.gz")
    with open(f"{outfile_name}.derived.vcf", 'w') as f:
        with gzip.open(vcf_infile, 'r') as vcf:
            for line in vcf:
                line = line.decode()
                if line.startswith("#"):
                    if line.startswith("##FORMAT"):
                        f.write('##INFO=<ID=AA,Number=1,Type=String,Description="Anc Allele">\n')
                        f.write('##INFO=<ID=AAProb,Number=A,Type=Float,Description="Prob Maj is Anc">\n')
                    f.write(line)
                else:
                    lin = line.split()
                    chrom = lin[0]
                    pos = lin[1]
                    ref = lin[3]
                    alt = lin[4]
                    fields = lin[7].split(";")
                    # find derived
                    chrom_pos = f"{chrom}_{pos}"
                    try:
                        anc = est_dt[chrom_pos]
                        maj = anc[0]
                        counts = list(map(int, anc[1].split(",")))
                        # num_alleles = counts.count(0)
                        min_ix, max_ix = sorted(counts)[-2:]
                        assert bases[counts.index(max_ix)] == maj
                        minor = bases[counts.index(min_ix)]
                        prob = float(anc[2])
                        if prob >= 0.50:
                            AA, AAprob = [maj, prob]
                        else:
                            aa_root = list(map(float, anc[3:]))
                            # check for ties
                            if sorted(aa_root)[-1] > sorted(aa_root)[-2]:
                                alt_ix = aa_root.index(max(aa_root))
                                nb = node_bases[alt_ix]
                                if nb[0] == nb[1]:
                                    if minor in nb:
                                        p = f"{max(aa_root)}-min"
                                        flipped += 1
                                    elif maj in nb:
                                        p = max(aa_root)
                                    else:
                                        p = f"{max(aa_root)}-not"
                                        nomatch += 1
                                    AA, AAprob = [nb[0], p]
                                else:
                                    AA, AAprob = [nb, f"{max(aa_root)}-dbl"]
                            else:
                                AA, AAprob = [maj, f"{max(aa_root)}-maj"]
                    except KeyError:
                        # count for major
                        calt = 0
                        cref = 0
                        for sample in lin[9:]:
                            gt = sample.split(":")[0]
                            calt += gt.count('1')
                            cref += gt.count('0')
                        maj = ref if cref >= calt else alt
                        AA, AAprob = [maj, "NA"]
                    if len(fields) == 1 and "." in fields:
                        lin[7] = f"AA={AA};AAProb={AAprob}"
                    else:
                        fields.insert(0, f"AA={AA};AAProb={AAprob}")
                        lin[7] = ";".join(fields)
                    f.write("{}\n".format("\t".join(lin)))
    print(f"{flipped} sites where anc is minor")
    print(f"{nomatch} sites where anc is not ref/alt")
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
