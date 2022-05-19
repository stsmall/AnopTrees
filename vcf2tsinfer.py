# -*-coding:utf-8 -*-
"""
@File    :  vcf2tsinfer.py
@Time    :  2022/05/19 09:20:25
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  Code to build tsinfer samples file from VCF.
            This code follows the example at:
            https://tsinfer.readthedocs.io/en/latest/tutorial.html
@Notes   :  For the redhat system, this version reserves too large of a file size and causes
            a memory error. The latest release contains the max file size option,
            max_file_size=2**37 that seems to fix this.
            https://tsinfer.readthedocs.io/en/latest/api.html#file-formats
            python -m pip install git+https://github.com/tskit-dev/tsinfer
@Usage   :  python vcf2tsinfer.py --vcf chr2L.recode.vcf --outfile chr2L \
            --meta FILE.meta.csv -t 2 --pops_header country &&
            tsinfer infer chr2L.samples -o -t 30
"""

import argparse
import sys

import cyvcf2
import pandas as pd
import tqdm
import tskit
import tsinfer

# TODO: allow parallel loading of large VCFs
# https://github.com/tskit-dev/tsinfer/issues/277#issuecomment-652024871


def add_metadata(vcf, samples, meta, label_by):
    """Add tsinfer meta data.

    Parameters
    ----------
    vcf : TYPE
        DESCRIPTION.
    samples : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    label_by : TYPE, optional
        DESCRIPTION. The default is country.

    Returns
    -------
    None.

    """
    pop_lookup = {}
    pop_lookup = {pop: samples.add_population(metadata={label_by: pop}) for pop in meta[label_by].unique()}
    for indiv in vcf.samples:
        meta_dict = meta.loc[indiv].to_dict()
        pop = pop_lookup[meta.loc[indiv][label_by]]
        lat = meta.loc[indiv]["latitude"]
        lon = meta.loc[indiv]["longitude"]
        samples.add_individual(ploidy=2, metadata=meta_dict, location=(lat, lon), population=pop)


def add_diploid_sites(vcf, samples):
    """Read the sites in the vcf and add them to the samples object.

    Reordering the alleles to put the ancestral allele first,
    if it is available.

    Parameters
    ----------
    vcf : TYPE
        DESCRIPTION.
    samples : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    with open("missing_data.txt", 'w') as f:
        progressbar = tqdm.tqdm(total=samples.sequence_length, desc="Read VCF", unit='bp')
        pos = 0
        for variant in vcf:
            progressbar.update(variant.POS - pos)
            if pos == variant.POS:
                raise ValueError("Duplicate positions for variant at position", pos)
            else:
                pos = variant.POS
            if any(not phased for _, _, phased in variant.genotypes):  # was ([TEXT]]
                raise ValueError("Unphased genotypes for variant at position", pos)
            alleles = [variant.REF] + variant.ALT
            ancestral = variant.INFO.get('AA', variant.REF)
            ordered_alleles = [ancestral] + list(set(alleles) - {ancestral})
            allele_index = {old_index: ordered_alleles.index(allele) for old_index,
                            allele in enumerate(alleles)}
            genotypes = [allele_index[old_index] for row in variant.genotypes for old_index in row[:2]]
            missing_genos = [i for i, n in enumerate(genotypes) if n == '.']
            for i in missing_genos:
                genotypes[i] = tskit.MISSING_DATA
                f.write("{}\t{}\n".format(pos, "\t".join(list(map(str, missing_genos)))))
            samples.add_site(pos, genotypes=genotypes, alleles=ordered_alleles)
        progressbar.close()


def chrom_len(vcf):
    """Get chromosome length."""
    assert len(vcf.seqlens) == 1
    return vcf.seqlens[0]


def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument("--vcfin", required=True, type=str)
    parser.add_argument("--outfile", required=True, type=str)
    parser.add_argument("--meta", required=True, type=str,
                        help="metadata for names and populations."
                        "Columns must include sample_id")
    parser.add_argument('-t', "--threads", type=int, default=1)
    parser.add_argument("--pops_header", type=str, default="country")

    return parser.parse_args(args_in)


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    vcf_path = args.vcfin
    outfile = args.outfile
    threads = args.threads
    label_by = args.pops_header
    meta = pd.read_csv(args.meta, sep=",", index_col="sample_id", dtype=object)
    # =========================================================================
    #  Main executions
    # =========================================================================

    vcf = cyvcf2.VCF(vcf_path)
    with tsinfer.SampleData(path=f"{outfile}.samples", sequence_length=chrom_len(vcf),
                            num_flush_threads=threads) as samples:
        add_metadata(vcf, samples, meta, label_by)
        add_diploid_sites(vcf, samples)

    print(f"Sample file created for {samples.num_samples} samples ({samples.num_individuals}) with {samples.num_sites} variable sites.", flush=True)


if __name__ == "__main__":
    main()
