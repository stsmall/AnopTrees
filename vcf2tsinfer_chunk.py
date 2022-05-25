# -*-coding:utf-8 -*-
"""
@File    :  vcf2tsinfer_chunk.py
@Time    :  2022/05/23 09:20:25
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
@Usage   :  python vcf2tsinfer_chunk.py --vcf chr2L.recode.vcf --outfile chr2L \
            --meta FILE.meta.csv -t 2 --pops_header country --chunk_size 100000 &&
            tsinfer infer chr2L.samples -o -t 30
"""

import argparse
import sys

import cyvcf2
import numpy as np
import pandas as pd
import tqdm
import tskit
import tsinfer
#TODO: vcf.set_samples(["AG1", AG2])

def add_metadata(vcf, samples, meta, label_by: str):
    """_summary_

    Parameters
    ----------
    vcf : _type_
        _description_
    samples : _type_
        _description_
    meta : _type_
        _description_
    label_by : str
        _description_
    """
    pop_lookup = {}
    pop_lookup = {pop: samples.add_population(metadata={label_by: pop}) for pop in meta[label_by].unique()}
    for indiv in vcf.samples:
        meta_dict = meta.loc[indiv].to_dict()
        pop = pop_lookup[meta.loc[indiv][label_by]]
        lat = meta.loc[indiv]["latitude"]
        lon = meta.loc[indiv]["longitude"]
        samples.add_individual(ploidy=2, metadata=meta_dict, location=(lat, lon), population=pop)


def create_sample_data(vcf, 
                       meta, 
                       label_by: str, 
                       outfile: str, 
                       threads: int, 
                       file_its: int):
    """_summary_

    Parameters
    ----------
    vcf : _type_
        _description_
    meta : _type_
        _description_
    label_by : str
        _description_
    outfile : str
        _description_
    threads : int
        _description_
    file_its : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    outfile_name = f"{outfile}.{file_its}"
    sample_data = tsinfer.SampleData(path=f"{outfile_name}.samples", num_flush_threads=threads)
    add_metadata(vcf, sample_data, meta, label_by)
    return sample_data


def add_meta_site(gff, chrom: str, pos: int):
    """_summary_

    Parameters
    ----------
    gff : _type_
        _description_
    chrom : str
        _description_
    pos : int
        _description_
    """
    gf_part = gff.query("type != 'chromosome'")
    gf_part = gf_part.query(f"contig == '{chrom}'")
    gf_part = gf_part.query(f"start <= '{pos}'")
    gf_part = gf_part.query(f"end >= '{pos}'")
    return gf_part.to_dict()


def add_diploid_sites(vcf,
                      meta, 
                      meta_gff, 
                      threads: int, 
                      outfile: str, 
                      label_by: str, 
                      chunk_size: int):
    """_summary_

    Parameters
    ----------
    vcf : _type_
        _description_
    meta : _type_
        _description_
    threads : int
        _description_
    outfile : str
        _description_
    label_by : str
        _description_
    chunk_size : int
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    exclude_ls = []
    file_its = 0
    chunk_count = 0
    sample_data = create_sample_data(vcf, meta, label_by, outfile, threads, file_its)
    chrom = vcf.seqnames[0]
    with open(f"{chrom}.not_inferred.txt", 'w') as t:
        with open(f"{chrom}.missing_data.txt", 'w') as f:
            progressbar = tqdm.tqdm(total=vcf.seqlens[0], desc="Read VCF", unit='bp')
            pos = 0
            for variant in vcf:
                assert variant.CHROM == vcf.seqnames[0]
                progressbar.update(variant.POS - pos)
                # quality checks
                if pos == variant.POS:
                    raise ValueError("Duplicate positions for variant at position", pos)
                else:
                    pos = variant.POS
                # must be phased
                if any(not phased for _, _, phased in variant.genotypes):  # was ([TEXT]]
                    raise ValueError("Unphased genotypes for variant at position", pos)
                # reordering around Ancestral
                alleles = [variant.REF] + variant.ALT
                ancestral = variant.INFO.get('AA')
                ancestral_prob = variant.INFO.get('AAProb')
                ordered_alleles = [ancestral] + list(set(alleles) - {ancestral})
                allele_index = {old_index: ordered_alleles.index(allele) for old_index,
                                allele in enumerate(alleles)}
                # should we use site for inference?
                # inference == False; triallelic: if len(ordered_alleles) > 2
                # inference == False; bad ancestral: AAProb in ['maje', 'majn', 'majm']
                inference = len(ordered_alleles) <= 2 and ancestral_prob not in ['maje', 'majn', 'majm']
                # genotypes
                genotypes = [allele_index[old_index] for row in variant.genotypes for old_index in row[:2]]
                # singleton/doubleton dont count in tsinfer, dont count towards chunk
                if sum(genotypes) > 2:
                    chunk_count += 1
                elif sum(genotypes) == 2:
                    if all(np.sum(np.array([genotypes[::2], genotypes[1::2]]), axis=0) != 2):
                        chunk_count += 1
                # handle missing genotypes
                missing_genos = [i for i, n in enumerate(genotypes) if n == '.']
                if len(missing_genos) > len(missing_genos) * .10:  # cap at 10% missing for a site
                    inference = False
                for i in missing_genos:
                    genotypes[i] = tskit.MISSING_DATA
                    f.write("{}\t{}\n".format(pos, "\t".join(list(map(str, missing_genos)))))
                # mark uninferred sites
                if not inference:
                    exclude_ls.append(pos)
                    t.write(f"{pos}\t{alleles}\t{ancestral_prob}\n")
                # add meta data to site from gff
                meta_pos = add_meta_site(meta_gff, variant.CHROM, pos)
                # add sites
                sample_data.add_site(pos, genotypes=genotypes, 
                                    alleles=ordered_alleles,
                                    metadata=meta_pos, 
                                    inference=inference)
                # check file size
                if chunk_count >= chunk_size:
                    sample_data.finalise()
                    file_its += 1
                    sample_data = create_sample_data(vcf, meta, label_by, outfile, threads, file_its)
                    chunk_count = 0
            progressbar.close()
    # catch final chunk
    if 0 < chunk_count < chunk_size:
        print("final sample, has {chunk_count} snps")
        sample_data.finalise()
    # save excluded position for use w/ generate ancestors
    np.savetxt(f"ga.{chrom}.exclude-pos.txt", np.array(exclude_ls))


def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument("--vcfin", required=True, type=str)
    parser.add_argument("--outfile", required=True, type=str)
    parser.add_argument("--meta", required=True, type=str,
                        help="metadata for names and populations."
                        "Columns must include sample_id")
    parser.add_argument("--gff", required=True, type=str,
                        help="metadata for positions.")
    parser.add_argument('-t', "--threads", type=int, default=1)
    parser.add_argument("--pops_header", type=str, default="country")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="number of snps per chunk, not counting singletons")
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
    chunks = args.chunk_size
    meta = pd.read_csv(args.meta, sep=",", index_col="sample_id", dtype=object)
    gff = pd.read_csv(args.gff, sep=",", dtype=object)    
    # =========================================================================
    #  Main executions
    # =========================================================================
    vcf = cyvcf2.VCF(vcf_path)
    add_diploid_sites(vcf=vcf,
                      meta=meta,
                      meta_gff=gff,
                      threads=threads,
                      outfile=outfile,
                      label_by=label_by,
                      chunk_size=chunks)

if __name__ == "__main__":
    main()
