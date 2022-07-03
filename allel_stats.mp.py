# -*-coding:utf-8 -*-
"""
@File    :  allel_stats.mp.py
@Time    :  2022/07/02 18:00:24
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  None
@Notes   :  None
@Usage   :  python allel_stats.mp.py
"""

import sys
import argparse
from multiprocessing import Pool
import pickle
from collections import Counter, defaultdict
from itertools import combinations
# analyses
import allel
import scipy.spatial as ssp
import moments.LD as mold
# data
import dask
import dask.array as da
dask.config.set(**{'array.slicing.split_large_chunks': True})
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr

# globals
chrom_lens = {
    "2R": 61_545_105,
    "2L": 49_364_325,
    "3R": 53_200_684,
    "3L": 41_963_435,
    "X": 24_393_108
    }

# TODO: add classes


def vcf2zarr(chrom, vcf_path, zarr_path):
    allel.vcf_to_zarr(vcf_path, zarr_path, group=chrom, fields='*', alt_number=3, log=sys.stdout,compressor=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=False))

def load_data(file):
    with open(file, 'rb') as handle:
        dt = pickle.load(handle)
    return dt

def check_order(panel, samples):
    if np.all(samples == panel['sample_id'].values):
        order = True
        print("All in order")
        samples_list = list(samples)
        samples_callset_index = [samples_list.index(s) for s in panel['sample_id']]
        panel['callset_index'] = samples_callset_index
    else:
        print("Out of order")
        order = False
        samples_list = list(samples)
        samples_callset_index = [samples_list.index(s) for s in panel['sample_id']]
        panel['callset_index'] = samples_callset_index
        panel = panel.sort_values(by='callset_index')
        if np.all(samples == panel['sample_id'].values):
            print("All in order")
            order = True
        panel = panel.reset_index()
    return panel, order

def load_meta(file_path, is_X=False):
    dtypes = {'sample_id':'object', 'country':'object', 'location':'object', 'year':'int64', 'month':'int64',
            'latitude':'float64', 'longitude':'float64', 'aim_species':'object', 'sex_call':'object'}
    cols = ['sample_id', 'country', 'location', 'year', 'month', 'latitude', 'longitude', 'aim_species', "sex_call"]
    panel = pd.read_csv(file_path, sep=',', usecols=cols, dtype=dtypes)

    if is_X:
        panel = panel[panel["sex_call"] == "F"]
    panel.groupby(by=(['country', "location"])).count()
    return panel

def load_phased(CHROMS, meta_path, zarr_path):
    callset = zarr.open_group(zarr_path, mode='r')
    chrom_dt = {}
    panel = pd.DataFrame()
    order_ls = []
    for chrom in CHROMS:
        print(chrom)
        samples = callset[f'{chrom}/samples'][:]
        if chrom == "X":
            panel_x = load_meta(meta_path, is_X=True)
            panel, order = check_order(panel_x, samples)
        else:
            if len(panel.index) == 0:
                panel = load_meta(meta_path)
            panel, order = check_order(panel, samples)
            order_ls.append(order)
        assert all(order_ls)
        # load gt
        gt = allel.GenotypeDaskArray(callset[f'{chrom}/calldata/GT'])
        pos = allel.SortedIndex(callset[f'{chrom}/variants/POS'])
        aa = callset[f'{chrom}/variants/AA'][:]
        ref = callset[f'{chrom}/variants/REF'][:]
        alt = callset[f'{chrom}/variants/ALT'][:]
        cond = callset[f'{chrom}/variants/AAcond'][:]
        # save chrom to dict
        chrom_dt[chrom] = AllelData(panel, gt, pos, ref, alt, aa, cond)
    return chrom_dt

def remap_alleles(CHROMS, chrom_dt):
    '''remap genotype array for ancestral allele'''
    # create allele mapping
    chrom_aa_dt = {}
    for c in CHROMS:
        panel = chrom_dt[c].meta
        gt = chrom_dt[c].gt  # uncompress for mapping?
        pos = chrom_dt[c].pos
        cond = chrom_dt[c].cond
        ref = chrom_dt[c].ref
        aa = chrom_dt[c].aa
        # make masking array
        mapping = []
        for a, r in zip(aa, ref):
            if a == r:
                mapping.append([0, 1])
            else:
                mapping.append([1, 0])
        map_ar = np.array(mapping)
        gt_aa = gt.map_alleles(map_ar)
        # save chrom data to dict
        chrom_aa_dt[c] = AllelDataAA(panel, gt_aa, pos, cond)
    return chrom_aa_dt

def get_accessible(chroms):
    f = np.load("agp3.accessible_pos.txt.npz")
    access_dt = {}
    for c in chroms:
        access_dt[c] = f[f"access_{c}"]
    return access_dt

from dataclasses import dataclass
@dataclass
class AllelData:
    __slots__ = ["meta", "gt", "pos", "ref", "alt", "aa", "cond"]
    meta: pd.DataFrame
    gt: list
    pos: list
    ref: list
    alt: list
    aa: list
    cond: list

@dataclass
class AllelDataAA:
    __slots__ = ["meta", "gt", "pos", "cond"]
    meta: pd.DataFrame
    gt: list
    pos: list
    cond: list

def get_windows(pos, start, stop, size=10000, step=None):
    return allel.position_windows(pos, start=start, stop=stop, size=size, step=step)

def get_equal_windows(accessible, start, stop, size=10000, step=None):
    return allel.equally_accessible_windows(accessible, size, start=start, stop=stop, step=step)

def write_stats(stat, stat_dt, outfile):
    with open(f"agp3.{stat}.{outfile}.txt", 'w') as f:
        header = f"chromosome\tpopulation\twin_start\twin_stop\t{stat}\tvariants\tbases\n"
        f.write(f"{header}")
        for c in stat_dt:
            for pop in stat_dt[c]:
                for s, w, b, v in zip(stat_dt[c][pop][0], stat_dt[c][pop][1], stat_dt[c][pop][2], stat_dt[c][pop][3]):
                    f.write(f"{c}\t{pop}\t{w[0]}\t{w[1]}\t{s}\t{v}\t{b}\n")

def get_ac(dt, pop=None, id="country"):
    # dt is an AllelData obj
    gt = dt.gt
    if pop:
        panel = dt.meta
        idx = panel[panel[f"{id}"] == pop].index.tolist()
        gt = gt.take(idx, axis=1)
    return gt.count_alleles(max_allele=1)

def get_ac_subpops(dt, pops_ls, id="country"):
    # dt is an AllelData obj
    # pop_ls : list of pop names
    panel = dt.meta
    subpops = {sub:panel[panel[f"{id}"] == sub].index.tolist() for sub in pops_ls}
    return dt.gt.count_alleles_subpops(subpops, max_allele=1)

def get_seg(pos, ac):
    loc_asc = ac.is_segregating()
    ac_seg = ac.compress(loc_asc, axis=0)
    pos_s = pos.compress(loc_asc)
    return pos_s, ac_seg

def tajd_win(pos, ac, accessible, windows):
    tajd, win, vars = allel.windowed_tajima_d(pos, ac, windows=windows)
    bases = [accessible[s:e].sum() for s, e, in win]
    return tajd, win, bases, vars

def pi_win(pos, ac, accessible, windows):
    pi, win, bases, vars = allel.windowed_diversity(pos, ac, windows=windows, is_accessible=accessible)
    return pi, win, bases, vars

def theta_win(pos, ac, accessible, windows):
    theta, win, bases, vars = allel.windowed_watterson_theta(pos, ac, windows=windows, is_accessible=accessible)
    return theta, win, bases, vars

def ld_win(chrom, dt, pop=None, id="country", maf=0.10):
    # corrected - 1/n, where n is sampled chroms
    if chrom not in ["3R", "3L"]:
        return None
    pos = dt.pos
    gt = dt.gt
    pos_mask = dt.pos < 37_000_00 if chrom == "3R" else ((dt.pos > 15_000_000) & (dt.pos < 41_000_000))
    pos = pos.compress(pos_mask)
    gt = gt.compress(pos_mask, axis=0)
    # get ac
    panel = dt.meta
    idx = panel[panel[f"{id}"] == pop].index.tolist()
    gt = gt.take(idx, axis=1)
    ac = gt.count_alleles(max_allele=1)
    # minor allele freq filter
    mac_filt = ac[:, :2].min(axis=1) > (maf * 2*len(idx))
    pos = pos.compress(mac_filt)
    gt = gt.compress(mac_filt, axis=0)
    # start window slide
    start_c = 1 if chrom == "3R" else 15_000_000
    end_c = 37_000_000 if chrom == "3R" else 41_000_000
    wins = get_windows(pos, start_c, end_c, size=100000, step=None)
    ld_ls = []
    for s, e in wins:
        win = (pos >= s) & (pos < e)
        pos_r = pos.compress(win)
        gt_r = gt.compress(win)
        gn = gt_r.to_n_alt().compute()
        # get LD
        c2 = pos_r[:, None]
        pw_dist = ssp.distance.pdist(c2, 'cityblock')
        pw_ld = mold.Parsing.compute_pairwise_stats(gn)[0]
        ld_ls.append([(dist, np.mean(pw_ld[pw_dist == dist])) for dist in range(1, 10000, 100)])
    # TODO: stack and take mean across distances
    #ld = np.mean(ld_ls, axis=1)
    return ld_ls

#TODO: let's figure out this MP
def pi_win_mp(args_ls):
    pos, ac, accessible, windows = args_ls
    pi, win, bases, vars = allel.windowed_diversity(pos, ac, windows=windows, is_accessible=accessible)
    return pi #, win, bases, vars

def set_parallel(func, windows, nprocs, args):
    nprocs = nprocs
    pool = Pool(nprocs)
    # set chunks
    nk = nprocs * 10
    win_chunks = [windows[i:i + nk] for i in range(0, len(windows), nk)]
    # start job queue
    for win in win_chunks:
        args_ls = tuple(args + [win,])
        job = pool.apply_async(pi_win_mp, args=args_ls)
    pool.close()
    pool.join()
    return job


def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument("zarr_path", help="zarr_path")
    parser.add_argument("meta_path", help="meta_path")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="outfile_prefix")
    parser.add_argument('-n', "--nprocs", type=int, default=1,
                        help="number of processors")
    parser.add_argument("--stats", nargs='+', default="all", choices=["pi", "theta", "tajd", "fst", 'dxy', "da", "zx", "ld"], 
                        help="choose stats")
    parser.add_argument("--pops", type=str, nargs='+', default="all",
                        help="list populations")
    parser.add_argument("--chromosomes", type=str, nargs='+', default='all',
                        help="list chromosomes")
    parser.add_argument("--window_size", type=int, default=10_000,
                        help="window size")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =================================================================
    #  Gather args
    # =================================================================
    args = parse_args(sys.argv[1:])
    zarr_path = args.zarr_path
    meta_path = args.meta_path
    outfile = args.out_prefix
    nprocs = args.nprocs
    win_size = args.window_size
    stats = args.stats
    if stats == "all":
        stats = ["pi", "theta", "tajd"]
    CHROMS = args.chromosomes
    pops = args.pops
    if CHROMS == "all":
        CHROMS = ["2R", "2L", "3R", "3L", "X"]
    if not outfile:
        c = 'all' if args.chromosomes == "all" else "_".join(CHROMS)
        p = 'all' if pops == 'all' else "_".join(pops)
        outfile = f"{c}_{p}"    
    # =================================================================
    #  Main executions
    # =================================================================
    chrom_dt = load_phased(CHROMS, meta_path = meta_path, zarr_path=zarr_path)
    chrom_aa_dt = remap_alleles(CHROMS, chrom_dt)
    access_dt = get_accessible(CHROMS)
    for s in stats:
        stat_dt = defaultdict(dict)
        stat_fx = eval(f"{s}_win")
        for c in CHROMS:
            if pops == 'all':
                sample_size = chrom_aa_dt[c].meta.groupby("country").count()["sample_id"]
                pops = sample_size.index[(sample_size >= 10).values].to_list()
            ac_subpops = get_ac_subpops(chrom_aa_dt[c], pops)
            windows = get_windows(chrom_aa_dt[c].pos, 1, chrom_lens[c], size=win_size, step=None)
            for pop in pops:
                ac = ac_subpops[pop]
                ac_pos, ac_seg = get_seg(chrom_aa_dt[c].pos, ac)
                if nprocs > 1:
                    ac_seg = ac_seg.compute()
                    print(ac_seg.shape)
                    jobs = set_parallel("pi_win_mp", windows, 20, [ac_pos, ac_seg, access_dt[c]])
                else:
                    stat, win, bases, vars = stat_fx(ac_pos, ac_seg, access_dt[c], windows)
                    stat_dt[c][pop] = (stat, win, bases, vars)
        write_stats(s, stat_dt, outfile)

if __name__ == "__main__":
    main()
