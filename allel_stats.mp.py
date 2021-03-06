# -*-coding:utf-8 -*-
"""
@File    :  allel_stats.mp.py
@Time    :  2022/07/02 18:00:24
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  Calculate various popgen stats using scikit-allel and dask chunking
@Notes   :  Slow runtimes forces a perl parallel usage, but DASK seems to consume all processors
@Usage   :  python allel_stats.mp.py
"""

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass

import allel
import moments.LD as mold
import numpy as np
import pandas as pd
import scipy.spatial as ssp
import zarr
import dask

from dask.diagnostics.progress import ProgressBar
dask.config.set(**{'array.slicing.split_large_chunks': True})

# globals
chrom_lens = {
    "2R": 61_545_105,
    "2L": 49_364_325,
    "3R": 53_200_684,
    "3L": 41_963_435,
    "X": 24_393_108
    }

# TODO: add classes

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

def get_accessible(chroms, access_path):
    file = np.load(access_path)
    return {c: file[f"access_{c}"] for c in chroms}

@dataclass
class AllelData:
    __slots__ = ["meta", "gt", "pos", "ref", "alt", "aa", "cond"]
    meta: pd.DataFrame
    gt: allel.GenotypeDaskArray
    pos: allel.SortedIndex
    ref: list
    alt: list
    aa: list
    cond: list

@dataclass
class AllelDataAA:
    __slots__ = ["meta", "gt", "pos", "cond"]
    meta: pd.DataFrame
    gt: allel.GenotypeDaskArray
    pos: allel.SortedIndex
    cond: list

def get_windows(pos, start, stop, size=10000, step=None):
    return allel.position_windows(pos, start=start, stop=stop, size=size, step=step)

def get_equal_windows(accessible, start, stop, size=10000, step=None):
    return allel.equally_accessible_windows(accessible, size, start=start, stop=stop, step=step)

def write_stats(stat, stat_dt, outfile):
    n1 = "avg_pi" if stat == "da" else "bases"
    with open(f"agp3.{outfile}.{stat}.txt", 'w') as f:
        header = f"chromosome\tpopulation\twin_start\twin_stop\t{stat}\t{n1}\tvariants\n"
        f.write(f"{header}")
        for c in stat_dt:
            for pop in stat_dt[c]:
                for s, w, b, v in zip(stat_dt[c][pop][0], stat_dt[c][pop][1], stat_dt[c][pop][2], stat_dt[c][pop][3]):
                    f.write(f"{c}\t{pop}\t{w[0]}\t{w[1]}\t{s}\t{b}\t{v}\n")

def write_stats_zx(stat_dt, outfile):
    with open(f"agp3.{outfile}.zx.txt", 'w') as f:
        header = f"chromosome\tpopulation\twin_start\twin_stop\tzx\tz1\tz2\n"
        f.write(f"{header}")
        for c in stat_dt:
            for pop in stat_dt[c]:
                for zx, z1, z2, win in zip(stat_dt[c][pop][0], stat_dt[c][pop][1], stat_dt[c][pop][2], stat_dt[c][pop][3]):
                    f.write(f"{c}\t{pop}\t{win[0]}\t{win[1]}\t{zx}\t{z1}\t{z2}\n")

def write_stats_ld(stat_dt, outfile):
    with open(f"agp3.{outfile}.ld.txt", 'w') as f:
        header = f"chromosome\tpopulation\tdist_bp\tmean_D\tlower_D\tupper_D\n"
        f.write(f"{header}")
        for c in stat_dt:
            for pop in stat_dt[c]:
                for d, m, l, h in zip(stat_dt[c][pop][3], stat_dt[c][pop][0], stat_dt[c][pop][1], stat_dt[c][pop][2]):
                    f.write(f"{c}\t{pop}\t{d}\t{m}\t{l}\t{h}\n")

def get_ac(dt, pop=None, id="country"):
    # dt is an AllelData obj
    gt = dt.gt
    if pop:
        panel = dt.meta
        idx = panel[panel[f"{id}"] == pop].index.tolist()
        gt = gt.take(idx, axis=1)
    return gt.count_alleles(max_allele=1).compute(num_workers=workers)

def get_ac_subpops(dt, pops_ls, id="country"):
    # dt is an AllelData obj
    # pop_ls : list of pop names
    panel = dt.meta
    subpops = {sub:panel[panel[f"{id}"] == sub].index.tolist() for sub in pops_ls}
    gt = dt.gt.compute(num_workers=workers)
    return gt.count_alleles_subpops(subpops, max_allele=1)

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

def ld_win(chrom, dt, pop, id="country", decay=True, maf=0.10, r_min=1, r_max=10000, r_bin=100):
    # corrected - 1/n, where n is sampled chroms
    pos = dt.pos
    gt = dt.gt
    start_c = 1
    end_c = chrom_lens[chrom]
    if decay and chrom in ["3R", "3L"]:
        pos_mask = dt.pos < 37_000_00 if chrom == "3R" else ((dt.pos > 15_000_000) & (dt.pos < 41_000_000))
        pos = pos.compress(pos_mask)
        gt = gt.compress(pos_mask, axis=0)
        start_c = 1 if chrom == "3R" else 15_000_000
        end_c = 37_000_000 if chrom == "3R" else 41_000_000
    # get ac
    panel = dt.meta
    idx = panel[panel[f"{id}"] == pop].index.tolist()
    gt = gt.take(idx, axis=1)
    ac = gt.count_alleles(max_allele=1)
    # minor allele freq filter
    mac_filt = ac[:, :2].min(axis=1) > (maf * 2*len(idx))
    pos = pos.compress(mac_filt)
    gt = gt.compress(mac_filt, axis=0)
    wins = get_windows(pos, start_c, end_c, size=100000, step=None)
    ld_ls = []
    for s, e in wins:
        win = (pos >= s) & (pos < e)
        pos_r = pos.compress(win)
        gt_r = gt.compress(win)
        gn = gt_r.to_n_alt().compute(num_workers=workers)
        # get LD
        c2 = pos_r[:, None]
        pw_dist = ssp.distance.pdist(c2, 'cityblock')
        pw_ld = mold.Parsing.compute_pairwise_stats(gn)[0]
        ld_ls.append([np.mean(pw_ld[pw_dist == dist]) for dist in range(r_min, r_max, r_bin)])
    med = np.nanmedian(np.vstack(ld_ls), axis=0)
    lq = np.nanquantile(np.vstack(ld_ls), axis=0, q=0.025)
    hq = np.nanquantile(np.vstack(ld_ls), axis=0, q=0.95)
    dist = list(range(r_min, r_max, r_bin))
    return (med, lq, hq, dist)
    
def get_seg_bewteen(pos, ac1, ac2):
    loc_asc = ac1.is_segregating() & ac2.is_segregating()
    ac1_seg = ac1.compress(loc_asc, axis=0)
    ac2_seg = ac2.compress(loc_asc, axis=0)
    pos_s = pos.compress(loc_asc)
    return pos_s, ac1_seg, ac2_seg

def da_win(pos, ac1, ac2, accessible, windows, da=True):
    dxy, win, bases, counts = allel.windowed_divergence(pos, ac1, ac2, windows=windows, is_accessible=accessible)
    pi, win, bases, vars = pi_win(pos, (ac1+ac2), accessible, windows)
    da = dxy - pi
    return da, win, pi, vars

def dxy_win(pos, ac1, ac2, accessible, windows, da=False):
    dxy, win, bases, counts = allel.windowed_divergence(pos, ac1, ac2, windows=windows, is_accessible=accessible)
    if da:
        pi, win, bases, vars = pi_win(pos, (ac1+ac2), accessible, windows)
        da = dxy - pi
        return da, win, pi, vars
    return dxy, win, bases, counts

def fst_weir(dt, chrom_len, pops_ls, win_size=10000):
    panel = dt.meta
    subpops = {sub:panel[panel[f"{id}"] == sub].index.tolist() for sub in pops_ls}
    fst, win, counts = allel.windowed_weir_cockerham_fst(dt.pos, dt.gt, subpops, size=win_size, start=1, stop=chrom_len)
    #bases = [accessible[s:e].sum() for s, e, in win]
    #return fst, win, bases, counts

def fst_win(pos, ac1, ac2, accessible, windows, fst_algo='hudson'):
    if fst_algo == 'hudson':
        fst, win, counts = allel.windowed_hudson_fst(pos, ac1, ac2, windows=windows)
    elif fst_algo == 'patterson':
        fst, win, counts = allel.windowed_patterson_fst(pos, ac1, ac2, windows=windows)
    else:
        "FST algorithm not recognized, check keyword spelling"
        return None
    bases = [accessible[s:e].sum() for s, e, in win]
    return fst, win, bases, counts

def zxy_win(dt, pop1, pop2, windows, id="country", maf=0.10):
    panel = dt.meta
    idx_1 = panel[panel[f"{id}"] == pop1].index.tolist()
    idx_2 = panel[panel[f"{id}"] == pop2].index.tolist()
    idx_1_2 = idx_1 + idx_2
    ld_dt = {}
    for i, idx in enumerate([idx_1, idx_2, idx_1_2]):
        gt = dt.gt.take(idx, axis=1)
        ac = gt.count_alleles(max_allele=1)
        # minor allele freq filter
        mac_filt = ac[:, :2].min(axis=1) > (maf * 2*len(idx))
        pos = dt.pos.compress(mac_filt)
        gt = gt.compress(mac_filt, axis=0).compute(num_workers=workers)
        ld_win = []
        for s, e in windows:
            win = (pos >= s) & (pos < e)
            gt_r = gt.compress(win)
            gn = gt_r.to_n_alt()
            ld_win.append(mold.Parsing.compute_average_stats(gn)[0])
        ld_dt[str(i)] = np.array(ld_win)
    z_s1 = ld_dt["0"]
    z_s2 = ld_dt["1"]
    z_all = ld_dt["2"]
    z_x = (z_s1 + z_s2)/(2 * z_all)
    return z_x, z_s1, z_s2

def zx_win(dt, idx, windows, id="country", maf=0.10):
    gt = dt.gt.take(idx, axis=1)
    ac = gt.count_alleles(max_allele=1)
    # minor allele freq filter
    mac_filt = ac[:, :2].min(axis=1) > (maf * 2*len(idx))
    pos = dt.pos.compress(mac_filt)
    gt = gt.compress(mac_filt, axis=0).compute(num_workers=workers)
    ld_win = []
    for s, e in windows:
        win = (pos >= s) & (pos < e)
        gt_r = gt.compress(win)
        gn = gt_r.to_n_alt()
        ld_win.append(mold.Parsing.compute_average_stats(gn)[0])
    return np.array(ld_win)

def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument("zarr_path", help="zarr_path")
    parser.add_argument("meta_path", help="meta_path")
    parser.add_argument("access_path", help="access_path")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="outfile_prefix")
    parser.add_argument('-w', "--workers", type=int,
                        help="number of workers for Dask")
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
    access_path = args.access_path
    outfile = args.out_prefix
    global workers
    workers = args.workers
    win_size = args.window_size
    stats = args.stats
    if stats == "all":
        stats = ["pi", "theta", "tajd", "ld", "fst", "dxy", "da", "zx"]
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
    min_pops_size = 10
    with ProgressBar():
        chrom_dt = load_phased(CHROMS, meta_path = meta_path, zarr_path=zarr_path)
        chrom_aa_dt = remap_alleles(CHROMS, chrom_dt)
        access_dt = get_accessible(CHROMS, access_path)
        if pops == 'all':
            sample_size = chrom_aa_dt[CHROMS[0]].meta.groupby("country").count()["sample_id"]
            pops = sample_size.index[(sample_size >= min_pops_size).values].to_list()
        for s in stats:
            stat_dt = defaultdict(dict)
            if s in ["pi", "theta", "tajd", "fst", "dxy", "da"]:
                stat_fx = eval(f"{s}_win")
                for c in CHROMS:
                    windows = get_windows(chrom_aa_dt[c].pos, 1, chrom_lens[c], size=win_size, step=None)
                    ac_subpops = get_ac_subpops(chrom_aa_dt[c], pops)
                    if s in ["pi", "theta", "tajd"]:
                        for pop in pops:
                            ac = ac_subpops[pop]
                            ac_pos, ac_seg = get_seg(chrom_aa_dt[c].pos, ac)
                            stat, win, bases, vars = stat_fx(ac_pos, ac_seg, access_dt[c], windows)
                            stat_dt[c][pop] = (stat, win, bases, vars)
                    elif s in ["fst", "dxy", "da"]:
                        for p1, p2 in combinations(pops, 2):
                            p, ac1, ac2 = get_seg_bewteen(chrom_aa_dt[c].pos, ac_subpops[p1], ac_subpops[p2])                  
                            stat, win, bases, counts = stat_fx(p, ac1, ac2, access_dt[c], windows)
                            stat_dt[c][f"{p1}-{p2}"] = (stat, win, bases, counts)
                write_stats(s, stat_dt, outfile)
            elif s == "ld":
                for c in CHROMS:
                    for pop in pops:
                        stat_dt[c][pop] = ld_win(c, chrom_aa_dt[c], pop)
                write_stats_ld(stat_dt, outfile)
            elif s == "zx":
                for c in CHROMS:
                    windows = get_windows(chrom_aa_dt[c].pos, 1, chrom_lens[c], size=win_size, step=None)
                    panel = chrom_aa_dt[c].meta
                    subpops = {sub:panel[panel["country"] == sub].index.tolist() for sub in pops}
                    zx_dt = {}
                    for p1, p2 in combinations(pops, 2):
                        if p1 not in zx_dt:
                            zx_dt[p1] = zx_win(chrom_aa_dt[c], subpops[p1], windows)
                        if p2 not in zx_dt:
                            zx_dt[p2] = zx_win(chrom_aa_dt[c], subpops[p2], windows)
                        if f"{p1}-{p2}" not in zx_dt:
                            zx_dt[f"{p1}-{p2}"] = zx_win(chrom_aa_dt[c], subpops[p1]+subpops[p2], windows)    
                        zxy = (zx_dt[p1] + zx_dt[p2])/(2 * zx_dt[f"{p1}-{p2}"])
                        stat_dt[c][f"{p1}-{p2}"] = (zxy, zx_dt[p1], zx_dt[p2], windows)
                write_stats_zx(stat_dt, outfile)

if __name__ == "__main__":
    main()
