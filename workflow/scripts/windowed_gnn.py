# -*-coding:utf-8 -*-
"""
@File    :  windowed_gnn.py
@Time    :  2022/10/17 15:59:13
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  Runs genealogical nearest neighbor function on input trees can
            use either windowed or whole chromosome
@Notes   :  This module is a direct implementation of:
                https://github.com/tskit-dev/tskit/issues/665
@Usage   :  python windowed_gnn.py --tree FOO.chrom.trees --groups country --outfile FOO.chrom.gnn.csv
            python windowed_gnn.py --tree FOO.chrom.trees --groups country --outfile FOO.chrom.gnn.csv
            --tar Mayotte --gnn_windows --ancestry
"""

import sys
import time
from os import path
import argparse
from collections import defaultdict
import tskit
import json
import numpy as np


def gnn_fx(outfile, ts, ref_samples, target_samples, groups, chrom, ancestry, samples="sample_name"):
    """_summary_

    Parameters
    ----------
    outfile : str
        name for outfile
    ts : object
        treesequence
    ref_samples : list
        list of int denoting nodes(haps) in the treesequence
    target_samples : list
        list of int denoting nodes(haps) in the treesequence
    groups : str
        keyword for grouping samples
    chrom : str
        chromosome, used in outfile
    ancestry : bool
        use ancestry by removing self in comparisons
    samples : str, optional
        name in metadata for sample, by default "sample_name"
    """
    if not any(isinstance(i, list) for i in target_samples):
        target_samples = [target_samples]
    # get all pop_ids
    sample_nodes = [ts.node(n) for r in ref_samples for n in r]
    ref_pop = [json.loads(ts.individual(n.individual).metadata)[groups] for n in sample_nodes]
    ref_pop_ids = list(dict.fromkeys(ref_pop))
    if not ancestry:
        target_samples = [[n for r in target_samples for n in r]]
    # open file yo
    with open(f"{outfile}.gnn.csv", 'w') as f:
        f.write(f'chromosome,sample_id,target_pop,{",".join(ref_pop_ids)}\n')
        # start looping
        for temp_tar in target_samples:
            if ancestry:
                temp_ref = [x for x in ref_samples if x != temp_tar]
            else:
                temp_ref = ref_samples
            # calc gnn
            gnn = ts.genealogical_nearest_neighbours(temp_tar, temp_ref)
            # get tar pops
            sample_nodes = [ts.node(n) for n in temp_tar]
            sample_names = [json.loads(ts.individual(n.individual).metadata)[samples] for n in sample_nodes]
            tar_pop = [json.loads(ts.individual(n.individual).metadata)[groups] for n in sample_nodes]
            # insert empty np.array into pos where tar_pop would be
            if ancestry:
                ix = ref_pop_ids.index(tar_pop[0])
                gnn = np.insert(gnn, ix, np.zeros(len(gnn)), axis=1)
            # gen sample IDs
            inds = []
            for samp in sample_names:
                totalcount = inds.count(f"{samp}_0")
                inds.append(f"{samp}_1" if totalcount > 0 else f"{samp}_0")
            # write out df
            try:
                for i, s in enumerate(inds):
                    gnns = ",".join(map(str, gnn[i]))
                    f.write(f"{chrom},{s},{tar_pop[i]},{gnns}\n")
            except IndexError:
                breakpoint()

def parse_time_windows(ts, time_windows):
    """Parse time windows.

    Parameters
    ----------
    ts : object
        treesequence
    time_windows : list
        int or float referencing time in the treesequence

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    T = ts.max_root_time
    if time_windows is None:
        time_windows = [0.0, T]
    return np.array(time_windows)


def windowed_gnn(ts,
                focal,
                reference_sets,
                windows=None,
                time_windows=None,
                span_normalise=True,
                time_normalise=True):
    """Run GNN in windows.

    Genealogical_nearest_neighbours with support for span- and time-based windows.

    Parameters
    ----------
    ts : Obj
        tskit tree object
    focal : list
        focal nodes
    reference_sets : list
        reference nodes
    windows : TYPE, optional
        DESCRIPTION. The default is None.
    time_windows : TYPE, optional
        DESCRIPTION. The default is None.
    span_normalise : TYPE, optional
        DESCRIPTION. The default is True.
    time_normalise : TYPE, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        if duplicate samples

    Returns
    -------
    A : numpy array


    """
    reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for k, reference_set in enumerate(reference_sets):
        for u in reference_set:
            if reference_set_map[u] != -1:
                raise ValueError("Duplicate value in reference sets")
            reference_set_map[u] = k
    # check set spatial windows
    if windows is None:
        windows_parsed = ts.parse_windows("trees")
    else:
        windows_parsed = ts.parse_windows(windows)
    num_windows = windows_parsed.shape[0] - 1
    # check set time windows
    if time_windows is None:
        time_windows_parsed = parse_time_windows(ts, time_windows)
    else:
        time_windows_parsed = time_windows  # ts.parse_windows(time_windows)
    num_time_windows = time_windows_parsed.shape[0] - 1
    # set recording mats
    A = np.zeros((num_windows, num_time_windows, len(focal), len(reference_sets)))
    K = len(reference_sets)
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    sample_count = np.zeros((ts.num_nodes, K), dtype=int)
    time = ts.tables.nodes.time
    norm = np.zeros((num_windows, num_time_windows, len(focal)))

    # Set the initial conditions.
    for j in range(K):
        sample_count[reference_sets[j], j] = 1

    window_index = 0
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = -1
            v = edge.parent
            while v != -1:
                sample_count[v] -= sample_count[edge.child]
                v = parent[v]
        for edge in edges_in:
            parent[edge.child] = edge.parent
            v = edge.parent
            while v != -1:
                sample_count[v] += sample_count[edge.child]
                v = parent[v]

        # Update the windows
        assert window_index < num_windows
        while (windows_parsed[window_index] < t_right and window_index + 1 <= num_windows):
            w_left = windows_parsed[window_index]
            w_right = windows_parsed[window_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            span = right - left
            # Process this tree.
            for j, u in enumerate(focal):
                focal_reference_set = reference_set_map[u]
                delta = int(focal_reference_set != -1)
                p = u
                while p != tskit.NULL:
                    total = np.sum(sample_count[p])
                    if total > delta:
                        break
                    p = parent[p]
                if p != tskit.NULL:
                    scale = span / (total - delta)
                    time_index = np.searchsorted(time_windows_parsed, time[p]) - 1
                    if time_index < num_time_windows and time_index >= 0:
                        for k, _reference_set in enumerate(reference_sets):
                            n = sample_count[p, k] - int(focal_reference_set == k)
                            A[window_index, time_index, j, k] += n * scale
                        norm[window_index, time_index, j] += span
            assert span > 0
            if w_right <= t_right:
                window_index += 1
            else:
                # This interval crosses a tree boundary, so we update it again
                # in the for the next tree
                break

    # Reshape norm depending on normalization selected
    # Return NaN when normalisation value is 0
    if span_normalise and time_normalise:
        # norm[norm == 0] = 1
        A /= norm.reshape((num_windows, num_time_windows, len(focal), 1))
    elif span_normalise and not time_normalise:
        breakpoint()
        norm = np.sum(norm, axis=1)
        # norm[norm == 0] = 1
        A /= norm.reshape((num_windows, 1, len(focal), 1))
    elif time_normalise and not span_normalise:
        breakpoint()
        norm = np.sum(norm, axis=0)
        # norm[norm == 0] = 1
        A /= norm.reshape((1, num_time_windows, len(focal), 1))

    A[np.all(A == 0, axis=3)] = np.nan

    # Remove dimension for windows and/or time_windows if parameter is None
    if windows is None and time_windows is not None:
        A = np.nanmean(A, axis=0)
        #A = A.reshape((num_time_windows, len(focal), len(reference_sets)))
    elif time_windows is None and windows is not None:
        A = A.reshape((num_windows, len(focal), len(reference_sets)))
    elif time_windows is None and windows is None:
        A = np.nanmean(A, axis=0)
        #A = A.reshape((len(focal), len(reference_sets)))
    return A, windows_parsed, time_windows_parsed


def gnn_windows_fx(outfile, ts,
                    ref_samples,
                    target_samples,
                    groups,
                    chrom,
                    ancestry,
                    win_size,
                    time_list,
                    samples="sample_name"):
    """_summary_

    Parameters
    ----------
    outfile : _type_
        _description_
    ts : _type_
        _description_
    ref_samples : _type_
        _description_
    target_samples : _type_
        _description_
    groups : _type_
        _description_
    chrom : _type_
        _description_
    ancestry : _type_
        _description_
    win_size : _type_
        _description_
    samples : str, optional
        _description_, by default "sample_name"
    """
    assert not any(isinstance(i, list) for i in target_samples), "gnn_windows can only work on a single target list"

    if ancestry:
        # remove target pop from refs
        ref_samples = [i for i in ref_samples if i != target_samples]

    # window size
    tables = ts.dump_tables()
    sites = tables.sites.position[:]
    first_site = sites[0]
    last_site = sites[-1]
    windows = None
    if win_size:
        L = int(ts.sequence_length)
        windows = np.round(np.linspace(first_site, last_site, num=L//win_size))
    time_windows = None
    if time_list:
        T = int(ts.max_root_time)
        time_windows = [0] + time_list + [T]
        time_windows = np.array(time_windows)
    # run gnn_windows fx
    gnn_win, win_i, time_i = windowed_gnn(ts, target_samples, ref_samples, windows=windows, time_windows=time_windows)
    print(time_windows)
    print(time_i)
    ## write out results
    # get ref pops
    if any(isinstance(i, list) for i in ref_samples):
        sample_nodes = [ts.node(n) for r in ref_samples for n in r]
    else:
        sample_nodes = [ts.node(n) for n in ref_samples]
    ref_pop = [json.loads(ts.individual(n.individual).metadata)[groups] for n in sample_nodes]
    ref_pop_ids = list(dict.fromkeys(ref_pop))

    # get tar pops
    if any(isinstance(i, list) for i in target_samples):
        sample_nodes = [ts.node(n) for r in target_samples for n in r]
    else:
        sample_nodes = [ts.node(n) for n in target_samples]
    sample_names = [json.loads(ts.individual(n.individual).metadata)[samples] for n in sample_nodes]
    tar_pop = [json.loads(ts.individual(n.individual).metadata)[groups] for n in sample_nodes]
    tar_pop_ids = list(dict.fromkeys(tar_pop))

    col_names = []
    for samp in sample_names:
        totalcount = col_names.count(f"{samp}_0")
        col_names.append(f"{samp}_1" if totalcount > 1 else f"{samp}_0")

    # write out to file
    left = list(win_i)[:-1]
    right = list(win_i)[1:]
    if windows is None and time_windows is None:
        # none of --win_size OR --max_time
        # 1, samples, refs should be equivalent to gnn whole genome
        with open(f"{outfile}.gnn_windows.csv", 'w') as f:
            f.write(f'chromosome,target_pop,sample_id,{",".join(ref_pop_ids)}\n')
            for j, s in enumerate(col_names):
                gnns = ",".join(map(str, gnn_win[0][j]))
                f.write(f"{chrom},{tar_pop_ids[0]},{s},{gnns}\n")
    elif windows is not None and time_windows is None:
        # --win_size INT
        #23 trees x 34 samples x 15 refpops
        with open(f"{outfile}.gnn_windows.csv", 'w') as f:
            f.write(f'chromosome,target_pop,sample_id,left_coord,right_coord,{",".join(ref_pop_ids)}\n')
            for i in range(len(left)):
                for j, s in enumerate(col_names):
                    gnns = ",".join(map(str, gnn_win[i][j]))
                    f.write(f"{chrom},{tar_pop_ids[0]},{s},{int(left[i])},{int(right[i])},{gnns}\n")
    elif windows is None:# and time_windows is not None:
        #--max_time INT
        #5 time x 34 samples x 15 refpops; what I want to plot
        with open(f"{outfile}.gnn_windows.csv", 'w') as f:
            f.write(f'chromosome,target_pop,sample_id,time,{",".join(ref_pop_ids)}\n')
            for k, epoch in enumerate(time_windows[1:]):
                for j, samp in enumerate(col_names):
                    gnns = ",".join(map(str, gnn_win[k][j]))
                    f.write(f"{chrom},{tar_pop_ids[0]},{samp},{epoch},{gnns}\n")
    else:
    #elif windows is not None and time_windows is not None:
        # --win_size INT --max_time INT
        #23 trees x 5 times x 34 samples x 15 refpops
        with open(f"{outfile}.gnn_windows.csv", 'w') as f:
            f.write(f'chromosome,target_pop,sample_id,time,left_coord,right_coord,{",".join(ref_pop_ids)}\n')
            for i in range(len(left)):
                for k, epoch in enumerate(time_windows[1:]):
                    for j, samp in enumerate(col_names):
                        gnns = ",".join(map(str, gnn_win[i][k][j]))
                        f.write(f"{chrom},{tar_pop_ids[0]},{samp},{epoch},{int(left[i])},{int(right[i])},{gnns}\n")
        

def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tree", type=str, required=True,
                        help="tskit tree object")
    parser.add_argument("--chrom", type=str, required=True,
                        help="name for chromosome in file")
    parser.add_argument("--groups", type=str, required=True,
                        help="How to cluster individuals")
    parser.add_argument("--outfile", type=str, default=None,
                        help="name for output file")
    parser.add_argument("--tar", type=str, default=None,
                        help="target nodes by group name, if None will use all")
    parser.add_argument("--ref", type=str, default=None,
                        help="reference nodes by group name, if None will use all")
    parser.add_argument("--ancestry", action="store_true",
                        help="remove target pop from refs, default is to include it")
    parser.add_argument("--gnn_windows", action="store_true",
                        help="run gnn in windows mode")
    parser.add_argument("--win_size", type=int, default=None,
                        help="size of default spatial window in gnn_windows mode")
    parser.add_argument("--times", nargs="+", type=int, default=None,
                        help="times to estimate gnn over")
    parser.add_argument("--down_sample", type=int, default=None,
                        help="down sample all groups to this size")
    parser.add_argument("--regions", type=str, default=None,
                        help="regions to include in analyses")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    # required
    tree = args.tree
    chrom = args.chrom
    groups = args.groups
    outfile = args.outfile
    if outfile is None:
        outfile = path.split(tree)[-1]
    # set vals, default all
    ref_set = args.ref
    tar_set = args.tar
    # flags as True
    gnn_win = args.gnn_windows
    ancestry = args.ancestry
    # options default set
    win_size = args.win_size
    time_list = args.times
    dwn_sample = args.down_sample
    regions = args.regions
    # =========================================================================
    #  Loading and Checks
    # =========================================================================
    # load tree
    # load tree sequence
    tic = time.perf_counter()
    ts = tskit.load(tree)
    toc = time.perf_counter()
    print(f"trees loaded in {toc - tic:0.4f} seconds")
    # get group data
    group_dt = defaultdict(list)
    for n in ts.samples():
        individual_data = ts.individual(ts.node(n).individual)
        gt = json.loads(individual_data.metadata)[groups]
        group_dt[gt].append(n)
    # down sample
    if dwn_sample:
        for gt, haps in group_dt.items():
            if len(haps) > dwn_sample:
                subset = list(np.random.choice(haps, dwn_sample, replace=False))
                group_dt[gt] = sorted(subset)
    # set target population
    tar_nodes = list(group_dt.values()) if tar_set is None else group_dt[tar_set]
    # set reference for comparison
    ref_nodes = list(group_dt.values()) if ref_set is None else group_dt[ref_set]
    if regions:
        regions_list = []
        with open(regions) as r:
            for line in r:
                rchrom, rstart, rend = line.split()
                if rchrom == chrom:
                    regions_list.append((int(rstart)+1, int(rend)))
        print(f"keeping intevals {regions_list}")
        regions_arr = np.array(regions_list)
        tic = time.perf_counter()
        ts = ts.keep_intervals(regions_arr)
        toc = time.perf_counter()
        print(f"trees keep intervals {toc - tic:0.4f} seconds")     
    # =========================================================================
    #  Main executions
    # =========================================================================
    # could also just pass a time windows list [0, 1000, 10000, 100000]
    if gnn_win:
        gnn_windows_fx(outfile,
                        ts,
                        ref_nodes,
                        tar_nodes,
                        groups,
                        chrom,
                        ancestry,
                        win_size,
                        time_list)
    else:
        gnn_fx(outfile, ts, ref_nodes, tar_nodes, groups, chrom, ancestry)


if __name__ == "__main__":
    main()
