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
            --tar Mayotte --gnn_windows
"""

import sys
from os import path
import argparse
from collections import defaultdict
import tskit
import json
import pandas as pd
import numpy as np


def gnn_fx(outfile, ts, ref_samples, target_samples, pop_ids, groups, samples="sample_name"):
    """Run mean GNN fx.

    Parameters
    ----------
    outfile : TYPE
        DESCRIPTION.
    ts : TYPE
        DESCRIPTION.
    ref_samples : TYPE
        DESCRIPTION.
    target_samples : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # calc gnn
    gnn = ts.genealogical_nearest_neighbours(target_samples, ref_samples)
    
    # write out df
    sample_nodes = [ts.node(n) for n in ts.samples()]
    sample_ids = [n.id for n in sample_nodes]
    sample_names = [json.loads(ts.individual(n.individual).metadata)[samples] for n in sample_nodes]
    sample_pops = [json.loads(ts.population(n.population).metadata)[groups] for n in sample_nodes]
    gnn_table = pd.DataFrame(data=gnn,
                            index=[pd.Index(sample_pops, name=groups)],
                            columns=pop_ids)
    gnn_table.insert(0, samples, sample_names) 
    gnn_table.insert(1, "node_id", sample_ids)
    gnn_table.to_csv(f"{outfile}.gnn.csv")


def parse_time_windows(ts, time_windows):
    """Parse time windows.

    Parameters
    ----------
    ts : TYPE
        DESCRIPTION.
    time_windows : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if time_windows is None:
        time_windows = [0.0, ts.max_root_time]
    return np.array(time_windows)


def windowed_gnn(ts, focal, reference_sets, windows=None, time_windows=None, span_normalise=True, time_normalise=True):
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
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    """
    reference_set_map = np.zeros(ts.num_nodes, dtype=int) - 1
    for k, reference_set in enumerate(reference_sets):
        for u in reference_set:
            if reference_set_map[u] != -1:
                raise ValueError("Duplicate value in reference sets")
            reference_set_map[u] = k

    windows_parsed = ts.parse_windows(windows)
    num_windows = windows_parsed.shape[0] - 1
    time_windows_parsed = parse_time_windows(ts, time_windows)
    num_time_windows = time_windows_parsed.shape[0] - 1
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
        norm = np.sum(norm, axis=1)
        # norm[norm == 0] = 1
        A /= norm.reshape((num_windows, 1, len(focal), 1))
    elif time_normalise and not span_normalise:
        norm = np.sum(norm, axis=0)
        # norm[norm == 0] = 1
        A /= norm.reshape((1, num_time_windows, len(focal), 1))

    A[np.all(A == 0, axis=3)] = np.nan

    # Remove dimension for windows and/or time_windows if parameter is None
    if windows is None and time_windows is not None:
        A = A.reshape((num_time_windows, len(focal), len(reference_sets)))
    elif time_windows is None and windows is not None:
        A = A.reshape((num_windows, len(focal), len(reference_sets)))
    elif time_windows is None and windows is None:
        A = A.reshape((len(focal), len(reference_sets)))
    return A


def gnn_windows_fx(outfile, ts, ref_samples, target_samples, pop_ids, groups, samples="sample_name"):
    """Calculate gnn in windows.

    Parameters
    ----------
    ts : Iterator
        tskit object, iterator of trees
    ref_samples : List
        list of reference sample nodes; [[0,2,3],[9,10,11]]
    target_sample : List
        list of target samples; [0,1,2,3]
    pop_ids : List
        list of population ids; ["K", "F"]

    Returns
    -------
    None.

    """
    windows = list(ts.breakpoints())  # all trees
    gnn_win = windowed_gnn(ts, target_samples, ref_samples, windows=windows)

    # save as df
    sample_nodes = [ts.node(n) for n in target_samples]
    sample_names = [json.loads(ts.individual(n.individual).metadata)[samples] for n in sample_nodes]
    sample_names = list(dict.fromkeys(sample_names))
    col_names = [f"{n}_{i}" for n in sample_names for i in [0, 1]]
    iterables = [col_names, pop_ids]
    index = pd.MultiIndex.from_product(iterables, names=[samples, groups])
    gnn_table = pd.DataFrame(data=np.reshape(gnn_win,[len(gnn_win), np.product(gnn_win.shape[1:])]), columns=index)
    gnn_table.insert(loc=0, column="left_bp", value = list(ts.breakpoints())[:-1])
    gnn_table.insert(loc=1, column="right_bp", value = list(ts.breakpoints())[1:])
    
    gnn_table.to_csv(f"{outfile}.gnn_windows.csv")
    


def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tree", type=str, required=True,
                        help="tskit tree object")
    parser.add_argument("--tar", type=str, default=None,
                        help="target nodes")
    parser.add_argument("--ref", type=str, default=None, 
                        help="reference nodes")
    parser.add_argument("--groups", type=str, required=True, 
                        help="How to cluster individuals")
    parser.add_argument("--gnn_windows", action="store_true",
                        help="run gnn in windows mode")
    parser.add_argument("--outfile", type=str, default=None,
                        help="name for output file")
    parser.add_argument("--threads", type=int, default=1,
                        help="number threads")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    tree = args.tree
    outfile = args.outfile
    if outfile is None:
        outfile = path.split(tree)[-1]
    ref_set = args.ref
    tar_set = args.tar
    gnn_win = args.gnn_windows
    groups = args.groups
    # =========================================================================
    #  Loading and Checks
    # =========================================================================
    # load tree
    ts = tskit.load(tree)
    print("tree loaded")
    # get group data
    group_dt = defaultdict(list)
    for n in ts.samples():
        individual_data = ts.individual(ts.node(n).individual)
        gt = json.loads(individual_data.metadata)["country"]
        group_dt[gt].append(n)
    # make pop_ids list for column labels
    pop_ids = list(group_dt.keys())
    # set reference for comparison
    if ref_set is None:
        ref_nodes = list(group_dt.values())
    elif path.exists(ref_set):
        ref_nodes = []
        with open(ref_set) as f:
            for line in f:
                x = line.split(",")
                assert len(x) > 1, "recheck delimiter should be ,"
                ref_nodes.append(list(map(int, x)))
    else:
        ref_nodes = group_dt[ref_set]
    # set target population
    if tar_set is None:
        tar_nodes = [item for sublist in group_dt.values() for item in sublist]
    elif path.exists(tar_set):
        tar_nodes = []
        with open(tar_set) as f:
            for line in f:
                x = line.split(",")
                assert len(x) > 1, "recheck delimiter should be ,"
                tar_nodes.extend(list(map(int, x)))
    else:
        tar_nodes = group_dt[tar_set]
    # =========================================================================
    #  Main executions
    # =========================================================================
    if gnn_win:
        gnn_windows_fx(outfile, ts, ref_nodes, tar_nodes, pop_ids, groups)
    else:
        gnn_fx(outfile, ts, ref_nodes, tar_nodes, pop_ids, groups)


if __name__ == "__main__":
    main()
