# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:21:01 2021

@author: Scott T. Small

This module calculates the tmrca half and cross coalescent 10 statistics
from the paper of Hejase et al. 2020
(https://www.biorxiv.org/content/10.1101/2020.03.07.977694v2.full).

Example
-------
Examples using relate trees. Note that relate trees need to be converted to
tree sequencing format using --mode ConvertToTreeSequence.

$ python tree_stats.py --trees FOO.relate.trees --outfile FOO.1_2
    --groups country --pop_ids Mayotte Mali --down_sample 50 --fx tmrca_half

$ python tree_stats.py --trees FOO.relate.trees --outfile FOO.1_2
    --groups country --pop_file FOO.node.txt --down_sample 50 --fx tmrca_half

Notes
-----
The node file input was needed since Relate didnt transfer over information
about the populations, meaning the populations were not stored in the tree seq.
The node file has a single line of comma delimited integers denoting the leaf
id associated with the desired population or group. The pop_ids need to be in
the same order as the node file to ensure proper naming. The first entry should
be the population name.

$ > head FOO.node.txt
Mali,1,2,3,4,5,6,7,8
Mayotte,11,12,13,14,15


"""
import argparse
from itertools import combinations
from os import path
import sys
import time
from collections import defaultdict
import json
from p_tqdm import p_map
#import multiprocessing as mp
#from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import tskit
print(f"using tskit version {tskit.__version__}, tested in version '0.3.5'")


def calc_mrcah(p12):
    """Calculate the tmraca half as defined in Hejase 2020.

    "...test on the time to the most recent common ancestor of half the haploid
    samples from a given species (TMRCAH). Requiring only half the samples
    allows us to consider partial sweeps and provides robustness to the
    inherent uncertainty in the inferred local trees."

    Parameters
    ----------
    ts : Object
        object of type tskit tree seqeunce.
    p_nodes : List
        List of node ids as integers, [[0,1,2],[4,5,6]]

    Returns
    -------
    mid : List
        the center of the interval in base pairs position
    tmrcah_rel : List
        the tmrca of half the population
    time_rel : List
        Full TMRCA of that population.
    time_rel2 : List
        Age of the youngest subtree that contains at least half of the samples.

    """
    pop_name, p_nodes = p12
    int1 = []
    int2 = []
    tmrcah_rel = []
    time_rel = []
    time_rel2 = []
    tmrca = []
    mrca_t = ""
    p_half = len(p_nodes) / 2  # half the pop
    sample_half = ts.num_samples / 2  # half of all inds in tree
    iter1 = ts.trees(tracked_samples=p_nodes, sample_lists=True)
    for t in tqdm(iter1, total=ts.num_trees):
        if t.num_edges == 0:
            continue
        int1.append(t.interval[0])
        int2.append(t.interval[1])
        tmrcah = None
        time_r = None
        for u in t.nodes(order='timeasc'):
            if t.num_tracked_samples(u) >= p_half and tmrcah is None:
                tmrcah = t.time(u)  # mrca_half
            if t.num_samples(u) >= sample_half and time_r is None:
                time_r = t.time(u)  # total_sample mrca_half
            if tmrcah is not None and time_r is not None:
                mrca = t.mrca(*p_nodes)  # mrca
                mrca_t = t.time(mrca)
                break

        tmrcah_rel.append(tmrcah)  # mrca_half
        time_rel.append(mrca_t)  # mrca
        time_rel2.append(time_r)  # sample_half mrca
        tmrca.append(t.time(t.root))  # tree height

    return pop_name, int1, int2, tmrcah_rel, time_rel, time_rel2, tmrca


def mrca_half(outfile, group_dt):
    """Calculats the tmrca half fx from Hejase et al 2020.

        "...test on the time to the most recent common ancestor of half the haploid
    samples from a given species (TMRCAH). Requiring only half the samples
    allows us to consider partial sweeps and provides robustness to the
    inherent uncertainty in the inferred local trees."

    Parameters
    ----------
    ts : Object
        object of type tskit tree seqeunce.
    pop_nodes : List
        population leaves as integers loaded from file.
    pop_ids : List
        id of population nodes to be written in DataFrame.
    outfile : str
        base name of DataFrame file output.

    Returns
    -------
    None.

    """
    with open(f"{outfile}.mrca_half.csv", 'w') as f:
        f.write("chromosome,population,tree_start,tree_end,mrca_h,mrca,tmrca_h,tmrca\n")
        if ncpus == 1:
            for pop, nodes in group_dt.items():
                pop_i, int1, int2, mrca_h, mrca, mrca_sh, tmrca = calc_mrcah((pop, nodes))
                for i in range(len(int1)):
                    f.write(f"{chrom},{pop},{int1[i]},{int2[i]},{mrca_h[i]},{mrca[i]},{mrca_sh[i]},{tmrca[i]}\n")
        else:
            #pool = Pool(ncpus)
            its_list = [(pop, nodes) for pop, nodes in group_dt.items()]
            results = [p_map(calc_mrcah, its_list)]
            #results = [pool.map(calc_mrcah, its_list)]
            for i in range(len(results[0])):
                pop_i, int1, int2, mrca_h, mrca, mrca_sh, tmrca = results[0][i]
                for i in range(len(int1)):
                    f.write(f"{chrom},{pop_i},{int1[i]},{int2[i]},{mrca_h[i]},{mrca[i]},{mrca_sh[i]},{tmrca[i]}\n")
            #pool.close()
    return f"{outfile}.mrca_half.csv"

def calc_pwmrca(p12):
    pop1, npop1 = p12[0]
    pop2, npop2 = p12[1]
    p_nodes = npop1 + npop2
    int1 = []
    int2 = []
    pw_mrca = []
    mrca_1 = []
    mrca_2 = []
    tmrca = []
    skipped_trees = []
    iter1 = ts.trees(tracked_samples=p_nodes, sample_lists=True)
    for t in tqdm(iter1, total=ts.num_trees):
        if t.num_edges == 0:
            skipped_trees.append(t.index)
            continue
        int1.append(t.interval[0])
        int2.append(t.interval[1])
        tmrca.append(t.time(t.root))
        m1 = t.mrca(*npop1)
        m2 = t.mrca(*npop2)
        mrca_1.append(t.time(m1))
        mrca_2.append(t.time(m2))
        #pw_mrca.append(t.time(t.mrca(*p_nodes)))
        pw_mrca.append(t.tmrca(m1, m2))
    div = ts.divergence([npop1, npop2], mode="branch", windows="trees")
    mask = np.zeros(len(div), dtype=bool)
    mask[skipped_trees] = True
    div = div[~mask]
    pi_1 = ts.diversity(npop1, mode="branch", windows="trees")
    pi_1 = pi_1[~mask]
    pi_2 = ts.diversity(npop2, mode="branch", windows="trees")
    pi_2 = pi_2[~mask]
    return pop1, pop2, int1, int2, tmrca, mrca_1, mrca_2, pw_mrca, div, pi_1, pi_2

def pairwise_mrca(outfile, group_dt):
    """
    Parameters
    ----------
    ts : Object
        object of type tskit tree seqeunce.
    outfile : str
        base name of DataFrame file output.

    Returns
    -------
    None.

    """
    with open(f"{outfile}.pw_mrca.csv", 'w') as f:
        f.write("chromosome,pop1,pop2,tree_start,tree_end,mrca_1,mrca_2,mcra_12,pw_mrca,pi_1,pi_2,tmrca\n")
        if ncpus == 1:
            for p1, p2 in combinations(group_dt.items(), 2):
                pop1, pop2, int1, int2, tmrca, mrca_1, mrca_2, mrca_12, pw_mrca, p1, p2 = calc_pwmrca((p1,p2))
                for i, t in enumerate(tmrca):
                    f.write(f"{chrom},{pop1},{pop2},{int1[i]},{int2[i]},{mrca_1[i]},{mrca_2[i]},{mrca_12[i]},{pw_mrca[i]/2},{p1[i]/2},{p2[i]/2},{t}\n")
        else:
            #pool = Pool(ncpus)
            its_list = [(p1, p2) for p1, p2 in combinations(group_dt.items(), 2)]
            results = [p_map(calc_pwmrca, its_list)]
            #results = [pool.map(calc_pwmrca, its_list)]
            for i in range(len(results[0])):
                pop1_i, pop2_i, int1, int2, tmrca, mrca_1, mrca_2, mrca_12, pw_mrca, p1, p2 = results[0][i]
                for i, t in enumerate(tmrca):
                    f.write(f"{chrom},{pop1_i},{pop2_i},{int1[i]},{int2[i]},{mrca_1[i]},{mrca_2[i]},{mrca_12[i]},{pw_mrca[i]/2},{p1[i]/2},{p2[i]/2},{t}\n")
            #pool.close()
    return f"{outfile}.pw_mrca.csv"

def calc_cross_coalescent(p12, ccN_events=10):
    """Calculate the cross coalescent of two lists of nodes.
    Parameters
    ----------
    tree_seq : Object
        object of type tskit tree seqeunce.
    nodes_ls : List
        List of node ids as integers, [[0, 1, 2],[4, 5, 6]]
    cc_N_events : int, optional
        the number of cross coalescent events to track. The default is 10.
    Returns
    -------
    ccN_rel : List
        the cross coalescent of the population from 1 ... N
    """
    ccN_ts = []
    int1 = []
    int2 = []
    time_rel = []
    sample_half = ts.num_samples / 2
    pop1_name, pop_1 = p12[0]
    pop2_name, pop_2 = p12[1]
    iter1 = ts.trees(tracked_samples=pop_1, sample_lists=True)
    iter2 = ts.trees(tracked_samples=pop_2, sample_lists=True)
    p1_samples = set(pop_1)
    p2_samples = set(pop_2)
    for tree1, tree2 in tqdm(zip(iter1, iter2), total=ts.num_trees):
        if tree1.num_edges == 0:
            continue
        int1.append(tree1.interval[0])
        int2.append(tree1.interval[1])
        sample_half_time = None
        ccN_tree = []
        num_cc = 0
        used_nodes = set()
        for u in tree1.nodes(order='timeasc'):
            num_pop1 = tree1.num_tracked_samples(u)
            num_pop2 = tree2.num_tracked_samples(u)
            if num_cc < ccN_events:
                if num_pop1 > 0 and num_pop2 > 0:
                    proposed_cc1 = set(tree2.samples(u)) - p2_samples - used_nodes
                    proposed_cc2 = set(tree1.samples(u)) - p1_samples - used_nodes
                    if proposed_cc1 and proposed_cc2:
                        used_nodes |= proposed_cc1 | proposed_cc2
                        simul_cc_events = min(
                            [len(proposed_cc1), len(proposed_cc2)])
                        num_cc += simul_cc_events
                        ccN_tree.extend([tree1.time(u)] * simul_cc_events)  # some nodes are the parent of >1 cc
            if tree1.num_samples(u) > sample_half:
                if sample_half_time is None:
                    sample_half_time = tree1.time(u)
                if num_cc >= ccN_events:
                    break
        time_rel.append(sample_half_time)
        if num_cc < ccN_events:
            # padding so all trees return the same length
            ccN_tree.extend(np.repeat(np.nan, (ccN_events - num_cc)))
        ccN_ts.append(ccN_tree[:ccN_events])
    fst = ts.Fst([pop_1, pop_2], windows="trees")
    return pop1_name, pop2_name, int1, int2, ccN_ts, time_rel, fst


def cross_coal(outfile, group_dt, ccN_events=10):
    """Calculate the cross coalescent 10 stat from Hejase et al 2020.

    "...For a given local tree and pair of species, we considered the 10 most
    recent cross coalescent events between the two species and normalized these
    ages, as in test 2, by the age of the youngest subtree that contains at
    least half of the total number of haploid samples."

    Parameters
    ----------
    ts : Object
        object of type tskit tree seqeunce.
    pop_nodes : List
        population leaves as integers loaded from file.
    pop_ids : List
        id of population nodes to be written in DataFrame.
    outfile : str
        base name of DataFrame file output.

    Returns
    -------
    None.

    """
    cols = ",".join([f"cc_{i+1}" for i in range(ccN_events)])
    with open(f"{outfile}.cc{ccN_events}.csv", 'w') as f:
        f.write(f"chromosome,pop1,pop2,tree_start,tree_end,fst,tmrca_h,{cols}\n")
        if ncpus == 1:
            for p1, p2 in combinations(group_dt.items(), 2):
                pop1, pop2, int1, int2, ccN, time_rel, fst = calc_cross_coalescent((p1, p2))
                for i, int_1 in enumerate(int1):
                    ccN_i = ",".join(map(str, ccN[i]))
                    f.write(f"{chrom},{pop1},{pop2},{int_1},{int2[i]},{fst[i]},{time_rel[i]},{ccN_i}\n")
        else:
            #pool = Pool(ncpus)
            its_list = [(p1, p2) for p1, p2 in combinations(group_dt.items(), 2)]
            #results = [pool.map(calc_cross_coalescent, its_list)]
            results = [p_map(calc_cross_coalescent, its_list)]
            for i in range(len(results[0])):
                pop1_i, pop2_i, int1, int2, ccN, time_rel, fst = results[0][i]
                for i, int_1 in enumerate(int1):
                    ccN_i = ",".join(map(str, ccN[i]))
                    f.write(f"{chrom},{pop1_i},{pop2_i},{int_1},{int2[i]},{fst[i]},{time_rel[i]},{ccN_i}\n")
            #pool.close()
    return f"{outfile}.cc{ccN_events}.csv"

def compress_csv(fn):
    df = pd.read_csv(fn, engine="pyarrow", dtype={'chromosome': 'str'}, sep=",")
    fn = fn.rstrip(".csv")
    df.to_feather(f"{fn}.ftr")
    #df.to_parquet(f"{fn}.parquet")


def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trees", type=str, required=True,
                        help="file containing ARGs in tskit format")
    parser.add_argument("--chrom", type=str, required=True,
                        help="name of chromosome or contig")
    parser.add_argument("--groups", type=str, default="country",
                        help="group to cluster pops ids")
    parser.add_argument("--outfile", type=str, default=None,
                        help="base name for output file")
    parser.add_argument("--pop_ids", type=str, nargs="*", action="append", default=None,
                        help="pop ids for naming columns in output dataframe")
    parser.add_argument("--pop_file", type=str,
                        help="load pop nodes from this file, one per line and"
                        "comma delimited")
    parser.add_argument("--down_sample", type=int, default=None,
                        help="down sample all groups to this size")
    parser.add_argument("--ncpus", type=int, default=None,
                        help="number cpus to use for parallel processing")
    parser.add_argument("--fx", type=str, default=None,
                        choices=("mrca_half", "cross_coal", "pwmrca"),
                        help="which fx to run ... since they both can take a long time")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    global chrom
    chrom = args.chrom
    tree_file = args.trees
    # load tree sequence
    global ts
    tic = time.perf_counter()
    ts = tskit.load(tree_file)
    toc = time.perf_counter()
    print(f"trees loaded in {toc - tic:0.4f} seconds")
    # args
    outfile = args.outfile
    if outfile is None:
        outfile = path.split(tree_file)[-1]
    fx = args.fx
    groups = args.groups
    dwn_sample = args.down_sample
    # get group data
    group_dt = defaultdict(list)
    if args.pop_file:
        with open(args.pop_file) as f:
            for line in f:
                lin = line.split(",")
                assert len(lin) > 1, "recheck nodes file, delimiter should be ,"
                group_dt[lin[0]] = list(map(int, lin[1:]))
    else:
        # load by name
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
        if args.pop_ids:
            group_dt = {pop:group_dt[pop] for pop in args.pop_ids[0]}
    # =========================================================================
    #  Main executions
    # =========================================================================
    global ncpus
    ncpus = args.ncpus
    default_cpus = len(group_dt)
    if fx == "mrca_half":
        if ncpus is None:
            ncpus = default_cpus
        fn = mrca_half(outfile, group_dt)
    elif fx == "cross_coal":
        if ncpus is None:
            ncpus = int((default_cpus * (default_cpus - 1)) / 2)
        fn = cross_coal(outfile, group_dt)
    elif fx == "pwmrca":
        if ncpus is None:
            ncpus = int((default_cpus * (default_cpus - 1)) / 2)
        fn = pairwise_mrca(outfile, group_dt)
    else:
        print("fx not recognized")
        sys.exit()
    
    compress_csv(fn)

if __name__ == "__main__":
    main()
