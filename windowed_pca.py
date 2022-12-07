# -*-coding:utf-8 -*-
"""
@File    :  windowed_pca.py
@Time    :  2022/12/05 11:38:42
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  Sliding window PCA of genomic data
@Notes   :  modified from github.com/MoritzBlumer/inversion_scripts
@Usage   :  python windowed_pca.py zarr/ meta.tsv outfile --chrom 2L --chrom_start 10_000_000 
            --chrom_end 20_000_000 --win_size 1e6 --win_step 1e5 --group country 
            --group_id 'Burkina Faso' --color_by 2L --var_thresh 9 --mean_thresh 3

"""
import allel
import zarr
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px


def check_order(panel, samples):
    """_summary_

    Parameters
    ----------
    panel : _type_
        _description_
    samples : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
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
    """_summary_

    Parameters
    ----------
    file_path : _type_
        _description_
    is_X : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    dtypes = {'sample_id':'object', 'country':'object', 'location':'object', 'year':'int64', 'month':'int64',
            'latitude':'float64', 'longitude':'float64', 'aim_species':'object', 'sex_call':'object',
            "2La":"object", "2Rb":"object", "2Rc":"object", "2Rd":"object", "2Rj":"object", "2Ru":"object"}
    cols = ['sample_id', 'country', 'location', 'year', 'month', 'latitude', 'longitude', 'aim_species', "sex_call",
            "2La", "2Rb", "2Rc", "2Rd", "2Rj", "2Ru"]
    panel = pd.read_csv(file_path, sep=',', usecols=cols, dtype=dtypes)

    if is_X:
        panel = panel[panel["sex_call"] == "F"]
    panel.groupby(by=(['country', "location"])).count()
    return panel

def load_phased(chrom, meta_path, zarr_path):
    """_summary_

    Parameters
    ----------
    chrom : _type_
        _description_
    meta_path : _type_
        _description_
    zarr_path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    callset = zarr.open_group(zarr_path, mode='r')
    panel = pd.DataFrame()
    order_ls = []
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
    return gt, pos, panel

def prepare_data(chrom, gt_path, metadata_path, group=None, group_id=None, maf=0.05):
    """_summary_

    Parameters
    ----------
    gt_path : _type_
        _description_
    metadata_path : _type_
        _description_
    group : _type_, optional
        _description_, by default None
    group_id : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    gt, pos, metadata_df = load_phased(chrom, metadata_path, gt_path)
    
    # subset input 
    if group and group_id:
        metadata_df = metadata_df.loc[metadata_df[group].isin(group_id.split(','))]
        pix = metadata_df["callset_index"].values
        gt = gt.take(pix, axis=1)
        
    ac = gt.count_alleles()
    fq = ac.to_frequencies()
    flt = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 1) & (fq[:, 1] >= maf)
    pos_flt = pos[flt]
    gt_flt = gt.compress(flt, axis=0)

    return gt_flt, pos_flt, metadata_df

def compile_window_arrays(chrom_start, chrom_end, window_size, window_step):
    """_summary_

    Parameters
    ----------
    chrom_start : _type_
        _description_
    chrom_end : _type_
        _description_
    window_size : _type_
        _description_
    window_step : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    window_start_arr = np.array(range(chrom_start, chrom_end-window_size, window_step), dtype=int)
    windows_stop_arr = np.array(window_start_arr + window_size, dtype=int)
    windows_mid_arr = np.array(window_start_arr + (0.5 * window_size), dtype=int)
    return window_start_arr, windows_stop_arr, windows_mid_arr

def window_pca(gt_arr, pos_arr, window_start, window_stop, phased, min_var_per_window=50):
    """_summary_

    Parameters
    ----------
    gt_arr : _type_
        _description_
    pos_arr : _type_
        _description_
    window_start : _type_
        _description_
    window_stop : _type_
        _description_
    haploid : _type_
        _description_
    min_var_per_window : int, optional
        _description_, by default 50

    Returns
    -------
    _type_
        _description_
    """
    window_idx_arr = np.where((pos_arr >= window_start) & (pos_arr < window_stop))[0]
    window_gt_arr = gt_arr.take(window_idx_arr, axis=0)
    
    if len(window_idx_arr) <= min_var_per_window:
        empty_array = [None] * window_gt_arr.shape[1]
        print(f"[INFO] Skipped window {window_start} - {window_stop} with {window_gt_arr.shape[0]} variants (threshold is {min_var_per_window} variants per window)", file=sys.stderr, flush=True)
        return empty_array, empty_array, None, None, window_gt_arr.shape[0]
    elif phased:
        gn = window_gt_arr.to_haplotypes()
        pca = allel.pca(gn, n_components=2, copy=True, scaler='patterson', ploidy=1)
        return pca[0][: , 0], pca[0][: , 1], pca[1].explained_variance_ratio_[0]*100, pca[1].explained_variance_ratio_[1]*100, window_gt_arr.shape[0]
    else:
        gn = window_gt_arr.to_n_alt()
        pca = allel.pca(gn, n_components=2, copy=True, scaler='patterson', ploidy=2)
        return pca[0][: , 0], pca[0][: , 1], pca[1].explained_variance_ratio_[0]*100, pca[1].explained_variance_ratio_[1]*100, window_gt_arr.shape[0]

def do_pca(gt_arr, pos_arr, window_start_arr, windows_mid_arr, windows_stop_arr, metadata_df, phased, win_size):
    """_summary_

    Parameters
    ----------
    gt_arr : _type_
        _description_
    pos_arr : _type_
        _description_
    window_start_arr : _type_
        _description_
    windows_mid_arr : _type_
        _description_
    windows_stop_arr : _type_
        _description_
    metadata_df : _type_
        _description_
    win_size : _type_
        _description_
    haploid : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # initiate empty data frames for PC1 and PC2
    pc_1_df = pd.DataFrame(columns=windows_mid_arr, index = list(metadata_df['sample_id']))
    pc_2_df = pd.DataFrame(columns=windows_mid_arr, index = list(metadata_df['sample_id']))
    pc_1_pct_explained_lst = []
    pc_2_pct_explained_lst = []
    n_variants_lst = []

    # iterrate and conduct PCAs
    for window_start, window_mid, window_stop in tqdm(zip(window_start_arr, windows_mid_arr, windows_stop_arr), total=len(window_start_arr)):
        pc_1_df[window_mid], pc_2_df[window_mid], pc_1_pct_explained, pc_2_pct_explained, n_variants = window_pca(gt_arr, pos_arr, window_start, window_stop, phased)
        pc_1_pct_explained_lst.append(pc_1_pct_explained)
        pc_2_pct_explained_lst.append(pc_2_pct_explained)
        n_variants_lst.append(n_variants)

    pc_1_pct_explained_arr = np.array(pc_1_pct_explained_lst, dtype=float)
    pc_2_pct_explained_arr = np.array(pc_2_pct_explained_lst, dtype=float)

    # compile a data frame of additional info (% variance explained for PC_1 and PC_1, the % of sites per window)
    additional_info_df = pd.DataFrame(np.array([windows_mid_arr, pc_1_pct_explained_arr, pc_2_pct_explained_arr, np.array(n_variants_lst)/win_size]).transpose(), columns=['Genomic_Position', 'perc explained PC 1', 'perc explained PC 2', 'perc included sites'], dtype=float)

    return pc_1_df, pc_2_df, additional_info_df

def calibrate_annotate(pc_df, metadata_df, pc, var_threshold, mean_threshold):
    """_summary_
    - take a pc_df and adjust window orientation using a selection of a few samples 
        with high absolute values and small variability
    - then annotate the df with metadata.
    - hack: if setting var_threshold=False and mean_threshold to a set of primary_ids 
    (e.g. "cichlid7050764,cichlid7050776,cichlid7050768"), 
    those will be used as guide samples for polarizaion
    Parameters
    ----------
    pc_df : _type_
        _description_
    metadata_df : _type_
        _description_
    pc : _type_
        _description_
    var_threshold : _type_
        _description_
    mean_threshold : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # select the 9 samples with the least variance, 
    # and from those the 3 with the highest absolute value accross 
    # all windows as guide samples to calibrate the orientation of all windows

    if var_threshold:  # use random samples
        guide_samples = list(pc_df.dropna(axis=1).abs().var(axis=1).sort_values(ascending=True).index[:var_threshold])
        guide_samples_df = pc_df.loc[guide_samples]
        guide_samples = list(guide_samples_df.dropna(axis=1).abs().sum(axis=1).sort_values(ascending=False).index[:mean_threshold])
    else:  # use set samples
        guide_samples = mean_threshold.split(',')
        guide_samples_df = pc_df.loc[guide_samples]
    guide_samples_df = guide_samples_df.loc[guide_samples]

    # for each guide sample, determine whether the positive or negative absolute value of each window is closer 
    # to the value in the previous window. If the negative value is closer, switch that windows orientation
    # (1 --> switch, 0 --> keep)
    
    rows_lst = []
    for row in guide_samples_df.iterrows():
        row = list(row[1])
        last_window = row[0] if row[0] is not None else 0 
        # only if the first window is None, last_window can be None,
        # in that case set it to 0 to enable below numerical comparisons
        out = [0]
        for window in row[1:]:
            if window is None:
                out.append(0)
                continue
            elif abs(window - last_window) > abs(window - (last_window*-1)):
                out.append(1)
                last_window = (window*-1)
            else:
                out.append(-1)
                last_window = window
        rows_lst.append(out)

    # sum up values from each row and save to switch_lst
    rows_arr = np.array(rows_lst, dtype=int).transpose()
    switch_lst = list(rows_arr.sum(axis=1))

    # switch individual windows according to switch_lst (switch if value is negative)
    for idx, val in zip(list(pc_df.columns), switch_lst):
        if val < 0:
            pc_df[idx] = pc_df[idx]*-1

    # switch Y axis if largest absolute value is negative
    if abs(pc_df.to_numpy(na_value=0).min()) > abs(pc_df.to_numpy(na_value=0).max()):
        pc_df = pc_df * -1

    # annotate with metadata
    for column_name in metadata_df.columns:
        pc_df[column_name] = list(metadata_df[column_name])

    # replace numpy NaN with 'NA' for plotting (hover_data display)
    pc_df = pc_df.replace(np.nan, 'NA')

    # convert to long format for plotting
    pc_df = pd.melt(pc_df, id_vars=metadata_df.columns, var_name='window_mid', value_name=pc)

    return pc_df

def plot_pc(pc_df, pc, color_taxon, chrom, chrom_start, chrom_end):
    """_summary_

    Parameters
    ----------
    pc_df : _type_
        _description_
    pc : _type_
        _description_
    color_taxon : _type_
        _description_
    chrom : _type_
        _description_
    chrom_len : _type_
        _description_
    window_size : _type_
        _description_
    window_step : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    chrom_len = chrom_end - chrom_start
    fig = px.line(pc_df, x='window_mid', y=pc, line_group='sample_id', color=color_taxon, hover_name='sample_id', 
                    hover_data=[x for x in list(pc_df.columns) if x not in ['window_mid', pc]],
                    width=chrom_len/20000, height=500,
                    title=f"<b>Windowed PCA of {chrom} </b><br> ({chrom_start} - {chrom_end})", 
                    labels = dict(pc1 = '<b>PC 1<b>', pc2 = '<b>PC 2<b>', window_mid = '<b>Genomic position<b>'))

    fig.update_layout(template='simple_white', font_family='Arial', font_color='black',
                    xaxis=dict(ticks='outside', mirror=True, showline=True),
                    yaxis=dict(ticks='outside', mirror=True, showline=True),
                    legend={'traceorder':'normal'}, 
                    title={'xanchor': 'center', 'y': 0.9, 'x': 0.45})

    fig.update_traces(line=dict(width=0.5))

    return fig

def plot_additional_info(additional_info_df, chrom, chrom_start, chrom_end):
    """_summary_

    Parameters
    ----------
    additional_info_df : _type_
        _description_
    chrom : _type_
        _description_
    chrom_start : _type_
        _description_
    chrom_end : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    chrom_len = chrom_end - chrom_start
    fig = px.line(additional_info_df, x='Genomic_Position', y=['perc explained PC 1', 'perc explained PC 2', 'perc included sites'], 
                    width=chrom_len/20000, height=500,
                    title=f"<b>Explained variance and of proportion of variants for windowed PCAs of {chrom} </b><br> ({chrom_start} - {chrom_end})",
                    labels = dict(Genomic_Position = '<b>Genomic Position<b>', value = '<b>variable [%]<b>', window_mid = '<b>Genomic position<b>'))
    
    fig.update_layout(template='simple_white', font_family='Arial', font_color='black',
                    xaxis=dict(ticks='outside', mirror=True, showline=True),
                    yaxis=dict(ticks='outside', mirror=True, showline=True),
                    legend={'traceorder':'normal'}, 
                    title={'xanchor': 'center', 'y': 0.9, 'x': 0.45},
                    hovermode='x unified')
    
    fig.update_traces(line=dict(width=1.0))
    
    #fig.show()

    return fig

def save_results(additional_info_df, pc_dfs, outfile, color_by, chrom, chrom_start, chrom_end):
    """_summary_

    Parameters
    ----------
    additional_info_df : _type_
        _description_
    pc_dfs : _type_
        _description_
    outfile : _type_
        _description_
    color_by : _type_
        _description_
    chrom : _type_
        _description_
    chrom_start : _type_
        _description_
    chrom_end : _type_
        _description_
    """
    for c_taxon in color_by.split(','):
        for i, df in enumerate(pc_dfs):
            df.to_csv(f"{outfile}.pc{i+1}.tsv", sep='\t', index=False)
            pc_plot = plot_pc(df, f"pc{i+1}", c_taxon, chrom, chrom_start, chrom_end)
            pc_plot.write_html(f"{outfile}.pc{i+1}.{c_taxon}.html")
            pc_plot.write_image(f"{outfile}.pc{i+1}.{c_taxon}.pdf", engine='kaleido', scale=2.4)
    # supplementary df
    supplementary_plot = plot_additional_info(additional_info_df, chrom, chrom_start, chrom_end)
    supplementary_plot.write_html(f"{outfile}.supplementary_info.html")
    supplementary_plot.write_image(f"{outfile}.supplementary_info.pdf", engine='kaleido', scale=2.4)
    additional_info_df = additional_info_df.fillna(value='NA')
    additional_info_df.to_csv(f"{outfile}.supplementary_info.tsv", sep='\t', index=False)
    
def read_existing_data(data_file, supplemental_file):
    """_summary_

    Parameters
    ----------
    data_file : _type_
        _description_
    supplemental_file : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    print('\n[INFO] Reading in existing data\n', file=sys.stderr, flush=True)
    pc_df = pd.read_csv(data_file,  sep='\t', index_col=None)
    pc_df.fillna('NA', inplace=True)
    additional_info_df = pd.read_csv(supplemental_file,  sep='\t', index_col=None)
    additional_info_df.fillna(np.nan, inplace=True)
    return pc_df, additional_info_df

def parse_args(args_in):
    """Parse args."""
    prog = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=prog)
    parser.add_argument("gt", type=str, required=True, help="path to genotypes as zarr")
    parser.add_argument("meta", type=str, required=True, help="path to meta data, tab delim")
    parser.add_argument("outfile", type=str, required=True, help="path or name for outfiles")
    parser.add_argument("--chrom", type=str, default='all', help="name of chromosome")
    parser.add_argument("--chrom_start", type=float, default=0, help="start of chromosome")
    parser.add_argument("--chrom_end", type=float, default=None, help="end of chromosome")
    parser.add_argument("--win_size", type=float, default=1e6, help="size of window")
    parser.add_argument("--win_step", type=float, default=1e5, help="step size of window")
    parser.add_argument("--min_snp", type=int, default=50, help="min number of variant sites")
    parser.add_argument("--group", type=str, default=None, help="metadata column name to be used"
                        " to select individuals to be included in the analysis")
    parser.add_argument("--group_id", type=str, default=None, help="select a value to be filtered"
                        " for in the defined filter column")
    parser.add_argument("--color_by", type=str, default=None, help="metadata column that will serve to partition included"
                        " individuals into color groups in the output plots")
    parser.add_argument("--var_thresh", type=int, default=9, help="correct random switching along PC axes")
    parser.add_argument("--mean_thresh", type=str, help="correct random switching along PC axes")
    parser.add_argument("--phased", action="store_true", help="use as phased data")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    # =================================================================
    #  Gather args
    # =================================================================
    args = parse_args(sys.argv[1:])
    genos = args.gt
    assert os.path.exists(genos)
    meta = args.meta
    assert os.path.exists(meta)
    outfile_prefix = args.outfile.lower()
    if outfile_prefix.endswith('/') and not os.path.exists(outfile_prefix):
        os.makedirs(outfile_prefix)
    chrom = args.chrom
    chrom_start = int(args.chrom_start)
    assert chrom_start >= 0
    chrom_end = int(args.chrom_end)
    assert chrom_end > 0
    assert chrom_start < chrom_end
    win_size = int(args.win_size)
    assert win_size > 0
    win_step = int(args.win_step)
    assert win_step > 0
    min_snp = args.min_snp
    assert min_snp > 0
    color_by = args.color_by
    group = args.group
    group_id = args.group_id
    if color_by is None and group is not None:
        color_by = group
    var_thresh = args.var_thresh
    mean_thresh = args.mean_thresh
    if mean_thresh is None:
        mean_thresh = 3  # set a default
    else:
        var_thresh = False  # override default
        assert len(mean_thresh.split(",")) > 0
    phased = args.phased
    #TODO: add phased expansion of metadata_df
    # =================================================================
    #  Main executions
    # =================================================================
    print('\n[INFO] Processing data and plotting\n', file=sys.stderr)
    outfile = f"{outfile_prefix}_{chrom}"
    if not os.path.exists(f"{outfile}.pc1.tsv") and not os.path.exists(f"{outfile}.pc1.supplementary_info.tsv"):
        window_start_arr, windows_stop_arr, windows_mid_arr = compile_window_arrays(chrom_start, chrom_end, win_size, win_step)
        gt_arr, pos_arr, metadata_df = prepare_data(genos, meta, group, group_id)
        # run PCA in windows
        pc_1_df, pc_2_df, additional_info_df = do_pca(gt_arr, pos_arr, window_start_arr, windows_mid_arr, windows_stop_arr, metadata_df, phased, win_size)
        del gt_arr, pos_arr
        # calibrate and annotate
        pc_1_df = calibrate_annotate(pc_1_df, metadata_df, 'pc1', var_threshold=var_thresh, mean_threshold=mean_thresh)
        pc_2_df = calibrate_annotate(pc_2_df, metadata_df, 'pc2', var_threshold=var_thresh, mean_threshold=mean_thresh)
        # save and plot
        save_results(additional_info_df, [pc_1_df, pc_2_df], outfile, color_by, chrom, chrom_start, chrom_end)
    else:  #exists so reread info
        print('\n[INFO] Reading in existing data\n', file=sys.stderr, flush=True)
        pc_1_df, additional_info_df1 = read_existing_data(f"{outfile_prefix}_{chrom}.pc1.tsv", f"{outfile_prefix}_{chrom}.pc1.supplementary_info.tsv")
        pc_2_df, additional_info_df2 = read_existing_data(f"{outfile_prefix}_{chrom}.pc2.tsv", f"{outfile_prefix}_{chrom}.pc2.supplementary_info.tsv")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"[INFO] Done in — {time.time()-start_time} seconds — \n", file=sys.stderr)
