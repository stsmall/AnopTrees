# -*-coding:utf-8 -*-
"""
@File    :  smooth_treestats.py
@Time    :  2023/02/12 11:39:02
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2023 Scott T. Small
@Desc    :  None
@Notes   :  None
@Usage   :  python smooth_treestats.py
"""
import sys
import argparse
import pandas as pd


def method1(df, group, stat, roll):
    # even windows size where you want to plot fewer windows
    # group = ["pop1", "pop2"]
    chroms = df["chromosome"].unique()
    c = chroms.pop()
    df[f"avg_{stat}"] = df.query(f"chromosome=='{c}'").groupby(f"{group}")[f"{stat}"].rolling(roll, min_periods=1, center=True).mean().reset_index(0, drop=True)
    for c in chroms:
        df.loc[(df.chromosome == c), f"avg_{stat}"] = df.query(f"chromosome=='{c}'").groupby(f"{group}")[f"avg_{stat}"].rolling(roll, min_periods=1).mean().reset_index(0, drop=True)
    return df

def method2(df, group, win_size):
    # uneven window sizes, e.g., tree breakpoints
    df["avg_win"] = df["mid"] // win_size  # mid points
    ddf = df.groupby(group, as_index=False).mean()
    ddf["mid"] = ddf["avg_win"] * win_size
    return ddf


def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", type=str, required=True,
                        help="dataframe")
    parser.add_argument("--win_size", type=int, default=100_000,
                        help="size of smoothing window")
    parser.add_argument("--group", nargs="+", type=str, required=True,
                        help="groupy string")
    return parser.parse_args(args_in)


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])

    file = args.file
    win_size = args.win_size
    group = args.group
    # [pop1, pop2, chromosome, avg_win]  # cross_coal, pwmrca
    # [pop1, chromosome, avg_win]  # mrca_half
    compress_file = False
    if file.endswith(".gz"):
        if file.endswith("csv.gz"):
            df = pd.read_csv(file, engine="pyarrow", compression="gzip", dtype={'chromosome': 'str'}, sep=",")
            file = file.rstrip(".csv.gz")
            compress_file = True
        elif file.endswith("parquet.gz"):
            df = pd.read_parquet(file, engine="fastparquet")
            file = file.rstrip(".parquet.gz")
        elif file.endswith('ftr.gz'):
            df = pd.read_feather(file)
            file = file.rstrip(".ftr.gz")
    else:
        if file.endswith("csv"):
            #df = pd.read_csv(file, dtype={'chromosome': 'str'}, sep=",")
            df = pd.read_csv(file, engine="pyarrow", dtype={'chromosome': 'str'}, sep=",")
            file = file.rstrip(".csv")
            compress_file = True
        elif file.endswith("parquet"):
            df = pd.read_parquet(file, engine="fastparquet")
            file = file.rstrip(".parquet")
        elif file.endswith('ftr'):
            df = pd.read_feather(file)
            file = file.rstrip(".ftr")

    # clean
    df = df.dropna()
    # add mid
    df["mid"] = ((df["tree_end"] - df["tree_start"]) / 2) + df["tree_start"]
    ddf = method2(df, group, win_size)
    ddf.to_csv(f"{file}.{win_size}.csv", sep=",", index=False)
    if compress_file:
        df.to_parquet(f"{file}.parquet")


if __name__ == "__main__":
    main()
