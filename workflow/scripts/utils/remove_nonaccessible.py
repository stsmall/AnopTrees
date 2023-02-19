import operator
import numpy as np
import pandas as pd
import sys
import tskit

def merged(intervals, *, closed: bool):
    """
    Merge overlapping and adjacent intervals.
    :param intervals: An iterable of (start, end) coordinates.
    :param bool closed: If True, [start, end] coordinates are closed,
        so [1, 2] and [3, 4] are adjacent intervals and will be merged.
        If False, [start, end) coordinates are half-open,
        so [1, 2) and [3, 4) are not adjacent and will not be merged.
    """

    def iter_merged(intervals, *, closed: bool):
        """
        Generate tuples of (start, end) coordinates for merged intervals.
        """
        intervals = sorted(intervals, key=operator.itemgetter(0))
        start, end = intervals[0]
        for a, b in intervals[1:]:
            assert a <= b
            if a > end + closed:
                # No intersection with the current interval.
                yield start, end
                start, end = a, b
            else:
                # Intersects, or is contiguous with, the current interval.
                end = max(end, b)
        yield start, end

    return list(iter_merged(intervals, closed=closed))

# load args
mask_file = sys.argv[1]
chrom = sys.argv[2]
tree = sys.argv[3]

# import mask bed to intervals
mask_table = pd.read_csv(mask_file, sep="\t", header=None)
mask = mask_table[mask_table[0] == chrom]
intervals = np.array(mask.values[:, 1:3])
mask_intervals = np.array(merged(intervals, closed=False))
mask_intervals = mask_intervals[:-2] + 1  # not bed coords
# read in tree and delete_intervals
ts = tskit.load(tree)
seq_len = ts.get_sequence_length()
for i in range(1, len(mask_intervals)):
    s, e = mask_intervals[-i]
    if s >= seq_len:
        continue
    elif e >= seq_len:
        mask_intervals[-i][1] = int(seq_len)
        break
    else:
        if i != 1:
            mask_intervals = mask_intervals[:-i]
            break
        break
ts_mask = ts.delete_intervals(mask_intervals)
ts_mask.dump(f"{tree}-accessible")
# some stats
total_length = 0
total_length += np.sum(mask_intervals[:, 1] - mask_intervals[:, 0])
print(f" total accessible sites for {chrom} = {total_length}")
