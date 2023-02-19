import sys
import numpy as np
from collections import defaultdict

file = sys.argv[1]
pop_dict = defaultdict(list)
col_list = []
tar_list = []
with open(file) as f:
    header = next(f).split(",")
    pw_idx = header.index("pw_mrca") 
    for line in f:
        x = line.split(",")
        pop_dict[f"{x[0]}_{x[1]}"].append(float(x[pw_idx]))
        tar_list.append(x[0])
        col_list.append(x[1])

cols = list(dict.fromkeys(col_list+tar_list))
with open(f"{file}-wide", 'w') as w:
    w.write(f"target_pop,{','.join(cols)}\n")
    for pop in cols:
        pw = []
        for ref in cols:
            if pop_dict[f"{pop}_{ref}"]:
                pw.append(np.nanmean(pop_dict[f"{pop}_{ref}"]))
            elif pop_dict[f"{ref}_{pop}"]:
                pw.append(np.nanmean(pop_dict[f"{ref}_{pop}"]))
            else:
                pw.append(0)
        pw_str = ",".join(map(str, pw))
        w.write(f"{pop},{pw_str}\n")