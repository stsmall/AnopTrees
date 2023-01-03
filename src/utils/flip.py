import sys
import gzip

file1 = sys.argv[1]
file2 = sys.argv[2]

with open(file1, 'r') as fp:
    flip_dt = {}
    for line in fp:
        x = line.strip().split(";")
        flip_dt[x[0]] = x[1]
        
# TODO: add cond that it was Relate
with gzip.open(f"{file2}-flip", 'wt') as f:
    with gzip.open(file2, 'rt') as vcf:
        for line in vcf:
            line = line
            if line.startswith("#"):
                f.write(line)
            else:
                x = line.split()
                pos = x[1]
                try:
                    flip_dt[pos]
                    ref = x[3]
                    alt = x[4]
                    fields = x[7].split(";")
                    old_aa = fields[0].split("=")[1]
                    if old_aa == ref:
                        fields[0] = f"AA={alt}"
                    elif old_aa == alt:
                        fields[0] = f"AA={ref}"
                    x[7] = ";".join(fields)
                    f.write("{}\n".format("\t".join(x)))
                except KeyError:
                    f.write(line)