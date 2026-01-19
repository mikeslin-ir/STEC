import json, csv
import numpy as np

with open("results/msrvtt_test1k/stec_scores.json") as f:
    data = json.load(f)

samplers = ["uniform", "random", "katna", "stacfp"]
rows = []

for s in samplers:
    S, T, R, C = [], [], [], []
    for v in data.values():
        if s in v:
            S.append(v[s]["S"])
            T.append(v[s]["T"])
            R.append(v[s]["R"])
            C.append(v[s]["STEC"])
    rows.append([s, np.mean(S), np.mean(T), np.mean(R), np.mean(C)])

with open("results/msrvtt_test1k/summary.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Sampler", "S", "T", "R", "STEC"])
    writer.writerows(rows)
