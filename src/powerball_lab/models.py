import math
import numpy as np
import pandas as pd

WHITE_MAX = 69
PB_MAX = 26

def empirical_weights(df, half_life_draws=None):
    idx_w = range(1, WHITE_MAX+1)
    weights_white = pd.Series(0.0, index=idx_w)
    decay = None
    if half_life_draws:
        lam = math.log(2)/half_life_draws
        decay = np.exp(-lam * np.arange(len(df))[::-1])
    for i, row in df.reset_index(drop=True).iterrows():
        wset = [row["w1"],row["w2"],row["w3"],row["w4"],row["w5"]]
        factor = decay[i] if decay is not None else 1.0
        for n in wset:
            weights_white[n]+=factor
    weights_pb = pd.Series(0.0, index=range(1, PB_MAX+1))
    for i, row in df.reset_index(drop=True).iterrows():
        factor = decay[i] if decay is not None else 1.0
        pb_val = int(row["pb"])
        if pb_val < 1 or pb_val > PB_MAX:  # guard
            continue
        weights_pb[pb_val]+=factor
    weights_white = weights_white.replace(0,1e-6)
    weights_pb = weights_pb.replace(0,1e-6)
    return weights_white, weights_pb

def sample_ticket(weights_white, weights_pb, rng):
    whites = list(range(1, WHITE_MAX+1))
    ww = weights_white.values.astype(float).copy()
    chosen = []
    for _ in range(5):
        p = ww / ww.sum()
        pick = rng.choice(whites, p=p)
        chosen.append(int(pick))
        idx = whites.index(int(pick))
        whites.pop(idx); ww = np.delete(ww, idx)
    chosen = sorted(chosen)
    pballs = np.arange(1, PB_MAX+1)
    pp = (weights_pb.values.astype(float))
    pp = pp / pp.sum()
    pb = int(rng.choice(pballs, p=pp))
    return chosen, pb
