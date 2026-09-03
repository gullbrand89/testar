def to_tokens(levels, lengths, order_type, length_type, jitter):
    # 1. Nivådefinitioner: sorterade efter värde, döpta A, B, C...
    uniq = sorted(set(levels))
    name = {v: chr(65 + i) for i, v in enumerate(uniq)}
    t = ["LEVELS", f"N{len(uniq)}"]
    for v in uniq:
        t += [name[v], f"N{bin_of(v)}", f"N{jitter[v]}"]

    # 2. Ordning
    if order_type == "fixed":
        seq = [name[v] for v in levels]
        rot = min(range(len(seq)), key=lambda i: seq[i:] + seq[:i])  # kanonisk rotation
        seq = seq[rot:] + seq[:rot]
        t += ["ORDER", "FIXED"] + seq
    elif order_type == "permute":
        t += ["ORDER", "PERMUTE"]
    else:
        t += ["ORDER", "RANDOM"] + transition_tokens(levels, name)  # räkna övergångar

    # 3. Längder
    if length_type == "inf":      t += ["DWELL", "INF"]
    elif length_type == "single": t += ["DWELL", "SINGLE"]
    elif length_type == "fixed":
        L = lengths if order_type != "fixed" else lengths[rot:] + lengths[:rot]  # samma rotation
        t += ["DWELL", "FIXED"] + [f"N{x}" for x in L]
    else:
        t += ["DWELL", length_type.upper()] + [f"N{x}" for x in sorted(lengths)]
    return t + ["END"]
    
    
    PRI_MIN, PRI_MAX = 100.0, 1000.0   # emitterrymden
N_BINS = 256

def bin_of(pri):
    b = int((pri - PRI_MIN) / (PRI_MAX - PRI_MIN) * (N_BINS - 1))
    return max(0, min(N_BINS - 1, b))

def pri_of_bin(b):
    return PRI_MIN + (b + 0.5) / (N_BINS - 1) * (PRI_MAX - PRI_MIN)   # bin-def canon_rot(seq):
    seq = list(seq)
    i = min(range(len(seq)), key=lambda k: seq[k:] + seq[:k])
    return i

def to_tokens(levels, lengths, order_fixed, length_fixed, jitter):
    levels, lengths = list(levels), [int(x) for x in lengths]
    uniq = sorted(set(levels))
    name = {v: chr(65 + i) for i, v in enumerate(uniq)}
    t = ["LEVELS", f"N{len(uniq)}"]
    for v in uniq:
        t += [name[v], f"N{bin_of(v)}", f"N{jitter.get(v, 0)}"]

    seq = [name[v] for v in levels]
    if order_fixed:
        r = canon_rot(seq)
        t += ["ORDER", "FIXED"] + seq[r:] + seq[:r]
    else:
        t += ["ORDER", "RANDOM"]

    if length_fixed:
        if order_fixed:
            L = lengths[r:] + lengths[:r]          # samma rotation som nivåerna
        else:
            r2 = canon_rot(lengths)
            L = lengths[r2:] + lengths[:r2]        # egen rotation
        t += ["DWELL", "FIXED"] + [f"N{x}" for x in L]
    else:
        t += ["DWELL", "RANDOM"] + [f"N{x}" for x in sorted(lengths)]
    return t + ["END"]
    
    import numpy as np

lv_all, gaps, len_all = [], [], []
for _ in range(10000):
    levels, lengths, *_ = my_generator()
    levels = np.asarray(levels, dtype=float)
    lv_all.extend(levels)
    len_all.extend(lengths)
    u = np.unique(levels)
    if len(u) > 1:
        gaps.append(np.min(np.diff(u)))       # minsta avstånd mellan två nivåer

lv_all, gaps, len_all = map(np.asarray, (lv_all, gaps, len_all))
print("PRI     min/max:", lv_all.min(), lv_all.max())
print("PRI     1%/99%: ", np.percentile(lv_all, [1, 99]))
print("min gap min/1%: ", gaps.min(), np.percentile(gaps, 1))
print("längd   min/max:", len_all.min(), len_all.max())

import numpy as np
import matplotlib.pyplot as plt

def roll_out(levels, lengths, order_fixed, length_fixed, n_pulses, jitter_us, rng):
    """cykel -> lista av sanna PRI-värden (µs)"""
    levels, lengths = list(levels), list(lengths)
    pri, i = [], 0
    while len(pri) < n_pulses:
        lvl = levels[i % len(levels)] if order_fixed else rng.choice(levels)
        L   = lengths[i % len(lengths)] if length_fixed else rng.choice(lengths)
        j   = jitter_us.get(lvl, 0.0)
        for _ in range(int(L)):
            pri.append(lvl + rng.uniform(-j, j))
            if len(pri) == n_pulses: break
        i += 1
    return np.array(pri)

def corrupt(pri, rng, p_drop=0.1, p_spur=0.02, meas_noise_us=0.5):
    """sanna PRI -> observerade PRI via TOA"""
    toa = np.concatenate([[0.0], np.cumsum(pri)])
    keep = rng.random(len(toa)) > p_drop                # bortfall
    keep[0] = True
    toa = toa[keep]
    n_spur = rng.binomial(len(toa), p_spur)              # spuriösa pulser
    spur = rng.uniform(toa[0], toa[-1], n_spur)
    toa = np.sort(np.concatenate([toa, spur]))
    toa = toa + rng.normal(0, meas_noise_us, len(toa))    # mätbrus
    toa = np.sort(toa)
    return toa, np.diff(toa)                              # observerad TOA och PRI

def make_input(pri_obs):
    bins = np.array([in_bin_of(p) for p in pri_obs])
    cont = np.array([in_cont_of(p) for p in pri_obs])
    return bins, cont

def show(pri_true, pri_obs, tokens):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    ax[0].plot(pri_true, ".-"); ax[0].set_title("sann PRI")
    ax[1].plot(pri_obs, ".-", color="C1"); ax[1].set_title("observerad PRI (efter bortfall)")
    ax[1].axhline(PRI_MAX, ls="--", color="gray")
    fig.suptitle(" ".join(tokens), fontsize=8)
    plt.tight_layout(); plt.show()

    
    

