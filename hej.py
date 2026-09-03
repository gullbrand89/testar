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
    return PRI_MIN + (b + 0.5) / (N_BINS - 1) * (PRI_MAX - PRI_MIN)   # bin-mitten

    
    
    
