"""
Suffixarray + LCP for numpy-arrayer med godtyckliga heltalsetiketter.
Hanterar float-dtype, sparsa/stora komponent-ID och NaN.
"""

import numpy as np


# ---------------------------------------------------------------- forbehandling
def to_labels(arr, tol=1e-6):
    """
    numpy-array (int eller float) -> kompakta heltal 0..k-1 + uppslagstabell.

    Sparsa ID som [3, 17, 250, 1004] blir [0, 1, 2, 3]. Nodvandigt for
    chr-varianten och snallare mot minnet.
    """
    a = np.asarray(arr).ravel()

    if np.issubdtype(a.dtype, np.floating):
        if np.isnan(a).any():
            raise ValueError("NaN i sekvensen - fyll eller ta bort forst")
        r = np.rint(a)
        avvik = np.abs(a - r).max()
        if avvik > tol:
            raise ValueError(f"Inte heltalsvarden (max avvikelse {avvik:.3g})")
        a = r.astype(np.int64)
    else:
        a = a.astype(np.int64)

    vals, inv = np.unique(a, return_inverse=True)   # vals ar sorterad
    return inv.astype(np.int64), vals


def to_str(labels, offset=0x100):
    """Kompakta etiketter -> unicodestrang. Ett tecken = en symbol."""
    if labels.max() + offset > 0x10FFFF:
        raise ValueError("For manga unika symboler for chr-mappning")
    return "".join(map(chr, labels + offset))


# ---------------------------------------------------------------- kärnan
def suffix_array(s):
    """Fungerar for str, list och tuple - allt som jamfors lexikografiskt."""
    return sorted(range(len(s)), key=lambda i: s[i:])


def lcp_array(s, sa):
    lcp = [0] * len(sa)
    for i in range(1, len(sa)):
        a, b = sa[i - 1], sa[i]
        n = 0
        while a + n < len(s) and b + n < len(s) and s[a + n] == s[b + n]:
            n += 1
        lcp[i] = n
    return lcp


def repeats(s, min_len=3):
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)
    out = []
    for i in range(1, len(sa)):
        if lcp[i] >= min_len:
            a, b = sorted((sa[i - 1], sa[i]))
            out.append({"langd": lcp[i], "pos_a": a, "pos_b": b, "skift": b - a})
    return sorted(out, key=lambda r: -r["langd"])


# ---------------------------------------------------------------- analys
def analysera(arr, min_len=3, variant="str"):
    labels, vals = to_labels(arr)
    s = to_str(labels) if variant == "str" else labels.tolist()

    print(f"  n = {len(s)},  unika symboler = {len(vals)}")
    print(f"  original-ID : {vals[:8]}{' ...' if len(vals) > 8 else ''}")

    r = repeats(s, min_len)
    if not r:
        print(f"  Ingen repeat >= {min_len}.")
        return None

    b = r[0]
    monster = vals[labels[b["pos_a"] : b["pos_a"] + b["langd"]]]
    print(f"  Langsta repeat : {b['langd']} pulser, pos {b['pos_a']} och {b['pos_b']}")
    print(f"  Monster (ID)   : {monster}")
    print(f"  Skift/period   : {b['skift']}")

    skift = np.array([x["skift"] for x in r])
    u, c = np.unique(skift, return_counts=True)
    topp = u[np.argsort(-c)][:3]
    print(f"  Vanligaste skift: {list(zip(topp, np.sort(c)[::-1][:3]))}")
    return r


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Stora sparsa ID, float-dtype ===")
    cykel = np.array([12.0, 47.0, 1003.0, 47.0])
    analysera(np.tile(cykel, 6))

    print("\n=== Samma, med en tappad puls (fel komponent) ===")
    x = np.tile(cykel, 6).copy()
    x[11] = 9999.0
    analysera(x)

    print("\n=== Rent brus, kontroll ===")
    analysera(rng.integers(0, 40, 60).astype(float), min_len=4)
