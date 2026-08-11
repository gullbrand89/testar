"""
Aterskapa sant monster nar VARJE repetition har ett fel pa olika plats.

Tva fixar mot den naiva versionen:
  1) period via vikningspoang, inte via skiftfrekvens
  2) progressiv profil-alignment som kan VAXA - kravs vid deletion,
     eftersom ingen enskild kopia innehaller hela monstret
"""

import numpy as np
from collections import Counter

from suffix_numpy import to_labels, to_str, repeats

GAP = -1  # markor for lucka


# ------------------------------------------------- 1. period via vikningspoang
def vikningspoang(labels, p):
    """Medelandel av raderna som enas per kolumn. 1.0 = perfekt periodisk."""
    n = (len(labels) // p) * p
    if n < 2 * p:
        return 0.0
    rader = labels[:n].reshape(-1, p)
    return float(np.mean([Counter(k).most_common(1)[0][1] / len(k) for k in rader.T]))


def skatta_period(labels, s=None, min_len=3, trosk=0.75):
    """
    Kandidater fran suffixarrayens skift + deras delare.
    Valj MINSTA period vars vikningspoang nar troskeln.
    """
    s = s or to_str(labels)
    skift = {x["skift"] for x in repeats(s, min_len)}
    kand = set()
    for d in skift:
        for k in range(1, d + 1):
            if d % k == 0 and 2 * k <= len(labels):
                kand.add(k)
    poang = [(p, vikningspoang(labels, p)) for p in sorted(kand) if p >= 2]
    bra = [(p, v) for p, v in poang if v >= trosk]
    return (bra[0] if bra else (None, 0.0)), sorted(poang, key=lambda t: -t[1])[:4]


def vik_och_rosta(labels, p):
    n = (len(labels) // p) * p
    rader = labels[:n].reshape(-1, p)
    kons = np.array([Counter(k).most_common(1)[0][0] for k in rader.T])
    stod = np.array([Counter(k).most_common(1)[0][1] for k in rader.T])
    return kons, stod


# ------------------------------------- 2. progressiv profil-alignment (deletion)
def _kolpoang(kol, sym, n):
    """+1 om alla i kolumnen matchar, -1 om ingen gor det."""
    return (2 * kol.get(sym, 0) - n) / max(n, 1)


def lagg_till(profil, n, seq, gap=-1.2):
    """
    Alignera seq mot profilen. Profilen kan vaxa med nya kolumner,
    vilket ar det som gor att ett monster langre an varje enskild
    kopia kan aterskapas.
    """
    L, M = len(profil), len(seq)
    D = np.zeros((L + 1, M + 1))
    D[:, 0] = np.arange(L + 1) * gap
    D[0, :] = np.arange(M + 1) * gap
    for i in range(1, L + 1):
        for j in range(1, M + 1):
            D[i, j] = max(
                D[i - 1, j - 1] + _kolpoang(profil[i - 1], seq[j - 1], n),
                D[i - 1, j] + gap,      # profilkolumn utan motsvarighet
                D[i, j - 1] + gap,      # ny kolumn (insertion)
            )
    ny, i, j = [], L, M
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + _kolpoang(
            profil[i - 1], seq[j - 1], n
        ):
            kol = profil[i - 1].copy()
            kol[seq[j - 1]] += 1
            ny.append(kol)
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + gap:
            kol = profil[i - 1].copy()
            kol[GAP] += 1
            ny.append(kol)
            i -= 1
        else:
            kol = Counter({GAP: n, seq[j - 1]: 1})
            ny.append(kol)
            j -= 1
    return ny[::-1], n + 1


def profil_konsensus(kopior):
    profil = [Counter({k: 1}) for k in kopior[0]]
    n = 1
    for k in kopior[1:]:
        profil, n = lagg_till(profil, n, k)
    ut = []
    for kol in profil:
        icke_gap = {s: c for s, c in kol.items() if s != GAP}
        if icke_gap and max(icke_gap.values()) > n / 2:
            ut.append(max(icke_gap, key=icke_gap.get))
    return np.array(ut), n


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    P = np.array([12, 47, 1003, 47, 12, 88, 47, 1003, 12, 47, 88, 1003, 47, 12])
    p = len(P)
    fel_pos = [3, 9, 1, 12, 6]

    print("Sant monster :", P, f"(period {p})\n")

    # ---- A: substitution ------------------------------------------
    seq = np.tile(P, 5).copy()
    for rep, off in enumerate(fel_pos):
        seq[rep * p + off] = 9999
    labels, vals = to_labels(seq)

    (est, v), topp = skatta_period(labels)
    print("=== A: substitution ===")
    print(f"  vikningspoang (basta): {[(a, round(b,2)) for a,b in topp]}")
    print(f"  vald period          : {est}  (poang {v:.2f})")
    kons, stod = vik_och_rosta(labels, est)
    ater = vals[kons]
    print(f"  kolumnstod           : min {stod.min()}/5")
    print(f"  aterskapat           : {ater}")
    print(f"  KORREKT              : {np.array_equal(ater, P)}\n")

    # ---- B: deletion ----------------------------------------------
    kopior_raw = [np.delete(P.copy(), off) for off in fel_pos]
    seq_b = np.concatenate(kopior_raw)
    labels_b, vals_b = to_labels(seq_b)
    granser = np.cumsum([0] + [len(k) for k in kopior_raw])
    kopior = [labels_b[granser[i]:granser[i + 1]] for i in range(5)]

    print("=== B: deletion ===")
    print(f"  kopielangder : {[len(k) for k in kopior]}  (sant monster ar {p})")
    kons_b, n = profil_konsensus(kopior)
    ater_b = vals_b[kons_b]
    print(f"  aterskapat   : {ater_b}")
    print(f"  KORREKT      : {np.array_equal(ater_b, P)}")
