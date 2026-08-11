"""
Tvastegsklustring av pulssekvenser.

  Steg 1 (grovt) : komponentprofil - ordningsblind, billig
  Steg 2 (fint)  : generaliserad suffixarray - fangar ordningen

Steg 2 kors bara pa par som steg 1 slappt igenom.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from suffix_numpy import to_labels, to_str


# ------------------------------------------------------ steg 1: komponentprofil
def profil(seq, alla):
    """Normaliserad histogramvektor over komponenter. Ignorerar ordning."""
    h = np.array([np.sum(seq == k) for k in alla], float)
    return h / max(h.sum(), 1)


def grov_match(a, b, alla, trosk=0.15):
    """L1-avstand mellan komponentprofiler. Litet = kandidatpar."""
    return np.abs(profil(a, alla) - profil(b, alla)).sum() < trosk


# ------------------------------------- steg 2: generaliserad suffixarray
def matchlangder(X, Y):
    """
    For varje position i i X: langsta prefix av X[i:] som ocksa finns i Y.
    Beraknas via gemensam suffixarray over X + sep + Y.
    """
    sep = "\x00"
    T = X + sep + Y
    nX = len(X)
    sa = sorted(range(len(T)), key=lambda i: T[i:])

    def lcp(a, b):
        n = 0
        while (a + n < len(T) and b + n < len(T)
               and T[a + n] == T[b + n] and T[a + n] != sep):
            n += 1
        return n

    ml = np.zeros(nX, int)
    for r, pos in enumerate(sa):
        if pos >= nX:                      # suffix ur Y
            continue
        basta = 0
        for granne in (r - 1, r + 1):      # narmaste Y-suffix at bada hall
            if 0 <= granne < len(sa) and sa[granne] > nX:
                basta = max(basta, lcp(pos, sa[granne]))
        ml[pos] = basta
    return ml


def likhet(X, Y, cirkular=True):
    """
    ACS-likhet (average common substring), symmetriserad och normaliserad.
    cirkular=True later monstret matcha aven vid fasforskjutning.
    """
    Yx, Xx = (Y + Y, X + X) if cirkular else (Y, X)
    a = matchlangder(X, Yx).mean()
    b = matchlangder(Y, Xx).mean()
    acs = (a + b) / 2
    return acs / (acs + 1.0)           # 0..1, hogre = mer likt


# --------------------------------------------------------------------- pipeline
def klustra(sekvenser, trosk_grov=0.15, trosk_fin=0.5):
    alla = np.unique(np.concatenate(sekvenser))
    strangar = [to_str(to_labels(s)[0]) for s in sekvenser]
    n = len(sekvenser)

    D = np.ones((n, n))
    np.fill_diagonal(D, 0.0)
    slappta, avvisade = 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            if not grov_match(sekvenser[i], sekvenser[j], alla, trosk_grov):
                continue                         # steg 1 stoppade paret
            slappta += 1
            d = 1 - likhet(strangar[i], strangar[j])
            D[i, j] = D[j, i] = d
            if d > 1 - trosk_fin:
                avvisade += 1

    Z = linkage(squareform(D, checks=False), method="average")
    etiketter = fcluster(Z, t=1 - trosk_fin, criterion="distance")
    return etiketter, D, slappta, avvisade


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    P = np.array([12, 47, 1003, 47, 12, 88])

    def dropout(seq, k):
        idx = rng.choice(len(seq), k, replace=False)
        return np.delete(seq, idx)

    sekv = {
        "A1 monster P":            np.tile(P, 6),
        "A2 P, fasskiftad":        np.tile(np.roll(P, 3), 6),
        "A3 P med dropouts":       dropout(np.tile(P, 6), 4),
        "B1 SAMMA komponenter,\n     annan ordning":
                                   np.tile(np.array([12, 12, 47, 88, 47, 1003]), 6),
        "C1 annan emitter":        np.tile(np.array([500, 47, 500, 900]), 9),
    }
    namn, seqs = list(sekv), list(sekv.values())

    et, D, slappta, avvisade = klustra(seqs)

    print(f"Par som passerade steg 1 : {slappta}")
    print(f"...darav avvisade i steg 2: {avvisade}\n")

    print("Avstandsmatris (0 = identisk):")
    print("        " + "".join(f"{i:>7}" for i in range(len(namn))))
    for i, r in enumerate(D):
        print(f"  {i}   " + "".join(f"{v:>7.2f}" for v in r))

    print("\nKluster:")
    for k in sorted(set(et)):
        print(f"  kluster {k}:")
        for i in np.where(et == k)[0]:
            print(f"    [{i}] {namn[i]}")
