"""
Jamfor sekvenser nar GMM-komponenternas ID INTE ar gemensamma.

Losning: matcha komponenter pa deras parametrar (mu, Sigma) med
Bhattacharyya-avstand + ungersk algoritm, oversatt signal B till
signal A:s alfabet, kor sedan suffixarrayjamforelsen som vanligt.

Fungerar for 1D (bara PRI) och nD (PRI, RF, pulsbredd, ...).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from suffix_numpy import to_str
from klustring import likhet

OMATCHAD = -1


# ------------------------------------------------- avstand mellan komponenter
def bhattacharyya(mu1, S1, mu2, S2):
    """Sluten form for tva gaussfordelningar. Hanterar godtycklig dimension."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    S1, S2 = np.atleast_2d(S1), np.atleast_2d(S2)
    S = (S1 + S2) / 2
    d = (mu1 - mu2).reshape(-1, 1)
    term1 = (0.125 * d.T @ np.linalg.inv(S) @ d).item()
    term2 = 0.5 * np.log(np.linalg.det(S) / np.sqrt(np.linalg.det(S1) * np.linalg.det(S2)))
    return term1 + term2


def hellinger(mu1, S1, mu2, S2):
    """0 = identiska, 1 = helt atskilda. Latt att satta troskel pa."""
    return float(np.sqrt(max(0.0, 1 - np.exp(-bhattacharyya(mu1, S1, mu2, S2)))))


# ------------------------------------------------------------ matcha alfabeten
def matcha_komponenter(gmmA, gmmB, trosk=0.5):
    """
    gmmX = lista av (mu, Sigma). Returnerar dict B-index -> A-index,
    samt kostnadsmatrisen. Par over troskeln lamnas omatchade.
    """
    C = np.array([[hellinger(*a, *b) for b in gmmB] for a in gmmA])

    # kvadratisk utfyllnad sa att olika antal komponenter fungerar
    n = max(C.shape)
    Cp = np.full((n, n), 1.0)
    Cp[: C.shape[0], : C.shape[1]] = C

    rader, kol = linear_sum_assignment(Cp)
    karta = {}
    for r, k in zip(rader, kol):
        if r < len(gmmA) and k < len(gmmB) and C[r, k] <= trosk:
            karta[k] = r
    return karta, C


def oversatt(seqB, karta, nA):
    """B:s komponent-ID -> A:s alfabet. Omatchade far egna unika symboler."""
    ut, nasta = [], nA
    privat = {}
    for x in seqB:
        if x in karta:
            ut.append(karta[x])
        else:
            if x not in privat:          # unik symbol som aldrig kan matcha A
                privat[x] = nasta
                nasta += 1
            ut.append(privat[x])
    return np.array(ut)


def jamfor(seqA, gmmA, seqB, gmmB, trosk=0.5):
    karta, C = matcha_komponenter(gmmA, gmmB, trosk)
    seqB_ov = oversatt(seqB, karta, len(gmmA))
    sA = to_str(np.asarray(seqA))
    sB = to_str(seqB_ov)
    return likhet(sA, sB), karta, C


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Samma emitter, tva oberoende GMM-anpassningar.
    # A: tre komponenter. B: samma tre men i ANNAN ordning, lite andra
    #    parametrar, plus en extra falsk komponent fran en dropout (~2x PRI).
    gmmA = [(np.array([12.0]), np.array([[0.30]])),      # A:0
            (np.array([47.0]), np.array([[0.90]])),      # A:1
            (np.array([100.0]), np.array([[1.50]]))]     # A:2

    gmmB = [(np.array([99.4]), np.array([[1.70]])),      # B:0 ~ A:2
            (np.array([11.8]), np.array([[0.35]])),      # B:1 ~ A:0
            (np.array([200.0]), np.array([[4.00]])),     # B:2 falsk, ingen match
            (np.array([47.4]), np.array([[0.80]]))]      # B:3 ~ A:1

    P_A = np.array([0, 1, 2, 1, 0, 1])                   # monster i A:s ID
    P_B = np.array([1, 3, 0, 3, 1, 3])                   # SAMMA monster, B:s ID

    seqA = np.tile(P_A, 6)
    seqB = np.tile(P_B, 6)

    print("Hellingeravstand A-rader x B-kolumner:")
    _, C = matcha_komponenter(gmmA, gmmB)
    print("        " + "".join(f"{f'B{j}':>8}" for j in range(len(gmmB))))
    for i, r in enumerate(C):
        print(f"  A{i}  " + "".join(f"{v:>8.3f}" for v in r))

    lik, karta, _ = jamfor(seqA, gmmA, seqB, gmmB)
    print(f"\n  matchning B->A : {karta}")
    print(f"  (B2 saknas = falsk komponent utan motsvarighet)")
    print(f"\n  likhet EFTER oversattning : {lik:.3f}")

    # kontroll: utan oversattning
    lik_naiv = likhet(to_str(seqA), to_str(seqB))
    print(f"  likhet UTAN oversattning  : {lik_naiv:.3f}")

    # kontroll: annan emitter ska fortfarande avvisas
    P_C = np.array([1, 1, 3, 0, 3, 3])                   # samma komponenter, annan ordning
    lik_c, _, _ = jamfor(seqA, gmmA, np.tile(P_C, 6), gmmB)
    print(f"  annan ordning, samma komp : {lik_c:.3f}")
