"""
pri_baseline.py
================
En deterministisk baseline för att plocka ut och klassa PRI-sekvenser
(stagger / dwell-and-switch) ur ett deinterleavat pulståg.

Tänkt som jämförelsepunkt mot ett nätverk. Ingen träning, inga vikter,
inget slumpmässigt i själva metoden — kör du den två gånger får du samma
svar. Det är precis vad du vill mäta ett nätverk mot.

Pipeline:
    råa PRI-värden  --(BGMM, valfritt)-->  hårda labels + responsibilities
    labels i TOA-ordning  --(kanonikalisering)-->  jämförbar periodsträng
    periodsträng  --(avstånd mot mallar)-->  klass

Kanonikaliseringen gör två saker, båda nödvändiga (se resonemanget om
AABAC vs ABAAC):
    1. Döper om symboler efter PRI-nivåns rang  -> tar bort label switching
    2. Roterar till minimal necklace-form         -> tar bort fas-godtycke

Beroenden: numpy, scikit-learn (bara för from_pri).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Diskretisering: råa PRI-värden -> labels  (valfritt steg)
# ---------------------------------------------------------------------------

def pri_to_labels(pri, max_components=8, random_state=0):
    """
    Klustra PRI-värden med en Bayesiansk gaussisk mixtur.

    Returnerar (labels, responsibilities, means) DÄR labels redan är
    ommappade så att symbol 0 = lägsta PRI-nivån, 1 = näst lägsta, osv.
    Det gör dem stabila mellan körningar och matchar ground-truth-ordning.

    pri : array (n_pulser,) av PRI-värden i TOA-ordning.
    """
    from sklearn.mixture import BayesianGaussianMixture

    pri = np.asarray(pri, dtype=float).reshape(-1, 1)
    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        # låg prior => modellen får beskära oanvända komponenter själv
        weight_concentration_prior=1e-2,
        max_iter=500,
        random_state=random_state,
    )
    raw = bgmm.fit_predict(pri)                 # godtyckliga kluster-ID
    resp = bgmm.predict_proba(pri)              # (n, max_components)

    # Behåll bara komponenter som faktiskt används, sortera på medel-PRI.
    used = np.unique(raw)
    means_used = bgmm.means_.ravel()[used]
    order = used[np.argsort(means_used)]        # kluster-ID i PRI-ordning
    remap = {old: new for new, old in enumerate(order)}

    labels = np.array([remap[c] for c in raw], dtype=int)
    resp = resp[:, order]                        # kolumner i samma ordning
    means = np.sort(means_used)
    return labels, resp, means


# ---------------------------------------------------------------------------
# 2. Periodsökning på den ordnade symbolsträngen
# ---------------------------------------------------------------------------

def find_period(labels, max_period=None, min_match=0.9):
    """
    Minsta P sådant att labels[n] == labels[n+P] för minst `min_match`
    andel av överlappet. Symbolisk autokorrelation — tål viss brist.

    Returnerar (period, score) där score är matchgraden vid vald period.
    Faller tillbaka på hela längden om inget kortare P kvalar in
    (t.ex. rena dwell-and-switch-block utan repetition i fönstret).
    """
    labels = np.asarray(labels)
    n = len(labels)
    if max_period is None:
        max_period = n // 2
    best = (n, 1.0)
    for p in range(1, max_period + 1):
        a, b = labels[:-p], labels[p:]
        if len(a) == 0:
            continue
        score = np.mean(a == b)
        if score >= min_match:
            return p, score
        if p == 1:
            best = (n, score)
    return best


# ---------------------------------------------------------------------------
# 3. Kanonikalisering: gör två periodsträngar jämförbara
# ---------------------------------------------------------------------------

def minimal_rotation(seq):
    """
    Booths algoritm: lexikografiskt minsta rotationen (necklace-form),
    O(n). Tar bort fas-godtycket så AABAC och dess rotation ACAAB
    hamnar på samma kanoniska form.
    """
    s = list(seq) * 2
    n = len(seq)
    f = [-1] * len(s)
    k = 0
    for j in range(1, len(s)):
        sj = s[j]
        i = f[j - k - 1]
        while i != -1 and sj != s[k + i + 1]:
            if sj < s[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if sj != s[k + i + 1]:
            if sj < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return tuple(s[k:k + n])


def canonical_period(labels, max_period=None, min_match=0.9):
    """
    Fullständig kanonform: hitta period -> klipp ut en period ->
    minimal rotation. `labels` antas redan vara PRI-rangordnade
    (det gör pri_to_labels; har du egna labels, se relabel_by_rank).

    Returnerar (canonical_tuple, period, score).
    """
    p, score = find_period(labels, max_period, min_match)
    period_seq = np.asarray(labels)[:p]
    return minimal_rotation(period_seq), p, score


def relabel_by_rank(labels, values):
    """
    Om du har egna labels men de inte är PRI-ordnade: mappa om dem så att
    symbol 0 = lägsta nivå. `values` = representativt PRI-värde per symbol
    (t.ex. medelvärdet). Behövs för att jämföra mot ground-truth vars
    A/B/C är nivåordnade.
    """
    uniq = np.unique(labels)
    order = uniq[np.argsort([values[u] for u in uniq])]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in labels], dtype=int)


# ---------------------------------------------------------------------------
# 4. Avstånd + klassning mot mallar
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    """Levenshtein mellan två symbolsekvenser. Tål insättning/borttagning
    (missade/falska pulser), inte bara substitution som Hamming."""
    a, b = list(a), list(b)
    m, n = len(a), len(b)
    dp = np.arange(n + 1)
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return int(dp[n])


class PRISequenceClassifier:
    """
    Nearest-template-klassificerare på kanoniska periodsträngar.

    Mata den med kända mallar (namn -> sekvens). Vid predict
    kanonikaliseras både mallar och indata på samma sätt, så jämförelsen
    är oberoende av label switching och fas. Avståndet är edit-distans,
    normaliserat mot maxlängd så olika periodlängder kan jämföras rättvist.
    """

    def __init__(self, templates: dict):
        # kanonikalisera mallarna en gång; antas redan nivåordnade (0=lägst)
        self.templates = {
            name: minimal_rotation(seq) for name, seq in templates.items()
        }

    def predict(self, labels, max_period=None, min_match=0.9):
        canon, period, score = canonical_period(labels, max_period, min_match)
        results = []
        for name, tmpl in self.templates.items():
            d = edit_distance(canon, tmpl)
            norm = d / max(len(canon), len(tmpl))
            results.append((name, norm, d))
        results.sort(key=lambda r: r[1])
        best_name, best_norm, best_d = results[0]
        return {
            "prediction": best_name,
            "canonical": canon,
            "period": period,
            "period_score": round(float(score), 3),
            "distance": best_d,
            "confidence": round(1.0 - best_norm, 3),
            "ranking": results,
        }


# ---------------------------------------------------------------------------
# 5. Syntetisk demo: visar att AABAC och ABAAC separeras
# ---------------------------------------------------------------------------

def _synth(pattern_levels, pri_map, n_periods=40, jitter=0.01,
           p_missing=0.0, seed=0):
    """Bygg ett brusigt PRI-tåg från en symbolmall. pri_map: symbol->PRI."""
    rng = np.random.default_rng(seed)
    true_labels, pri = [], []
    for _ in range(n_periods):
        for sym in pattern_levels:
            if rng.random() < p_missing:
                continue
            true_labels.append(sym)
            pri.append(pri_map[sym] * (1.0 + rng.normal(0, jitter)))
    return np.array(true_labels), np.array(pri)


if __name__ == "__main__":
    # Två stagger-sekvenser med IDENTISK sammansättning och bigram-statistik.
    # Enda skillnaden är ordningen -> testar exakt det svåra fallet.
    pri_map = {0: 100.0, 1: 150.0, 2: 200.0}   # A=100us, B=150us, C=200us
    patterns = {
        "AABAC": [0, 0, 1, 0, 2],
        "ABAAC": [0, 1, 0, 0, 2],
    }

    clf = PRISequenceClassifier(patterns)

    print("Mallar (kanoniska):")
    for name, t in clf.templates.items():
        print(f"  {name:6s} -> {t}")
    print()

    for true_name, levels in patterns.items():
        # generera brusigt tåg av RÅA PRI-värden (som om från mottagaren)
        _, pri = _synth(levels, pri_map, n_periods=40,
                        jitter=0.015, p_missing=0.02, seed=42)

        # steg 1: BGMM diskretiserar utan att veta antal nivåer i förväg
        labels, resp, means = pri_to_labels(pri, max_components=8)

        # steg 2+3+4: kanonikalisera och klassa
        out = clf.predict(labels)

        ok = "OK" if out["prediction"] == true_name else "FEL"
        print(f"[{ok}] sann={true_name}  gissning={out['prediction']}"
              f"  conf={out['confidence']}")
        print(f"      BGMM hittade {len(means)} nivaer:"
              f" {np.round(means,1).tolist()}")
        print(f"      kanonisk periodstrang: {out['canonical']}"
              f"  (period={out['period']}, match={out['period_score']})")
        print(f"      ranking: "
              + ", ".join(f"{n}:{nd:.2f}" for n, nd, _ in out["ranking"]))
        print()
