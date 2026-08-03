"""
Träningsfri separabilitet: det billiga gallret. Kör detta FÖRST för att
sålla bort uppenbart dåliga representationer på sekunder utan att träna.

Idé: koda jittrade PRI-värden per nivå och mät hur väl nivåerna klustrar
(silhouette). Gäller endast PUNKTVISA front-ends (raw/bin/logbin/fourier_val),
där varje positions kodning bara beror på dess eget värde.
"""

import torch

try:
    from sklearn.metrics import silhouette_score
except ImportError as e:                              # pragma: no cover
    raise ImportError("separability kräver scikit-learn: pip install scikit-learn") from e


@torch.no_grad()
def separability(frontend, sample_by_level):
    """
    sample_by_level: list av 1D-tensorer (en per nivå) med jittrade PRI-värden.
    Returnerar silhouette-score i det kodade rummet. Högre = bättre separerat.
    """
    vals, labels = [], []
    for lvl, arr in enumerate(sample_by_level):
        vals.append(arr)
        labels.append(torch.full((arr.numel(),), lvl))
    vals = torch.cat(vals)                             # [N]
    labels = torch.cat(labels).numpy()

    # Koda som EN lång sekvens -> normaliseringsstatistik blir global över samplet,
    # och varje positions kodning beror bara på dess eget värde (för punktvisa FE).
    v2d = vals.view(1, -1)                             # [1, N]
    z = frontend(v2d)                                  # [1, N, d]
    z = z[0].cpu().numpy()                             # [N, d]
    return float(silhouette_score(z, labels))
