"""
Adapter mellan DIN generator och probe-harnesset.

Harnesset förväntar sig att get_batch(...) returnerar en dict av torch-tensorer:

    pri       : float [n, T]   observerad PRI-sekvens (modellens input)
    level_id  : long  [n, T]   vilken PRI-nivå varje puls hör till   (probe: level)
    boundary  : long  [n, T]   1 där en run byter, annars 0          (probe: boundary)
    period    : float [n]      sann repeterande period               (probe: period)
    mask      : bool  [n, T]   giltiga positioner (padding / droppar)

För att koppla in din egen data: implementera your_generator(...) så att den
anropar ditt script och mappar dess output till dicten ovan. Sätt sedan
USE_SYNTH = False.

Den syntetiska generatorn nedan är ENDAST till för att smoke-testa hela
pipelinen end-to-end innan du kopplar in riktig data.
"""

import torch

USE_SYNTH = True     # sätt False när your_generator() är implementerad
N_LEVELS = 4
BASE = None          # sätts lazily till linspace(50,200,N_LEVELS)


def _base(n_levels):
    return torch.linspace(50.0, 200.0, n_levels)


# ----------------------------------------------------------------------
# DIN generator — fyll i denna
# ----------------------------------------------------------------------
def your_generator(n, T, jitter_type="mult", jitter_level=0.0,
                   drop_rate=0.0, seed=None, n_levels=N_LEVELS):
    """
    Anropa ditt eget data-script här och returnera dicten i kontraktet ovan.
    Måste minst exponera: pri, level_id, boundary, period, mask.
    """
    raise NotImplementedError(
        "Koppla in ditt generator-script här och sätt USE_SYNTH = False."
    )


# ----------------------------------------------------------------------
# Syntetisk fallback (stagger / dwell-and-switch) för smoke-test
# ----------------------------------------------------------------------
def synth_batch(n, T, jitter_type="mult", jitter_level=0.0,
                drop_rate=0.0, seed=None, n_levels=N_LEVELS,
                dwell_min=2, dwell_max=8):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    base = _base(n_levels)

    pri = torch.zeros(n, T)
    level_id = torch.zeros(n, T, dtype=torch.long)
    boundary = torch.zeros(n, T, dtype=torch.long)
    period = torch.zeros(n)

    for i in range(n):
        # fast dwell per nivå inom denna sekvens -> väldefinierad period
        dwells = torch.randint(dwell_min, dwell_max + 1, (n_levels,), generator=g)
        period[i] = float(dwells.sum())
        vals, lvls, bnd = [], [], []
        li = 0
        while len(vals) < T:
            d = int(dwells[li])
            for k in range(d):
                vals.append(float(base[li]))
                lvls.append(li)
                bnd.append(1 if k == 0 else 0)
            li = (li + 1) % n_levels
        v = torch.tensor(vals[:T])

        if jitter_level > 0:
            noise = torch.randn(T, generator=g)
            if jitter_type == "mult":                      # multiplikativt (procentuellt)
                v = v * (1.0 + jitter_level * noise)
            else:                                          # additivt
                v = v + jitter_level * float(base.mean()) * noise

        pri[i] = v
        level_id[i] = torch.tensor(lvls[:T])
        boundary[i] = torch.tensor(bnd[:T])

    mask = torch.ones(n, T, dtype=torch.bool)
    if drop_rate > 0:                                      # förenklad korruption: markera ogiltiga
        mask &= ~(torch.rand(n, T, generator=g) < drop_rate)
    # NOT: 'spurious pulses' (falska pulser) kräver reindexering och lämnas
    # medvetet till din riktiga generator, där det gör mest nytta.

    return dict(pri=pri, level_id=level_id, boundary=boundary,
                period=period, mask=mask)


def get_batch(n, T, jitter_type="mult", jitter_level=0.0,
              drop_rate=0.0, seed=None, n_levels=N_LEVELS, **kw):
    fn = synth_batch if USE_SYNTH else your_generator
    return fn(n, T, jitter_type=jitter_type, jitter_level=jitter_level,
              drop_rate=drop_rate, seed=seed, n_levels=n_levels, **kw)


def sample_by_level(n_levels=N_LEVELS, n_per=500,
                    jitter_type="mult", jitter_level=0.0, seed=0):
    """Rena skalärvärden per nivå (för den träningsfria separabilitets-proben)."""
    g = torch.Generator().manual_seed(seed)
    base = _base(n_levels)
    out = []
    for lvl in range(n_levels):
        noise = torch.randn(n_per, generator=g)
        if jitter_type == "mult":
            v = base[lvl] * (1.0 + jitter_level * noise)
        else:
            v = base[lvl] + jitter_level * float(base.mean()) * noise
        out.append(v)
    return out
