"""
Front-ends (representationer) att jämföra empiriskt.

Kontrakt: varje front-end tar en PRI-sekvens v: [n, T]  (+ valfri mask [n, T])
och returnerar features.
  - Punktvisa & sekvens-FE:  [n, T, d_model]   (en vektor per tidssteg)
  - Global FE (fourier_time): [n, d_model]      (EN vektor per sekvens)

Den enda saken som varieras i experimentet är front-end. Allt annat
(probe-backbone, optimizer, schema, seeds) hålls fixt.
"""

import math
import torch
import torch.nn as nn

EPS = 1e-8


# ----------------------------------------------------------------------
# Normalisering (tyst confounder -> gör den explicit och logga läget)
# ----------------------------------------------------------------------
def _stats(v, mode, mask):
    if mode == "per_seq":
        cnt = mask.sum(1, keepdim=True).clamp(min=1)
        mu = (v * mask).sum(1, keepdim=True) / cnt
        var = (((v - mu) * mask) ** 2).sum(1, keepdim=True) / cnt
        return mu, (var + EPS).sqrt()
    if mode == "global":
        cnt = mask.sum().clamp(min=1)
        mu = (v * mask).sum() / cnt
        var = (((v - mu) * mask) ** 2).sum() / cnt
        return mu, (var + EPS).sqrt()
    raise ValueError(f"okänt norm-läge: {mode}")


def normalize(v, mode="per_seq", mask=None):
    if mask is None:
        mask = torch.ones_like(v)
    else:
        mask = mask.to(v.dtype)
    mu, sd = _stats(v, mode, mask)
    return (v - mu) / sd


# ----------------------------------------------------------------------
# Front-ends
# ----------------------------------------------------------------------
class FE_Raw(nn.Module):
    """Rå skalär, linjärt projicerad. Baseline: 'modellen får lära sig allt'."""
    def __init__(self, d_model, norm="per_seq"):
        super().__init__()
        self.norm = norm
        self.proj = nn.Linear(1, d_model)

    def forward(self, v, mask=None):
        x = normalize(v, self.norm, mask).unsqueeze(-1)   # [n,T,1]
        return self.proj(x)


class FE_Delta(nn.Module):
    """Värde + diff + log-kvot mot grannen. Behöver kontext (ej punktvis)."""
    def __init__(self, d_model, norm="per_seq"):
        super().__init__()
        self.norm = norm
        self.proj = nn.Linear(3, d_model)

    def forward(self, v, mask=None):
        vn = normalize(v, self.norm, mask)
        d1 = vn - torch.roll(vn, 1, dims=1)
        d1 = torch.cat([torch.zeros_like(d1[:, :1]), d1[:, 1:]], dim=1)  # nolla pos 0
        ratio = v / (torch.roll(v, 1, dims=1) + EPS)
        lr = torch.log(ratio.clamp(min=EPS))
        lr = torch.cat([torch.zeros_like(lr[:, :1]), lr[:, 1:]], dim=1)
        x = torch.stack([vn, d1, lr], dim=-1)             # [n,T,3]
        return self.proj(x)


class FE_HardBin(nn.Module):
    """Hård one-hot-binning via embedding. Alias-problem vid bin-kanter."""
    def __init__(self, d_model, edges):
        super().__init__()
        self.register_buffer("edges", edges)              # [K-1] interna kanter
        self.emb = nn.Embedding(edges.numel() + 1, d_model)

    def forward(self, v, mask=None):
        idx = torch.bucketize(v, self.edges)              # [n,T] i 0..K-1
        return self.emb(idx)


class FE_SoftBin(nn.Module):
    """Mjuk (överlappande) binning. log_space=True ger log-binning."""
    def __init__(self, d_model, centers, tau=None, log_space=False):
        super().__init__()
        self.register_buffer("centers", centers)          # [K]
        self.log_space = log_space
        if tau is None:
            sp = (centers[1:] - centers[:-1]).mean()
            tau = float((sp ** 2).clamp(min=EPS))
        self.tau = tau
        self.table = nn.Embedding(centers.numel(), d_model)

    def forward(self, v, mask=None):
        x = torch.log(v.clamp(min=EPS)) if self.log_space else v
        d2 = (x.unsqueeze(-1) - self.centers) ** 2        # [n,T,K]
        m = torch.softmax(-d2 / self.tau, dim=-1)
        return m @ self.table.weight                      # [n,T,d]


class FE_FourierVal(nn.Module):
    """Fourier-features av (log)värdet. Mjukt, multiskaligt mellanting."""
    def __init__(self, d_model, freqs, log_space=True):
        super().__init__()
        self.register_buffer("freqs", freqs)              # [F]
        self.log_space = log_space
        self.proj = nn.Linear(2 * freqs.numel(), d_model)

    def forward(self, v, mask=None):
        p = torch.log(v.clamp(min=EPS)) if self.log_space else v
        ph = 2 * math.pi * p.unsqueeze(-1) * self.freqs   # [n,T,F]
        x = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1)
        return self.proj(x)


class FE_FourierTime(nn.Module):
    """GLOBAL periodicitets-feature: FFT över tid -> [n, d]. Endast period-proben."""
    def __init__(self, d_model, n_freq, norm="per_seq"):
        super().__init__()
        self.norm = norm
        self.n_freq = n_freq
        self.proj = nn.Linear(n_freq, d_model)

    def forward(self, v, mask=None):
        vn = normalize(v, self.norm, mask)
        spec = torch.fft.rfft(vn, dim=1).abs()            # [n, T//2+1]
        spec = spec[:, 1:self.n_freq + 1]                 # släng DC, ta n_freq
        if spec.shape[1] < self.n_freq:                   # padda om för kort
            pad = self.n_freq - spec.shape[1]
            spec = torch.cat([spec, spec.new_zeros(spec.shape[0], pad)], dim=1)
        return self.proj(spec)                            # [n, d]


# ----------------------------------------------------------------------
# Byggare för kanter/centra/frekvenser (från ett referens-sample)
# ----------------------------------------------------------------------
def make_edges(sample_values, K):
    qs = torch.linspace(0, 1, K + 1)[1:-1]
    return torch.quantile(sample_values.flatten(), qs)


def make_centers(sample_values, K, log_space=False):
    x = sample_values.flatten()
    if log_space:
        x = torch.log(x.clamp(min=EPS))
    return torch.quantile(x, torch.linspace(0, 1, K))


def make_freqs(F, f_min=0.5, f_max=50.0):
    return torch.logspace(math.log10(f_min), math.log10(f_max), F)


# ----------------------------------------------------------------------
# Fabrik + kategorier
# ----------------------------------------------------------------------
def build_frontend(name, d_model, sample_values, K=32, F=16, norm="per_seq"):
    if name == "raw":          return FE_Raw(d_model, norm)
    if name == "delta":        return FE_Delta(d_model, norm)
    if name == "hardbin":      return FE_HardBin(d_model, make_edges(sample_values, K))
    if name == "softbin":      return FE_SoftBin(d_model, make_centers(sample_values, K, False), log_space=False)
    if name == "logbin":       return FE_SoftBin(d_model, make_centers(sample_values, K, True),  log_space=True)
    if name == "fourier_val":  return FE_FourierVal(d_model, make_freqs(F), log_space=True)
    if name == "fourier_time": return FE_FourierTime(d_model, n_freq=F, norm=norm)
    raise ValueError(f"okänd front-end: {name}")


# Vilka FE som är giltiga för den träningsfria separabilitets-proben
POINTWISE = ["raw", "hardbin", "softbin", "logbin", "fourier_val"]  # position -> position
SEQUENCE  = ["delta"]           # behöver kontext
GLOBAL    = ["fourier_time"]    # en vektor per sekvens (endast period-proben)