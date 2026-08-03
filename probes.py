"""
Liten, FIXT delad probe-modell. Backbone hålls konstant över alla front-ends
så att representationen är den enda variabeln.

Tre diagnostiska uppgifter:
  level    : per tidssteg -> vilken PRI-nivå (testar jitter-robusthet)
  boundary : per tidssteg -> byter en run här? (testar run-segmentering)
  period   : per sekvens  -> repeterande period (testar global periodicitet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class SmallEncoder(nn.Module):
    """Fast, delad backbone. Ändra inte mellan front-ends."""
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers,
                          batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d_model, d_model)

    def forward(self, h):                 # [n,T,d] -> [n,T,d]
        y, _ = self.gru(h)
        return self.out(y)


class Probe(nn.Module):
    def __init__(self, frontend, task, d_model, n_levels=None, is_global=False):
        super().__init__()
        self.fe = frontend
        self.task = task
        self.is_global = is_global
        if not is_global:
            self.enc = SmallEncoder(d_model)
        if task == "level":
            self.head = nn.Linear(d_model, n_levels)
        elif task in ("boundary", "period"):
            self.head = nn.Linear(d_model, 1)
        else:
            raise ValueError(task)

    def forward(self, v, mask=None):
        h = self.fe(v, mask)
        if self.is_global:                              # global FE: [n,d] -> period
            return self.head(h).squeeze(-1)             # [n]
        h = self.enc(h)                                 # [n,T,d]
        if self.task == "level":
            return self.head(h)                         # [n,T,C]
        if self.task == "boundary":
            return self.head(h).squeeze(-1)             # [n,T]
        # period via maskad medel-pool
        m = mask.float() if mask is not None else torch.ones_like(v)
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)            # [n]


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def probe_loss(task, pred, batch):
    mask = batch["mask"].bool()
    if task == "level":
        return F.cross_entropy(pred[mask], batch["level_id"][mask])
    if task == "boundary":
        tgt = batch["boundary"].float()
        pos_w = ((1 - tgt).sum() / tgt.sum().clamp(min=1)).clamp(max=100)  # gles positiv klass
        return F.binary_cross_entropy_with_logits(pred[mask], tgt[mask], pos_weight=pos_w)
    if task == "period":
        tgt = torch.log(batch["period"].clamp(min=EPS))   # regrera i log-rummet
        return F.mse_loss(pred, tgt)
    raise ValueError(task)


# ----------------------------------------------------------------------
# Metrik (högre = bättre för alla tre, så kurvorna blir jämförbara)
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_metric(task, pred, batch):
    mask = batch["mask"].bool()
    if task == "level":
        acc = (pred.argmax(-1)[mask] == batch["level_id"][mask]).float().mean()
        return acc.item()
    if task == "boundary":
        p = (torch.sigmoid(pred) > 0.5)[mask]
        t = batch["boundary"].bool()[mask]
        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()
        return (2 * tp / (2 * tp + fp + fn + EPS)).item()   # F1
    if task == "period":
        tgt = torch.log(batch["period"].clamp(min=EPS))
        rel = ((pred - tgt).abs() / tgt.abs().clamp(min=EPS)).mean()
        return (1.0 - rel).item()                            # ~1 = nära perfekt
    raise ValueError(task)
