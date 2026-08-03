"""
Svep-drivrutin. Två faser:

  1) Träningsfritt separabilitets-galler (sekunder) -> sålla dåliga FE.
  2) Full probe-svep: träna EN gång per (FE, task, seed) på blandat brus,
     evaluera sedan över hela brus-rutnätet -> degraderingskurvor.

Det informativa är KURVAN (accuracy vs brus), inte punktvärden. Kör med
>=3 seeds och rapportera medel ± std. Resultat sparas till results.csv.

Kör:  python sweep.py         (liten smoke-config)
"""

import csv
import statistics as stats
from dataclasses import dataclass, field

import torch

from frontends import build_frontend, POINTWISE, GLOBAL
from probes import Probe, probe_loss, eval_metric
from separability import separability
from generator_contract import get_batch, sample_by_level


@dataclass
class Config:
    d_model: int = 64
    K: int = 32                 # antal bins
    F: int = 16                 # antal frekvenser
    n_levels: int = 4
    T: int = 128
    batch: int = 64
    n_ref: int = 256            # referens-sample för bins/centra/frekvenser
    lr: float = 1e-3
    steps: int = 300            # höj för skarpa körningar
    norm: str = "per_seq"       # 'per_seq' | 'global'  (logga vilket!)
    seeds: tuple = (0, 1, 2)
    noise_grid: tuple = (0.0, 0.02, 0.05, 0.10, 0.20)   # jitter-nivåer att eval:a
    train_noise: dict = field(default_factory=lambda: dict(jitter_level=0.05, drop_rate=0.0))
    frontends: tuple = ("raw", "delta", "hardbin", "softbin",
                        "logbin", "fourier_val", "fourier_time")
    tasks: tuple = ("level", "boundary", "period")


def _ref_values(cfg, seed):
    return get_batch(cfg.n_ref, cfg.T, seed=seed, n_levels=cfg.n_levels,
                     **cfg.train_noise)["pri"]


def train_probe(fe_name, task, cfg, seed):
    torch.manual_seed(seed)
    fe = build_frontend(fe_name, cfg.d_model, _ref_values(cfg, seed),
                        K=cfg.K, F=cfg.F, norm=cfg.norm)
    is_global = fe_name in GLOBAL
    probe = Probe(fe, task, cfg.d_model, n_levels=cfg.n_levels, is_global=is_global)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr)
    probe.train()
    for _ in range(cfg.steps):
        b = get_batch(cfg.batch, cfg.T, n_levels=cfg.n_levels, **cfg.train_noise)
        pred = probe(b["pri"], b["mask"])
        loss = probe_loss(task, pred, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    return probe


def run_sweep(cfg):
    rows = []

    # --- Fas 1: träningsfritt galler (endast punktvisa FE) ----------------
    for fe_name in POINTWISE:
        for jl in cfg.noise_grid:
            fe = build_frontend(fe_name, cfg.d_model, _ref_values(cfg, 0),
                                K=cfg.K, F=cfg.F, norm="global")
            sbl = sample_by_level(cfg.n_levels, jitter_level=jl)
            s = separability(fe, sbl)
            rows.append(dict(kind="SEP", fe=fe_name, task="-",
                             noise=jl, seed="-", metric=s))
            print(f"[SEP ] {fe_name:12s} jitter={jl:<5} silhouette={s:.3f}")

    # --- Fas 2: full probe-svep ------------------------------------------
    for fe_name in cfg.frontends:
        for task in cfg.tasks:
            if fe_name in GLOBAL and task != "period":
                continue                         # global FE endast giltig för period
            for seed in cfg.seeds:
                probe = train_probe(fe_name, task, cfg, seed)
                for jl in cfg.noise_grid:
                    b = get_batch(cfg.batch, cfg.T, jitter_level=jl,
                                  n_levels=cfg.n_levels, seed=1234)
                    with torch.no_grad():
                        pred = probe(b["pri"], b["mask"])
                    m = eval_metric(task, pred, b)
                    rows.append(dict(kind="PROBE", fe=fe_name, task=task,
                                     noise=jl, seed=seed, metric=m))
                print(f"[PROBE] {fe_name:12s} {task:9s} seed={seed} klar")

    return rows


def summarize(rows):
    """Medel ± std över seeds per (fe, task, noise)."""
    agg = {}
    for r in rows:
        if r["kind"] != "PROBE":
            continue
        key = (r["fe"], r["task"], r["noise"])
        agg.setdefault(key, []).append(r["metric"])
    print("\n=== Sammanfattning (medel ± std över seeds) ===")
    for (fe, task, noise), xs in sorted(agg.items()):
        mu = stats.mean(xs)
        sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"{fe:12s} {task:9s} jitter={noise:<5} {mu:.3f} ± {sd:.3f}")


def save_csv(rows, path="results.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "fe", "task", "noise", "seed", "metric"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSparade {len(rows)} rader till {path}")


if __name__ == "__main__":
    # Liten smoke-config: bevisa att allt kör. Höj steps/seeds för skarpt.
    cfg = Config(steps=50, seeds=(0, 1), T=96, batch=32)
    rows = run_sweep(cfg)
    summarize(rows)
    save_csv(rows)
