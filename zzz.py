import math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from runlog import RunLog, evaluate
from data import make_pair, collate, StreamDataset

def make_eval_sets_mem(n=2000, p_drops=(0.0, 0.05, 0.1, 0.2), seed=123):
    sets = {}
    for p in p_drops:
        rng = np.random.default_rng(seed)
        sets[f"drop_{p:.2f}"] = [make_pair(rng, p_drop=p) for _ in range(n)]
    return sets

def train(model, args, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    run = RunLog("runs", config)
    eval_sets = make_eval_sets_mem()

    loader = DataLoader(StreamDataset(args.steps * args.batch, args.seed),
                        batch_size=args.batch, shuffle=True, collate_fn=collate,
                        num_workers=4, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 200
    def lr_lambda(s):
        if s < warm: return (s + 1) / warm
        p = (s - warm) / max(1, args.steps - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step, t0 = 0, time.time()
    model.train()
    for bins, cont, toa, rl, mask, tgt_in, tgt_out in loader:
        bins, cont, toa, rl, mask, tgt_in, tgt_out = [
            x.to(device) for x in (bins, cont, toa, rl, mask, tgt_in, tgt_out)]

        loss = loss_fn(model(bins, cont, toa, mask, tgt_in, rl), tgt_out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        run.log(step=step, loss=loss.item(), lr=sched.get_last_lr()[0])
        if step % 50 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  {time.time()-t0:.0f}s")

        if step % args.eval_every == 0 or step >= args.steps:
            metrics = evaluate(model, eval_sets, device, collate, run)
            run.log_eval(step, metrics)
            run.save_model(model)

        if step >= args.steps:
            break
    return run.dir
