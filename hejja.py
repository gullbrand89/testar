def make_eval_sets_mem(n=2000, p_drops=(0.0, 0.05, 0.1, 0.2), seed=123):
    sets = {}
    for p in p_drops:
        rng = np.random.default_rng(seed)          # samma startpunkt för varje p
        sets[f"drop_{p:.2f}"] = [make_pair(rng, p_drop=p) for _ in range(n)]
    return sets
