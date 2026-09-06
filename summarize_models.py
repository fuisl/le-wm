import json, sys, numpy as np
tags = sys.argv[1:] or ["L05ar", "L05ar_ac_v1", "L05ar_link"]
print(f"{'model':14s} {'CEM/MP':>7s} {'CEM/rand':>8s} {'regret m/r':>10s} {'drift':>6s} | {'rankA':>5s} {'rankB':>5s} {'top1B':>5s} | {'eps_plan/beh':>12s} | {'plan':>5s} {'rand_m':>6s} {'sw_share':>8s}")
for t in tags:
    try:
        c = json.load(open(f'results/compounding_{t}.json'))
        p = json.load(open(f'results/probe_vs_model_{t}.json'))
        b = json.load(open(f'results/plan_vs_beh_{t}.json'))
        m = json.load(open(f'results/matched_switching_{t}.json'))
    except FileNotFoundError as e:
        print(t, 'missing', e.filename); continue
    print(f"{t:14s} {c['tail_latent']/c['tail_mp']:7.2f} {c['tail_latent']/c['tail_random']:8.2f} {c['model_over_random']:10.2f} {c['regret_total_drift']:6.0f} | "
          f"{p['rank_A']:5.2f} {p['rank_B']:5.2f} {p['top1_B']:5.2f} | {b['mean_latent_ratio']:12.2f} | "
          f"{m['mean_lat_ratio']['plan']:5.2f} {m['mean_lat_ratio']['rand_matched']:6.2f} {m['switch_share']:8.2f}")
