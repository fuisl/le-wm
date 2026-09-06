import json, glob, numpy as np, sys
order = ["pf_base_H5","pf_hold_H5","pf_holdwarm_H5","pf_mh2_H5","pf_legal_H5","pf_legal_hold_H5","pf_legal_holdwarm_H5","pf_legal_mh2_H5","pf_legal_mh4_H8","pf_legal_peragent_H5"]
print(f"{'variant':22s} {'CEM/MP':>18s} {'orc/MP':>7s} {'CEM/orc':>7s} {'sw CEM':>6s} {'sw orc':>6s} {'sw MP':>5s}")
out = {}
for tag in order:
    try: d = json.load(open(f'results/control_{tag}.json'))
    except FileNotFoundError: continue
    tails = d['tails']
    lat = np.array([r['latent-CEM'] for _, r in tails]); mp = np.array([r['max_pressure'] for _, r in tails])
    orc = np.array([r.get('oracle-CEM', np.nan) for _, r in tails])
    sw = np.mean([m['latent-CEM']['switch_rate'] for _, m in d['metrics']])
    swo = np.nanmean([m['oracle-CEM']['switch_rate'] for _, m in d['metrics'] if 'oracle-CEM' in m])
    swm = np.mean([m['max_pressure']['switch_rate'] for _, m in d['metrics']])
    r = lat / mp; ro = orc / mp
    out[tag] = dict(cem_mp=float(r.mean()), cem_mp_p10=float(np.percentile(r,10)), cem_mp_p90=float(np.percentile(r,90)),
                    orc_mp=float(np.nanmean(ro)), cem_orc=float(lat.mean()/np.nanmean(orc)), sw_cem=float(sw), sw_orc=float(swo), sw_mp=float(swm),
                    tail_cem=float(lat.mean()), tail_mp=float(mp.mean()), tail_orc=float(np.nanmean(orc)))
    print(f"{tag:22s} {r.mean():5.2f} [{np.percentile(r,10):.2f},{np.percentile(r,90):.2f}] {np.nanmean(ro):7.2f} {lat.mean()/np.nanmean(orc):7.2f} {sw:6.2f} {swo:6.2f} {swm:5.2f}")
json.dump(out, open('results/planfix_summary.json','w'), indent=1)
