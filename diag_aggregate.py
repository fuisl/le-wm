"""Aggregate diag_exploitation_coverage rows across seeds/models -> one table + plot."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DD = "traffic_data_cologne8"
FILES = {
    "L05ar/777": f"{DD}/diag_rows_777.csv",
    "L05ar/101": f"{DD}/diag_rows_101.csv",
    "L05ar/202": f"{DD}/diag_rows_202.csv",
    "L05/777":   f"{DD}/diag_rows_L05_777.csv",
}
XCOLS = ["cov_first", "cov_seq", "cov_first_true", "cov_seq_true",
         "switch_rate", "sw_max_agent", "n_switch0", "rare_phase"]


def within_step(df, x, y):
    rs = []
    for _, g in df.groupby("step"):
        if len(g) >= 10:
            r = spearmanr(g[x], g[y])[0]
            if np.isfinite(r):
                rs.append(r)
    return np.mean(rs) if rs else np.nan


def main():
    frames = {}
    for tag, path in FILES.items():
        try:
            frames[tag] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"(missing {path})")
    if not frames:
        sys.exit("no rows files found")

    print(f"\n{'tag':>12} {'n':>6} {'meanReg?':>9}  within-step spearman( x ~ optimism )")
    print(f"{'':>12} {'':>6} {'':>9}  " + "  ".join(f"{x[:11]:>11}" for x in XCOLS))
    for tag, df in frames.items():
        row = [within_step(df, x, "optimism") for x in XCOLS]
        print(f"{tag:>12} {len(df):>6} {'':>9}  " + "  ".join(f"{v:>11.3f}" for v in row))

    # pooled L05ar across the 3 seeds
    pool = pd.concat([frames[t] for t in frames if t.startswith("L05ar/")], ignore_index=True)
    print(f"\n=== L05ar pooled ({len(pool)} candidates, {pool['step'].nunique()} steps x seeds) ===")
    print(f"{'x':>15} {'pool pearson(opt)':>18} {'pool spearman(opt)':>18} "
          f"{'within-step spearman':>21}")
    best = None
    for x in XCOLS:
        m = np.isfinite(pool[x]) & np.isfinite(pool["optimism"])
        pp = np.corrcoef(pool[x][m], pool["optimism"][m])[0, 1]
        ps = spearmanr(pool[x][m], pool["optimism"][m])[0]
        ws = within_step(pool, x, "optimism")
        print(f"{x:>15} {pp:>18.3f} {ps:>18.3f} {ws:>21.3f}")
        if best is None or abs(ws) > abs(best[1]):
            best = (x, ws)
    # error magnitude predictability
    print(f"\n  |err_z| ~ best action feature (n_switch0) within-step spearman: "
          f"{within_step(pool, 'n_switch0', 'abs_err_z'):.3f}")
    print(f"  |err_z| ~ cov_seq_true within-step spearman: "
          f"{within_step(pool, 'cov_seq_true', 'abs_err_z'):.3f}")

    # multivariate optimism ~ features (pooled)
    feats = ["cov_seq_true", "sw_max_agent", "n_switch0", "rare_phase"]
    X = pool[feats].values.astype(float)
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
    y = pool["optimism"].values.astype(float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xd = np.c_[X[ok], np.ones(ok.sum())]
    beta, *_ = np.linalg.lstsq(Xd, y[ok], rcond=None)
    r2 = 1 - ((y[ok] - Xd @ beta) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    print(f"\n  OLS optimism ~ {feats}:  pooled R^2 = {r2:.3f}")
    print("  betas: " + "  ".join(f"{f}={b:+.3f}" for f, b in zip(feats, beta[:-1])))

    print(f"\n>>> best optimism predictor (pooled, |within-step spearman|): "
          f"{best[0]} = {best[1]:+.3f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        for x, a in zip(["cov_seq_true", "n_switch0"], ax):
            # sextile-binned optimism
            q = pd.qcut(pool[x].rank(method="first"), 8, labels=False)
            b = pool.groupby(q).agg(x=(x, "mean"), opt=("optimism", "mean"),
                                    err=("abs_err_z", "mean")).reset_index(drop=True)
            a.plot(b["x"], b["opt"], "o-", ax=a) if False else a.plot(b["x"], b["opt"], "o-")
            a.axhline(0, color="k", lw=.7)
            a.set_xlabel(x); a.set_ylabel("mean optimism (oracle_z - model_z)")
            a.set_title(f"{x}: model over-optimism vs. {x}")
        fig.suptitle("Step-0 diagnostic (L05ar, cologne8, 3 seeds pooled): "
                     "latent coverage is flat; joint switching weakly predicts over-optimism")
        fig.tight_layout()
        fig.savefig(f"{DD}/diag_aggregate.png", dpi=110)
        print(f"\nwrote {DD}/diag_aggregate.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
