"""Train LightGBM models (point + 10/90% quantiles) for 2/3/4-week horizons.
Chronological split: train <=2022 | validation 2023 | test 2024-2025 (never shuffled).
Run:  python -m src.train
"""
import json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from src.features import load_raw, build_features, add_targets, FEATURES, HORIZONS

TRAIN_END, VAL_END = "2022-12-31", "2023-12-31"
P = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)


def main():
    out = Path("models"); out.mkdir(exist_ok=True)
    df = add_targets(build_features(load_raw()))
    meta = {"features": FEATURES, "horizons": list(HORIZONS), "split": {"train_end": TRAIN_END, "val_end": VAL_END}}
    metrics, models = [], {}
    # risk-tier thresholds from TRAIN target distribution (percentiles of weekly cases)
    tr_all = df[df["date"] <= TRAIN_END]["dengue_cases"]
    q = tr_all.quantile([0.60, 0.85, 0.95]).tolist()
    meta["tier_thresholds"] = {"medium": q[0], "high": q[1], "critical": q[2]}

    def tier(x): return np.digitize(x, [q[0] + 1e-9, q[1] + 1e-9, q[2] + 1e-9])  # 0..3

    for h in HORIZONS:
        t = f"target_{h}w"
        d = df.dropna(subset=FEATURES + [t])
        tr, va = d[d["date"] <= TRAIN_END], d[(d["date"] > TRAIN_END) & (d["date"] <= VAL_END)]
        te = d[d["date"] > VAL_END]
        trva = d[d["date"] <= VAL_END]
        m = lgb.LGBMRegressor(objective="poisson", **P).fit(trva[FEATURES], trva[t])  # train+val; test untouched
        lo = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **P).fit(trva[FEATURES], trva[t])
        hi = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **P).fit(trva[FEATURES], trva[t])
        mv = lgb.LGBMRegressor(objective="poisson", **P).fit(tr[FEATURES], tr[t])  # train-only, for validation score
        seas = tr.assign(w=tr["date"].dt.isocalendar().week.astype(int)).groupby(["ward_id", "w"])[t].mean()
        for name, mod, part in [("val(2023)", mv, va), ("test(2024-25)", m, te)]:
            y, p = part[t].to_numpy(), mod.predict(part[FEATURES])
            base = tr[t].mean()
            keys = list(zip(part["ward_id"], part["date"].dt.isocalendar().week.astype(int)))
            sp = np.array([seas.get(k, base) for k in keys])
            r = dict(horizon=h, split=name, n=int(len(y)),
                     MAE=float(np.mean(np.abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                     MAE_mean_baseline=float(np.mean(np.abs(y - base))),
                     MAE_seasonal_baseline=float(np.mean(np.abs(y - sp))),
                     tier_accuracy=float(np.mean(tier(p) == tier(y))),
                     tier_within_1=float(np.mean(np.abs(tier(p) - tier(y)) <= 1)))
            if name.startswith("test"):
                l, u = lo.predict(part[FEATURES]), hi.predict(part[FEATURES])
                r["interval_coverage_80"] = float(np.mean((y >= np.minimum(l, p)) & (y <= np.maximum(u, p))))
            metrics.append(r)
        models[h] = {"point": m, "lo": lo, "hi": hi}
        print(f"h={h}w done")
    joblib.dump(models, out / "models.joblib", compress=3)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(metrics).round(3).to_string())


if __name__ == "__main__":
    main()
