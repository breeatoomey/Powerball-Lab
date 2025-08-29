import argparse, os, numpy as np, pandas as pd
from .data import fetch_powerball_history
from .models import empirical_weights, sample_ticket
from .viz import plot_frequency, heatmap_interactive
from .astro import compute_astro_features_for_dates
from .features import add_calendar_features, build_number_targets
from .ml import train_number_models, predict_number_probs, compute_feature_importances
from .utils import today_local_date, next_powerball_draw_date

def cmd_run_all(args):
    df = fetch_powerball_history()
    os.makedirs("out", exist_ok=True)
    plot_frequency(df, outdir="out")
    outfile = heatmap_interactive(df, outfile="out/heatmap_cooccurrence.html", normalize=False)
    print(f"Saved interactive heatmap to {outfile}")
    w_white, w_pb = empirical_weights(df, half_life_draws=500)
    dates = sorted(set(df["draw_date"].dt.tz_convert("US/Eastern").dt.date))
    astro = compute_astro_features_for_dates(dates)
    # simple blend (toy) can be added back here if desired

    rng = np.random.default_rng(42)
    samples=[]
    for _ in range(1000):
        ws, pb = sample_ticket(w_white, w_pb, rng)
        score = np.prod([w_white[w] for w in ws]) * w_pb[pb]
        samples.append({"w": ws, "pb": pb, "score": float(score)})
    cand = pd.DataFrame(samples).sort_values("score", ascending=False).head(50)
    cand.to_csv("out/powerball_candidates.csv", index=False)
    print("Top candidates saved to out/powerball_candidates.csv")

def cmd_heatmap(args):
    df = fetch_powerball_history()
    outfile = args.outfile or "out/heatmap_cooccurrence.html"
    path = heatmap_interactive(df, outfile=outfile, normalize=args.normalize)
    print(f"Saved interactive heatmap to {path}")

def cmd_candidates(args):
    df = fetch_powerball_history()
    w_white, w_pb = empirical_weights(df, half_life_draws=args.halflife)
    rng = np.random.default_rng(args.seed if args.seed is not None else 42)
    rows=[]
    for _ in range(args.n):
        ws, pb = sample_ticket(w_white, w_pb, rng)
        rows.append({"w": ws, "pb": pb})
    out = pd.DataFrame(rows)
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        out.to_csv(args.outfile, index=False)
        print(f"Saved {args.n} candidates to {args.outfile}")
    else:
        print(out.to_string(index=False))

def cmd_ml_candidates(args):
    df = fetch_powerball_history()
    # past features
    X_cal = add_calendar_features(df)
    dates = sorted(set(df["draw_date"].dt.tz_convert("US/Eastern").dt.date))
    astro = compute_astro_features_for_dates(dates)
    df_dates = df["draw_date"].dt.tz_convert("US/Eastern").dt.date.rename("date")
    astro_indexed = astro.reindex(df_dates.values)
    astro_indexed.index = df.index
    X = X_cal.join(astro_indexed, how="left").fillna(method="ffill").fillna(method="bfill")

    Yw, Yp = build_number_targets(df)

    models_white, models_pb = train_number_models(
        X.values, Yw, Yp, model_type=args.model, max_iter=args.max_iter, random_state=args.seed
    )

    # future row (next draw date)
    next_date = next_powerball_draw_date()

    # Calendar features for one row (tz-aware)
    Xf_cal = add_calendar_features(pd.DataFrame({
        "draw_date": pd.to_datetime([str(next_date)]).tz_localize("US/Eastern")
    }))

    # Astro features for the same date — align its index to Xf_cal and DO NOT keep a 'date' column
    astro_future = compute_astro_features_for_dates([next_date])
    astro_future.index = Xf_cal.index  # align on the single row’s index

    # Join and ensure numeric
    Xf = Xf_cal.join(astro_future, how="left").ffill().bfill()
    Xf_np = Xf.astype(float).values  # ensure no datetime/object types sneak through

    Pw_row, Pp_row = predict_number_probs(models_white, models_pb, Xf_np)

    rng = np.random.default_rng(args.seed if args.seed is not None else 123)
    whites_all = list(Pw_row.index)
    p_whites = np.array([Pw_row[c] for c in whites_all], dtype=float)
    p_whites = p_whites / p_whites.sum()
    reds_all = list(Pp_row.index)
    p_reds = np.array([Pp_row[c] for c in reds_all], dtype=float)
    p_reds = p_reds / p_reds.sum()

    rows=[]
    for _ in range(args.n):
        chosen = []
        avail = whites_all.copy()
        probs = p_whites.copy()
        for __ in range(5):
            probs = probs / probs.sum()
            idx = rng.choice(np.arange(len(avail)), p=probs)
            chosen.append(int(avail[idx]))
            avail.pop(idx)
            probs = np.delete(probs, idx)
        pb = int(rng.choice(np.array(reds_all), p=p_reds))
        rows.append({"w": sorted(chosen), "pb": pb})

    out = pd.DataFrame(rows)
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        out.to_csv(args.outfile, index=False)
        print(f"Saved {args.n} ML-based candidates to {args.outfile}")
    else:
        print(out.to_string(index=False))

def cmd_ml_importances(args):
    df = fetch_powerball_history()

    # Build past features (DataFrame with named columns)
    X_cal = add_calendar_features(df)
    dates = sorted(set(df["draw_date"].dt.tz_convert("US/Eastern").dt.date))
    astro = compute_astro_features_for_dates(dates)
    df_dates = df["draw_date"].dt.tz_convert("US/Eastern").dt.date.rename("date")
    astro_indexed = astro.reindex(df_dates.values)
    astro_indexed.index = df.index
    X = X_cal.join(astro_indexed, how="left").ffill().bfill()  # numeric only

    Yw, Yp = build_number_targets(df)

    # Train OVR models
    models_white, models_pb = train_number_models(
        X.values, Yw, Yp, model_type=args.model, max_iter=args.max_iter, random_state=args.seed
    )

    # Compute global importances
    fn = X.columns
    imp_w = compute_feature_importances(models_white, fn, args.model)
    imp_p = compute_feature_importances(models_pb, fn, args.model)

    # Save CSVs
    import os
    os.makedirs("out", exist_ok=True)
    imp_w.to_csv(f"out/feature_importances_whites_{args.model}.csv", header=["importance"])
    imp_p.to_csv(f"out/feature_importances_powerball_{args.model}.csv", header=["importance"])

    # Plot bar charts (top-k) with matplotlib
    import matplotlib.pyplot as plt

    def _barplot(series, title, outfile, topk):
        s = series.head(topk)
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(s)), s.values)            # no explicit colors per your matplotlib rules
        plt.xticks(range(len(s)), s.index, rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()

    _barplot(imp_w, f"Top {args.topk} features (WHITES, {args.model.upper()})",
             f"out/feature_importances_whites_{args.model}.png", args.topk)
    _barplot(imp_p, f"Top {args.topk} features (POWERBALL, {args.model.upper()})",
             f"out/feature_importances_powerball_{args.model}.png", args.topk)

    print("Saved:")
    print(f" - out/feature_importances_whites_{args.model}.csv")
    print(f" - out/feature_importances_powerball_{args.model}.csv")
    print(f" - out/feature_importances_whites_{args.model}.png")
    print(f" - out/feature_importances_powerball_{args.model}.png")

def main():
    ap = argparse.ArgumentParser(prog="powerball-lab")
    sub = ap.add_subparsers(dest="cmd")

    ap_all = sub.add_parser("run-all", help="Fetch data, make visuals (incl. interactive heatmap), and write top candidates")
    ap_all.set_defaults(func=cmd_run_all)

    ap_hm = sub.add_parser("heatmap", help="Generate interactive co-occurrence heatmap HTML")
    ap_hm.add_argument("--outfile", type=str, default="out/heatmap_cooccurrence.html")
    ap_hm.add_argument("--normalize", action="store_true", help="Show lift vs. independence instead of raw counts")
    ap_hm.set_defaults(func=cmd_heatmap)

    ap_cand = sub.add_parser("candidates", help="Generate candidate tickets (empirical/recency weights)")
    ap_cand.add_argument("--n", type=int, default=10)
    ap_cand.add_argument("--outfile", type=str, default=None)
    ap_cand.add_argument("--seed", type=int, default=42)
    ap_cand.add_argument("--halflife", type=int, default=500, help="Half-life in draws for recency weighting")
    ap_cand.set_defaults(func=cmd_candidates)

    ap_ml = sub.add_parser("ml-candidates", help="Generate candidate tickets using ML over calendar/astro features")
    ap_ml.add_argument("--model", choices=["logreg","gbdt"], default="logreg")
    ap_ml.add_argument("--n", type=int, default=20)
    ap_ml.add_argument("--outfile", type=str, default="out/ml_candidates.csv")
    ap_ml.add_argument("--max-iter", type=int, default=200)
    ap_ml.add_argument("--seed", type=int, default=42)
    ap_ml.set_defaults(func=cmd_ml_candidates)

    ap_import = sub.add_parser("ml-importances", help="Train OVR models and export feature importances (CSV + PNG)")
    ap_import.add_argument("--model", choices=["logreg", "gbdt"], default="logreg")
    ap_import.add_argument("--max-iter", type=int, default=200)
    ap_import.add_argument("--seed", type=int, default=42)
    ap_import.add_argument("--topk", type=int, default=12, help="Top-K features to plot")
    ap_import.set_defaults(func=cmd_ml_importances) 

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)

if __name__ == "__main__":
    main()
