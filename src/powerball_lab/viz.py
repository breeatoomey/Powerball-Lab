import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

WHITE_MAX = 69
PB_MAX = 26

def plot_frequency(df, outdir="out"):
    os.makedirs(outdir, exist_ok=True)
    counts_white = pd.Series(0, index=range(1, WHITE_MAX+1))
    for i in range(1,6):
        counts_white = counts_white.add(df[f"w{i}"].value_counts(), fill_value=0)
    counts_white = counts_white.sort_index()
    counts_pb = df["pb"].value_counts().reindex(range(1, PB_MAX+1), fill_value=0)

    plt.figure(figsize=(12,4))
    counts_white.plot(kind="bar")
    plt.title("White balls frequency since 2015-10-07 (current rules)")
    plt.xlabel("Number (1-69)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "freq_whites.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10,3))
    counts_pb.plot(kind="bar")
    plt.title("Powerball (red) frequency since 2015-10-07")
    plt.xlabel("Number (1-26)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "freq_pb.png"), dpi=150)
    plt.close()

def compute_cooccurrence(df):
    mat = np.zeros((WHITE_MAX, WHITE_MAX), dtype=int)
    for _, row in df.iterrows():
        s = sorted({row["w1"],row["w2"],row["w3"],row["w4"],row["w5"]})
        for a in range(len(s)):
            for b in range(a+1, len(s)):
                i, j = s[a]-1, s[b]-1
                mat[i, j] += 1
                mat[j, i] += 1
    np.fill_diagonal(mat, 0)
    return pd.DataFrame(mat, index=range(1,WHITE_MAX+1), columns=range(1,WHITE_MAX+1))

def heatmap_interactive(df, outfile="out/heatmap_cooccurrence.html", normalize=False):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    mat = compute_cooccurrence(df)
    if normalize:
        totals = mat.sum().sum() / 2
        freq = df[[f"w{i}" for i in range(1,6)]].stack().value_counts().reindex(mat.index, fill_value=0)
        p = freq / freq.sum()
        expected = np.outer(p, p) * totals * 2
        with np.errstate(divide='ignore', invalid='ignore'):
            lift = mat.values / expected
            lift[np.isnan(lift)] = 0
            mat = pd.DataFrame(lift, index=mat.index, columns=mat.columns)
    fig = px.imshow(mat, aspect="auto", origin="lower",
                    labels=dict(x="White number", y="White number", color="Co-occurrence" if not normalize else "Lift"),
                    x=mat.columns, y=mat.index,
                    title="White Ball Co-occurrence Heatmap" + (" (Lift)" if normalize else ""))
    fig.update_layout(hovermode="closest")
    fig.write_html(outfile, include_plotlyjs="cdn")
    return outfile
