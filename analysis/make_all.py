"""All tables, figures and numbers for the paper, from the saved predictions (CPU only).
python analysis/make_all.py   (run from /home/claude/hate)"""
import json, os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "analysis")
from metrics import pair_metrics, label_pairs, GRID
from core import drops, pair_means, fe_model, CONSTRUCT

warnings.filterwarnings("ignore")
OUT = "paper/out"; FIG = f"{OUT}/figures"; TAB = f"{OUT}/tables"
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
C4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical slots 1-4 (adjacent pairs)
MODELS = ["tfidf", "muril", "xlmr", "indicbert"]
MNAME = {"tfidf": "TF-IDF+LR", "muril": "MuRIL", "xlmr": "XLM-R", "indicbert": "IndicBERTv2"}
LLM = {"llama31_8b": "Llama-3.1-8B", "aya_expanse_8b": "Aya-Expanse-8B", "qwen25_7b": "Qwen2.5-7B"}
N = {}  # every number quoted in the text

U = pd.read_csv("s3a/out/unit_stats.csv")
META = pd.read_parquet("s3a/out/collection_meta.parquet")
LANG_ORDER = ["hi", "mr", "bn", "ur", "ta", "te", "kn", "ml"]
UNITS = sorted(U.unit, key=lambda u: (LANG_ORDER.index(U.set_index("unit").lang[u]), u))
U["construct"] = U.source.map(CONSTRUCT)

# ------------------------------------------------------------------ pair metrics (cached)
def cached(path, fn):
    if os.path.exists(path):
        return pd.read_csv(path)
    d = fn(); d.to_csv(path, index=False); return d

M = cached("analysis/pair_metrics_final.csv", lambda: label_pairs(pd.concat(
    [pair_metrics(pd.read_parquet(f"s3a/out/preds_{m}.parquet")) for m in MODELS]), U))
TRAIN_UNITS = sorted(M.src.unique(), key=UNITS.index)
P = cached("analysis/pooled_metrics.csv", lambda: pair_metrics(pd.read_parquet("s3b/out/preds_muril_pooled.parquet")))

# ------------------------------------------------------------------ Table: units
LIC = {"MACD": "CC BY-NC-SA 4.0", "Uli": "CC BY 4.0", "MahaHate": "CC BY-NC-SA 4.0", "MOLD": "CC BY 4.0", "DravidianCodeMix": "CC BY 4.0",
       "HASOC2021": "CC BY-NC 4.0", "CONSTRAINT": "CC BY 4.0", "Romim": "Apache-2.0", "MultiLabelBangla": "CC BY-NC 4.0",
       "TamilHS": "Apache-2.0", "TeluguNews": "CC BY 4.0", "HateCheckHi": "CC BY 4.0", "RUHSOLD": "MIT"}
ut = U.set_index("unit").loc[UNITS].reset_index()
ut["licence"] = ut.source.map(LIC)
ut[["unit", "source", "lang", "role", "construct", "n_train", "n_dev", "n_test", "n_test_eval", "pos_rate", "latin_share", "leak_dev",
    "leak_test", "cross_source_dups_removed", "licence"]].to_csv(f"{TAB}/units.csv", index=False)
N["n_units"] = len(U); N["n_sources"] = U.source.nunique(); N["n_langs"] = U.lang.nunique()
N["n_items"] = int(META[META.split != "test_eval"].shape[0]); N["n_train_units"] = len(TRAIN_UNITS)
N["n_test_eval"] = int((META.split == "test_eval").sum())
N["leak_removed"] = int(U.leak_dev.sum() + U.leak_test.sum()); N["xdup_removed"] = int(U.cross_source_dups_removed.sum())

# ------------------------------------------------------------------ Fig: MuRIL transfer heatmap
def heatmap(model, fname, title):
    h = M[M.model == model].groupby(["src", "tgt"]).auc.mean().unstack().reindex(index=TRAIN_UNITS, columns=UNITS)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    im = ax.imshow(h.values, cmap="Blues", vmin=0.45, vmax=1.0, aspect="auto")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=5.2, color="white" if v > 0.8 else "#0b0b0b",
                    weight="bold" if h.index[i] == h.columns[j] else "normal")
            if h.index[i] == h.columns[j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#0b0b0b", lw=0.9))
    ax.set_xticks(range(len(UNITS))); ax.set_xticklabels(UNITS, rotation=60, ha="right")
    ax.set_yticks(range(len(TRAIN_UNITS))); ax.set_yticklabels(TRAIN_UNITS)
    ax.set_xlabel("Target (test) unit"); ax.set_ylabel("Source (training) unit, 2,000 items")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("AUROC"); cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.savefig(f"{FIG}/{fname}.pdf"); fig.savefig(f"{FIG}/{fname}.png"); plt.close(fig)
    return h

H = heatmap("muril", "heatmap_muril", "MuRIL, size-matched transfer (mean of 3 seeds); boxed = in-dataset")
heatmap("tfidf", "heatmap_tfidf", "TF-IDF char n-grams, size-matched transfer (mean of 3 seeds); boxed = in-dataset")

# ------------------------------------------------------------------ 2x2 decomposition
B = drops(M)
FE = pd.read_csv("analysis/fe_decomposition.csv") if os.path.exists("analysis/fe_decomposition.csv") else None
if FE is None or len(FE) < 16:
    FE = pd.DataFrame([{"model": m, "metric": y, "exclMACD": ex, **fe_model(B, m, y, ex)} for m in MODELS for y in ("d_auc", "d_f1")
                       for ex in (False, True)])
    FE.to_csv("analysis/fe_decomposition.csv", index=False)
FE.round(3).to_csv(f"{TAB}/fe_decomposition.csv", index=False)
# raw cell means for the descriptive table
cell = B.groupby(["model", "ptype"])[["d_auc", "d_f1"]].mean().unstack().round(3)
cell.to_csv(f"{TAB}/drop_cells.csv")
cnt = pair_means(B, "muril").groupby("ptype").size().to_dict(); N["pairs_by_type"] = cnt

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, ex in zip(axes, (False, True)):
    f = FE[(FE.metric == "d_auc") & (FE.exclMACD == ex)].set_index("model").loc[MODELS]
    x = np.arange(3); w = 0.19
    for k, m in enumerate(MODELS):
        r = f.loc[m]; vals = [r.lang, r.data, r.inter]
        lo = [r.lang - r.lang_lo, r.data - r.data_lo, r.inter - r.inter_lo]; hi = [r.lang_hi - r.lang, r.data_hi - r.data, r.inter_hi - r.inter]
        ax.bar(x + (k - 1.5) * w, vals, w * 0.9, color=C4[k], label=MNAME[m], yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5, ecolor="#52514e"))
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["Language\nshift $\\beta_L$", "Dataset\nshift $\\beta_D$", "Interaction\n$\\gamma$"])
    ax.set_title("All pairs" if not ex else "Excluding MACD$\\to$MACD pairs", loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("AUROC drop vs. in-dataset")
axes[0].legend(frameon=False, ncol=2, loc="upper right")
fig.savefig(f"{FIG}/decomposition.pdf"); fig.savefig(f"{FIG}/decomposition.png"); plt.close(fig)

# ------------------------------------------------------------------ F1 identity: ranking part + threshold part
B["rank_part"] = B.f1_in - B.f1_oracle
idt = B.groupby(["model", "ptype"])[["d_f1", "rank_part", "thr_gain"]].mean().round(3).reset_index()
idt.to_csv(f"{TAB}/f1_identity.csv", index=False)
PT = ["language shift", "dataset shift", "both shift"]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)
for ax, m in zip(axes, MODELS):
    d = idt[idt.model == m].set_index("ptype").loc[PT]
    ax.bar(range(3), d.rank_part, 0.6, color=C4[0], label="ranking loss (oracle-threshold F1)")
    ax.bar(range(3), d.thr_gain, 0.6, bottom=d.rank_part, color=C4[1], label="threshold loss")
    ax.set_xticks(range(3)); ax.set_xticklabels(["lang.", "data.", "both"]); ax.set_title(MNAME[m], loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Macro-F1 drop")
axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2)
fig.savefig(f"{FIG}/f1_identity.pdf"); fig.savefig(f"{FIG}/f1_identity.png"); plt.close(fig)

# ------------------------------------------------------------------ prior-shift threshold correction (label-shift model)
def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))

def em_prior(p, pi_s, iters=100):
    """Saerens et al. (2002) EM for the target prior under label shift."""
    pi = pi_s
    for _ in range(iters):
        w1 = pi / pi_s; w0 = (1 - pi) / (1 - pi_s)
        q = w1 * p / (w1 * p + w0 * (1 - p)); new = q.mean()
        if abs(new - pi) < 1e-6:
            break
        pi = new
    return float(pi)

def f1m(y, p, t):
    return f1_score(y, p > t, average="macro")

TR_PRIOR = META[META.split == "train"].groupby("unit").broad.mean().to_dict()
def prior_shift(model):
    pr = pd.read_parquet(f"s3a/out/preds_{model}.parquet")
    rows = []
    for (s, src, tgt), g in pr[pr.src != pr.tgt].groupby(["seed", "src", "tgt"]):
        if tgt in ("TeluguNews-te", "HateCheck-hi"):
            continue
        y, p, thr = g.y.values, g.p.values, g.thr.iloc[0]
        if y.min() == y.max():
            continue
        pi_s = 0.5 if model == "tfidf" else TR_PRIOR[src]      # TF-IDF was trained with balanced class weights
        pi_t = y.mean(); pi_em = em_prior(p, pi_s)
        sh = lambda pt: 1 / (1 + np.exp(-(logit(thr) - logit(pt) + logit(pi_s))))
        f_or = [f1m(y, p, t) for t in GRID]
        rows.append({"model": model, "seed": s, "src": src, "tgt": tgt, "f1_src": f1m(y, p, thr), "f1_oracle": max(f_or),
                     "f1_prior_oracle": f1m(y, p, sh(pi_t)), "f1_prior_em": f1m(y, p, sh(pi_em)),
                     "dlogodds": logit(pi_t) - logit(pi_s), "dthr_oracle": logit(GRID[int(np.argmax(f_or))]) - logit(thr),
                     "pi_t": pi_t, "pi_em": pi_em})
    return pd.DataFrame(rows)

PS = cached("analysis/prior_shift.csv", lambda: pd.concat([prior_shift(m) for m in MODELS]))
PS = label_pairs(PS, U)
ps = PS.groupby(["model", "ptype"])[["f1_src", "f1_prior_em", "f1_prior_oracle", "f1_oracle"]].mean()
ps["recovered_oracle_prior"] = (ps.f1_prior_oracle - ps.f1_src) / (ps.f1_oracle - ps.f1_src)
ps["recovered_em_prior"] = (ps.f1_prior_em - ps.f1_src) / (ps.f1_oracle - ps.f1_src)
ps.round(3).to_csv(f"{TAB}/prior_shift.csv")
cor = PS.groupby("model").apply(lambda d: pd.Series({"slope": np.polyfit(-d.dlogodds, d.dthr_oracle, 1)[0],
                                                     "r": np.corrcoef(-d.dlogodds, d.dthr_oracle)[0, 1],
                                                     "em_abs_err": (d.pi_em - d.pi_t).abs().mean()}))
cor.round(3).to_csv(f"{TAB}/prior_shift_corr.csv")
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True, sharex=True)
for ax, m in zip(axes, MODELS):
    d = PS[PS.model == m]
    ax.scatter(-d.dlogodds, d.dthr_oracle, s=4, alpha=0.35, color=C4[0], lw=0)
    xx = np.linspace(-4, 4, 2); ax.plot(xx, xx, color="#52514e", lw=0.7, ls="--")
    ax.set_title(f"{MNAME[m]}  r={cor.loc[m, 'r']:.2f}", loc="left"); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.grid(color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("logit(oracle thr) - logit(source thr)")
fig.supxlabel("predicted shift  logit $\\pi_s$ - logit $\\pi_t$", fontsize=8, y=-0.08)
fig.savefig(f"{FIG}/prior_shift.pdf"); fig.savefig(f"{FIG}/prior_shift.png"); plt.close(fig)

# ------------------------------------------------------------------ residual regression on transfer pairs
g_all = []
for m in MODELS:
    g = pair_means(B, m)
    us = U.set_index("unit")
    g["construct_mismatch"] = (g.src_source.map(CONSTRUCT) != g.tgt_source.map(CONSTRUCT)).astype(int)
    g["d_pos"] = (g.src.map(us.pos_rate) - g.tgt.map(us.pos_rate)).abs()
    g["d_latin"] = (g.src.map(us.latin_share) - g.tgt.map(us.latin_share)).abs()
    g["model"] = m; g_all.append(g)
G = pd.concat(g_all)
rr = []
for m in MODELS:
    for sub, q in (("all transfer pairs", G[G.model == m]), ("dataset-shift pairs", G[(G.model == m) & (G.ptype == "dataset shift")])):
        r = smf.ols("d_auc ~ C(tgt) + lang_shift + data_shift + construct_mismatch + d_pos + d_latin", q).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(q.tgt)[0]}) if sub.startswith("all") else \
            smf.ols("d_auc ~ C(tgt) + construct_mismatch + d_pos + d_latin", q).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(q.tgt)[0]})
        row = {"model": m, "subset": sub, "n": int(r.nobs)}
        for k in ("construct_mismatch", "d_pos", "d_latin"):
            row[k] = r.params[k]; row[k + "_p"] = r.pvalues[k]
        base = smf.ols("d_auc ~ C(tgt)" + (" + lang_shift + data_shift" if sub.startswith("all") else ""), q).fit()
        row["dR2"] = r.rsquared - base.rsquared
        rr.append(row)
RR = pd.DataFrame(rr); RR.round(3).to_csv(f"{TAB}/residual_regression.csv", index=False)

# ------------------------------------------------------------------ LLMs, pooled, per-unit comparison on the common 500-item subset
LP = pd.concat([pd.read_parquet(f"s2/out/preds_llm_{t}.parquet") for t in LLM])
ev_ids = set(LP[(LP.model == "llama31_8b") & (LP.split == "test_eval")].uid)
N["llm_items"] = {t: int((LP.model == t).sum()) for t in LLM}

def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan

def llm_unit(t):
    rows = []
    d = LP[LP.model == t]
    for u, g in d.groupby("unit"):
        e, v = g[g.split == "test_eval"], g[g.split == "dev"]
        thr = max(GRID, key=lambda x: f1m(v.y.values, v.p.values, x)) if len(v) and v.y.nunique() > 1 else 0.5
        rows.append({"model": t, "unit": u, "n": len(e), "auc": auc(e.y, e.p), "f1_05": f1m(e.y.values, e.p.values, 0.5),
                     "f1_dev": f1m(e.y.values, e.p.values, thr), "says_yes": (e.p > 0.5).mean(), "pos": e.y.mean(), "mass": e["mass"].mean()})
    return rows
LU = pd.DataFrame(sum([llm_unit(t) for t in LLM], []))
LU.round(3).to_csv(f"{TAB}/llm_units.csv", index=False)
LS = LU.groupby("model")[["auc", "f1_05", "f1_dev", "says_yes", "pos"]].mean().loc[list(LLM)]
LS.round(3).to_csv(f"{TAB}/llm_summary.csv")

enc = pd.read_parquet("s3a/out/preds_muril.parquet"); pool = pd.read_parquet("s3b/out/preds_muril_pooled.parquet")
src_of = U.set_index("unit").source.to_dict()
def sub_auc(d):
    d = d[d.uid.isin(ev_ids)]
    return d.groupby(["seed", "src", "tgt"]).apply(lambda g: auc(g.y, g.p)).groupby(["src", "tgt"]).mean()
A_enc = sub_auc(enc).rename("auc").reset_index(); A_pool = sub_auc(pool).rename("auc").reset_index()
rows = []
for u in UNITS:
    r = {"unit": u}
    a = A_enc[A_enc.tgt == u]
    r["in_dataset"] = a[a.src == u].auc.mean() if u in TRAIN_UNITS else np.nan
    tr = a[a.src != u]; r["best_transfer"] = tr.auc.max(); r["median_transfer"] = tr.auc.median()
    pu = A_pool[A_pool.tgt == u]; r["pooled"] = pu[pu.src == "POOLED"].auc.mean()
    r["loso"] = pu[pu.src == f"LOSO:{src_of[u]}"].auc.mean() if f"LOSO:{src_of[u]}" in set(pu.src) else np.nan
    for t in LLM:
        r[t] = LU[(LU.model == t) & (LU.unit == u)].auc.mean()
    rows.append(r)
CMP = pd.DataFrame(rows); CMP.round(3).to_csv(f"{TAB}/per_unit_comparison.csv", index=False)
tr17 = CMP[CMP.unit.isin(TRAIN_UNITS)]
N["cmp_means_17"] = tr17.drop(columns="unit").mean().round(3).to_dict()
N["cmp_means_20"] = CMP.drop(columns="unit").mean().round(3).to_dict()
def wil(a, b, d):
    x = d[[a, b]].dropna(); st = wilcoxon(x[a], x[b])
    return {"pair": f"{a} vs {b}", "n": len(x), "mean_diff": float((x[a] - x[b]).mean()), "wins": int((x[a] > x[b]).sum()), "p": float(st.pvalue)}
TESTS = pd.DataFrame([wil("pooled", "in_dataset", CMP), wil("loso", "in_dataset", CMP), wil("pooled", "loso", CMP),
                      wil("in_dataset", "llama31_8b", CMP), wil("pooled", "llama31_8b", CMP), wil("best_transfer", "llama31_8b", CMP),
                      wil("loso", "llama31_8b", CMP), wil("median_transfer", "llama31_8b", CMP)])
TESTS.round(4).to_csv(f"{TAB}/paired_tests.csv", index=False)

fig, ax = plt.subplots(figsize=(7.0, 3.0))
series = [("in_dataset", "MuRIL in-dataset", C4[0], "o"), ("pooled", "MuRIL pooled", C4[1], "s"), ("loso", "MuRIL pooled, own source left out", C4[2], "^"),
          ("llama31_8b", "Llama-3.1-8B zero-shot", "#52514e", "D")]
x = np.arange(len(UNITS))
for k, (c, lab, col, mk) in enumerate(series):
    ax.scatter(x + (k - 1.5) * 0.15, CMP[c], s=16, color=col, marker=mk, label=lab, zorder=3, lw=0)
ax.set_xticks(x); ax.set_xticklabels(UNITS, rotation=60, ha="right"); ax.set_ylabel("AUROC (common 500-item test subset)")
ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0.45, 1.0)
for xv in range(len(UNITS) - 1):
    ax.axvline(xv + 0.5, color="#f0efec", lw=0.5, zorder=0)
ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.savefig(f"{FIG}/per_unit.pdf"); fig.savefig(f"{FIG}/per_unit.png"); plt.close(fig)

# LLM decision-rate collapse figure
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, (t, nm), col in zip(axes, LLM.items(), C4):
    d = LU[LU.model == t]
    ax.scatter(d.pos, d.says_yes, s=12, color=col, lw=0)
    ax.plot([0, 1], [0, 1], ls="--", color="#52514e", lw=0.7)
    ax.set_title(f"{nm}\nshare Yes {d.says_yes.mean():.2f}, gold {d.pos.mean():.2f}", loc="left", fontsize=7.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("positive rate (gold)"); ax.grid(color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("share predicted Yes (P>0.5)")
fig.savefig(f"{FIG}/llm_rate.pdf"); fig.savefig(f"{FIG}/llm_rate.png"); plt.close(fig)

# ------------------------------------------------------------------ script: within-unit native vs romanized, bootstrap CI
rng = np.random.default_rng(0)
scr = META.set_index("uid").script.to_dict()
def script_rows(d, name):
    d = d.copy(); d["script"] = d.uid.map(scr); out = []
    for u, g in d.groupby("unit"):
        a, b = g[g.script == "native"], g[g.script == "latin"]
        if min(len(a), len(b)) < 50 or a.y.nunique() < 2 or b.y.nunique() < 2 or min(a.y.sum(), b.y.sum()) < 10:
            continue
        diff = auc(a.y, a.p) - auc(b.y, b.p); bs = []
        for _ in range(500):
            ia, ib = a.sample(len(a), replace=True, random_state=rng.integers(1e9)), b.sample(len(b), replace=True, random_state=rng.integers(1e9))
            if ia.y.nunique() > 1 and ib.y.nunique() > 1:
                bs.append(auc(ia.y, ia.p) - auc(ib.y, ib.p))
        out.append({"model": name, "unit": u, "n_native": len(a), "n_latin": len(b), "auc_native": auc(a.y, a.p), "auc_latin": auc(b.y, b.p),
                    "diff": diff, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    return out
def scr_cached():
    rows = []
    e = enc[(enc.src == enc.tgt)].rename(columns={"tgt": "unit"})
    e = e.groupby(["uid", "unit"], as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    rows += script_rows(e, "MuRIL in-dataset")
    pl = pool[pool.src == "POOLED"].rename(columns={"tgt": "unit"}); rows += script_rows(pl, "MuRIL pooled")
    for t, nm in LLM.items():
        rows += script_rows(LP[(LP.model == t) & (LP.split == "test_eval")], nm)
    return pd.DataFrame(rows)
SC = cached("analysis/script_within.csv", scr_cached)
SC.round(3).to_csv(f"{TAB}/script_within.csv", index=False)
su = sorted(SC.unit.unique(), key=UNITS.index); names = ["MuRIL in-dataset", "MuRIL pooled", "Llama-3.1-8B", "Aya-Expanse-8B"]
fig, ax = plt.subplots(figsize=(7.0, 2.4))
for k, nm in enumerate(names):
    d = SC[SC.model == nm].set_index("unit").reindex(su)
    xs = np.arange(len(su)) + (k - 1.5) * 0.18
    ax.errorbar(xs, d["diff"], yerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="o", ms=3.5, color=C4[k], lw=0.7, capsize=1.5, label=nm)
ax.axhline(0, color="#52514e", lw=0.6)
ax.set_xticks(range(len(su))); ax.set_xticklabels(su); ax.set_ylabel("AUROC(native) - AUROC(romanized)")
ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.savefig(f"{FIG}/script.pdf"); fig.savefig(f"{FIG}/script.png"); plt.close(fig)

# ------------------------------------------------------------------ headline numbers
N["muril_in_seed_sd"] = float(M[(M.model == "muril") & (M.src == M.tgt)].groupby("tgt").auc.std().mean())
N["fe"] = FE.round(3).to_dict("records")
N["llm_summary"] = LS.round(3).to_dict()
N["tests"] = TESTS.round(4).to_dict("records")
N["prior_shift"] = ps.round(3).reset_index().to_dict("records"); N["prior_corr"] = cor.round(3).to_dict()
json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in N.items() if k not in ("fe", "prior_shift")}, indent=1, default=float))
