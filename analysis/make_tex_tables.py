"""LaTeX tables generated directly from paper/out/tables/*.csv so that numbers cannot drift."""
import os
import pandas as pd

T = "paper/out/tables"; O = "paper/tex/tables"; os.makedirs(O, exist_ok=True)
MN = {"tfidf": "TF-IDF+LR", "muril": "MuRIL", "xlmr": "XLM-R", "indicbert": "IndicBERTv2"}
ORD = ["tfidf", "muril", "xlmr", "indicbert"]
f3 = lambda x: f"{x:.3f}"
def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"
def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")

# Units
u = pd.read_csv(f"{T}/units.csv")
L = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ur": "Urdu (Roman)", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam"}
rows = []
for r in u.itertuples():
    role = "eval" if r.unit in ("HateCheck-hi", "TeluguNews-te", "DCM-ml") else "both"
    rows.append(f"{r.unit} & {L[r.lang]} & {r.construct} & {r.n_train:,} & {r.n_dev:,} & {r.n_test:,} & {r.n_test_eval:,} & {r.pos_rate:.2f} & "
                f"{r.latin_share:.2f} & {r.leak_dev + r.leak_test:,} & {r.cross_source_dups_removed:,} & {role} & {r.licence} \\\\")
open(f"{O}/units.tex", "w").write("\n".join(rows) + "\n")

# FE decomposition
fe = pd.read_csv(f"{T}/fe_decomposition.csv")
rows = []
for met, nm in (("d_auc", "AUROC"), ("d_f1", "Macro-F1")):
    for ex in (False, True):
        rows.append(r"\multicolumn{4}{l}{\emph{" + nm + (" drop, all pairs" if not ex else r" drop, excluding MACD$\to$MACD pairs") + r"}} \\")
        for m in ORD:
            r = fe[(fe.model == m) & (fe.metric == met) & (fe.exclMACD == ex)].iloc[0]
            c = lambda k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
            rows.append(f"\\quad {MN[m]} & {c('lang')} & {c('data')} & {c('inter')} \\\\")
open(f"{O}/fe.tex", "w").write("\n".join(rows) + "\n")

# F1 identity and prior-shift correction
idt = pd.read_csv(f"{T}/f1_identity.csv"); ps = pd.read_csv(f"{T}/prior_shift.csv"); cor = pd.read_csv(f"{T}/prior_shift_corr.csv").set_index("model")
rows = []
PT = [("language shift", "Language"), ("dataset shift", "Dataset"), ("both shift", "Both")]
for m in ORD:
    for k, (pt, nm) in enumerate(PT):
        a = idt[(idt.model == m) & (idt.ptype == pt)].iloc[0]; b = ps[(ps.model == m) & (ps.ptype == pt)].iloc[0]
        lead = f"\\multirow{{3}}{{*}}{{{MN[m]}}}" if k == 0 else ""
        rows.append(f"{lead} & {nm} & {a.d_f1:.3f} & {a.rank_part:.3f} & {a.thr_gain:.3f} ({100 * a.thr_gain / a.d_f1:.0f}\\%) & "
                    f"{b.f1_src:.3f} & {b.f1_prior_oracle:.3f} & {b.f1_prior_em:.3f} & {b.f1_oracle:.3f} \\\\")
    rows.append(r"\midrule" if m != ORD[-1] else "")
open(f"{O}/f1_identity.tex", "w").write("\n".join(rows) + "\n")
rows = [f"{MN[m]} & {cor.loc[m, 'slope']:.2f} & {cor.loc[m, 'r']:.2f} & {cor.loc[m, 'em_abs_err']:.2f} \\\\" for m in ORD]
open(f"{O}/prior_corr.tex", "w").write("\n".join(rows) + "\n")

# Residual regression
rr = pd.read_csv(f"{T}/residual_regression.csv")
rows = []
for sub in ("all transfer pairs", "dataset-shift pairs"):
    rows.append(r"\multicolumn{6}{l}{\emph{" + sub.capitalize() + r"}} \\")
    for m in ORD:
        r = rr[(rr.model == m) & (rr.subset == sub)].iloc[0]
        cell = lambda k: f"{r[k]:+.3f} ({pv(r[k + '_p'])})"
        rows.append(f"\\quad {MN[m]} & {int(r.n)} & {cell('construct_mismatch')} & {cell('d_pos')} & {cell('d_latin')} & {r.dR2:.3f} \\\\")
open(f"{O}/residual.tex", "w").write("\n".join(rows) + "\n")

# LLM summary
ls = pd.read_csv(f"{T}/llm_summary.csv").set_index("model")
LN = {"llama31_8b": "Llama-3.1-8B-Instruct", "aya_expanse_8b": "Aya-Expanse-8B", "qwen25_7b": "Qwen2.5-7B-Instruct"}
rows = [f"{LN[m]} & {r.auc:.3f} & {r.f1_05:.3f} & {r.f1_dev:.3f} & {r.says_yes:.3f} & {r.pos:.3f} \\\\" for m, r in ls.iterrows()]
open(f"{O}/llm.tex", "w").write("\n".join(rows) + "\n")

# Per-unit comparison
c = pd.read_csv(f"{T}/per_unit_comparison.csv")
cols = ["in_dataset", "median_transfer", "best_transfer", "pooled", "loso", "llama31_8b", "aya_expanse_8b", "qwen25_7b"]
rows = []
for r in c.itertuples():
    vals = [getattr(r, k) for k in cols]
    best = max(v for v in vals if not pd.isna(v))
    cells = [("\\textbf{" + f2(v) + "}") if (not pd.isna(v) and v == best) else f2(v) for v in vals]
    rows.append(f"{r.unit} & " + " & ".join(cells) + " \\\\")
tr = c[~c.unit.isin(["HateCheck-hi", "TeluguNews-te", "DCM-ml"])]
rows.append(r"\midrule")
rows.append("Mean, 17 training units & " + " & ".join(f2(tr[k].mean()) for k in cols) + " \\\\")
rows.append("Mean, all 20 units & " + " & ".join(("--" if k == "in_dataset" else f2(c[k].mean())) for k in cols) + " \\\\")
open(f"{O}/per_unit.tex", "w").write("\n".join(rows) + "\n")

# Paired tests
t = pd.read_csv(f"{T}/paired_tests.csv")
PN = {"pooled": "Pooled", "in_dataset": "In-dataset", "loso": "Pooled w/o own source", "llama31_8b": "Llama-3.1-8B zero-shot",
      "best_transfer": "Best single source (oracle choice)", "median_transfer": "Median single source"}
rows = []
for r in t.itertuples():
    a, b = r.pair.split(" vs ")
    rows.append(f"{PN[a]} vs {PN[b]} & {r.n} & {r.mean_diff:+.3f} & {r.wins}/{r.n} & {pv(r.p)} \\\\")
open(f"{O}/tests.tex", "w").write("\n".join(rows) + "\n")
print(os.listdir(O))

# Mean AUROC / macro-F1 by pair type (descriptive)
M = pd.read_csv("analysis/pair_metrics_final.csv")
M = M[~M.tgt.isin(["TeluguNews-te", "HateCheck-hi"])]
ind_t = set(M[M.src == M.tgt].tgt)
M = M[M.tgt.isin(ind_t)]
g = M.groupby(["model", "ptype"])[["auc", "f1"]].mean()
cnt = M[M.model == "muril"].drop_duplicates(["src", "tgt"]).groupby("ptype").size()
rows = []
for m in ORD:
    cells = [f"{g.loc[(m, p), 'auc']:.3f} / {g.loc[(m, p), 'f1']:.3f}" for p in ("in-dataset", "language shift", "dataset shift", "both shift")]
    rows.append(f"{MN[m]} & " + " & ".join(cells) + " \\\\")
open(f"{O}/ptype_means.tex", "w").write("\n".join(rows) + "\n")
print(cnt.to_dict())
