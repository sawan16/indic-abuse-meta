import numpy as np, pandas as pd
import statsmodels.formula.api as smf
EVAL_ONLY = ["TeluguNews-te", "HateCheck-hi", "DCM-ml"]
CONSTRUCT = {"MACD": "abuse", "Uli": "abuse", "MahaHate": "hate+offensive", "MOLD": "offensive", "DravidianCodeMix": "offensive",
             "HASOC2021": "hate+offensive", "CONSTRAINT": "hate+offensive", "Romim": "hate", "MultiLabelBangla": "hate",
             "TamilHS": "hate", "RUHSOLD": "abuse", "TeluguNews": "hate", "HateCheckHi": "hate"}


def drops(M):
    A = M[~M.tgt.isin(["TeluguNews-te", "HateCheck-hi"])]
    ind = A[A.ptype == "in-dataset"].set_index(["model", "seed", "tgt"])[["auc", "f1", "f1_oracle"]]
    B = A[A.ptype != "in-dataset"].join(ind, on=["model", "seed", "tgt"], rsuffix="_in").dropna(subset=["auc_in"]).copy()
    B["d_auc"] = B.auc_in - B.auc; B["d_f1"] = B.f1_in - B.f1; B["thr_gain"] = B.f1_oracle - B.f1
    return B


def pair_means(B, model):
    g = B[B.model == model].groupby(["src", "tgt", "src_source", "tgt_source", "src_lang", "tgt_lang", "same_source", "same_lang", "ptype"],
                                    as_index=False)[["d_auc", "d_f1", "thr_gain", "auc", "f1"]].mean()
    g["lang_shift"] = (~g.same_lang).astype(int); g["data_shift"] = (~g.same_source).astype(int)
    return g


def additive(g, y, exclude_macd_lang=False):
    if exclude_macd_lang:
        g = g[~((g.src_source == "MACD") & (g.tgt_source == "MACD"))]
    r = smf.ols(f"{y} ~ lang_shift + data_shift", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.tgt)[0]})
    ci = r.conf_int()
    return {"lang": r.params.lang_shift, "lang_lo": ci.loc["lang_shift", 0], "lang_hi": ci.loc["lang_shift", 1],
            "data": r.params.data_shift, "data_lo": ci.loc["data_shift", 0], "data_hi": ci.loc["data_shift", 1],
            "const": r.params.Intercept, "n_pairs": len(g)}


def fe_model(B, model, y="d_auc", exclude_macd=False):
    """Within-target 2x2: drop ~ target FE + lang + data + lang:data, with in-dataset rows (drop = 0) as the reference cell.
    Pair-level means over seeds; SEs clustered by target."""
    g = pair_means(B, model)
    tg = sorted(g.tgt.unique())
    zero = pd.DataFrame({"tgt": tg, "src": tg, y: 0.0, "lang_shift": 0, "data_shift": 0, "src_source": "", "tgt_source": ""})
    d = pd.concat([g, zero], ignore_index=True)
    if exclude_macd:
        d = d[~((d.src_source == "MACD") & (d.tgt_source == "MACD"))]
    r = smf.ols(f"{y} ~ C(tgt) + lang_shift * data_shift - 1", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.tgt)[0]})
    ci = r.conf_int(); out = {}
    for k, nm in (("lang_shift", "lang"), ("data_shift", "data"), ("lang_shift:data_shift", "inter")):
        out[nm] = r.params[k]; out[nm + "_lo"] = ci.loc[k, 0]; out[nm + "_hi"] = ci.loc[k, 1]
    out["n_pairs"] = len(g)
    return out
