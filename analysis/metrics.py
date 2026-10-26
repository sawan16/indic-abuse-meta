"""Per (model, seed, src, tgt) metrics from saved predictions."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

GRID = np.linspace(0.05, 0.95, 37)


def pair_metrics(preds):
    rows = []
    for (m, s, src, tgt), g in preds.groupby(["model", "seed", "src", "tgt"]):
        y, p, thr = g.y.values, g.p.values, g.thr.iloc[0]
        if y.min() == y.max():
            continue
        f_src = f1_score(y, p > thr, average="macro")
        f_or = max(f1_score(y, p > t, average="macro") for t in GRID)
        rows.append({"model": m, "seed": s, "src": src, "tgt": tgt, "auc": roc_auc_score(y, p), "f1": f_src, "f1_oracle": f_or,
                     "pos": y.mean(), "says_pos": (p > thr).mean()})
    return pd.DataFrame(rows)


def label_pairs(M, units):
    """Attach source/language of src and tgt and the pair type."""
    src_of = dict(zip(units.unit, units.source)); lang_of = dict(zip(units.unit, units.lang))
    M = M.copy()
    M["src_source"] = M.src.map(src_of); M["tgt_source"] = M.tgt.map(src_of)
    M["src_lang"] = M.src.map(lang_of); M["tgt_lang"] = M.tgt.map(lang_of)
    M["same_source"] = M.src_source == M.tgt_source; M["same_lang"] = M.src_lang == M.tgt_lang
    M["ptype"] = np.select([M.src == M.tgt, M.same_source & ~M.same_lang, ~M.same_source & M.same_lang],
                           ["in-dataset", "language shift", "dataset shift"], "both shift")
    return M
