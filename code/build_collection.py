"""Build the harmonized Indic hate/offensive/abusive meta-collection.

Every unit (source x language) is returned as train / dev / test with columns:
  text, broad (0/1: offensive-or-abusive-or-hateful), orig (original label string), unit, source, lang, script, uid
Usage on Kaggle:  python build_collection.py RAW_DIR OUT_DIR
"""
import hashlib, json, os, re, sys, traceback, unicodedata
import numpy as np, pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else "collection"
SEED = 0
TEST_CAP = 1000          # evaluation items per unit (stratified sample of the test split)

# --------------------------------------------------------------------------- helpers
def norm(t):
    """Lower-case, drop mentions/URLs, keep only letters, combining marks (vowel signs) and digits."""
    t = re.sub(r"(@\w+|https?://\S+|#)", " ", str(t).lower())
    return "".join(c for c in t if unicodedata.category(c)[0] in "LMN")


def script_of(t):
    letters = [c for c in str(t) if c.isalpha()]
    if not letters:
        return "other"
    latin = sum("a" <= c.lower() <= "z" for c in letters) / len(letters)
    return "latin" if latin >= 0.5 else "native"


def strat_split(df, fracs=(0.7, 0.1, 0.2), seed=SEED):
    rng = np.random.default_rng(seed)
    parts = {"train": [], "dev": [], "test": []}
    for _, g in df.groupby("broad"):
        idx = rng.permutation(len(g)); n = len(g)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        parts["train"].append(g.iloc[idx[:a]]); parts["dev"].append(g.iloc[idx[a:b]]); parts["test"].append(g.iloc[idx[b:]])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}


def dev_from_train(sp, frac=0.1, seed=SEED):
    tr = sp["train"]
    s = strat_split(tr, (1 - frac, frac, 0.0), seed)
    sp["train"], sp["dev"] = s["train"], s["dev"]
    return sp


def frame(text, broad, orig):
    d = pd.DataFrame({"text": pd.Series(text).astype(str).values, "broad": pd.Series(broad).values, "orig": pd.Series(orig).astype(str).values})
    d = d[d.broad.notna()].copy(); d["broad"] = d.broad.astype(int)
    d["text"] = d.text.str.strip()
    return d[d.text.str.len() > 0].reset_index(drop=True)


def hf(name, cfg=None):
    from datasets import load_dataset
    return load_dataset(name, cfg) if cfg else load_dataset(name)


# --------------------------------------------------------------------------- loaders
def L_macd(lang):
    out = {}
    for sp, f in (("train", "train"), ("dev", "val"), ("test", "test")):
        d = pd.read_csv(f"{RAW}/ShareChatAI_MACD/dataset/{lang}_{f}.csv")
        out[sp] = frame(d.text, (d.label == 0).astype(int), d.label.map({0: "abusive", 1: "non-abusive"}))  # 0 = abusive (README)
    return out


def L_uli(lang):
    """broad = majority of Q1 (gendered abuse) OR Q3 (explicit/aggressive); ties -> 1 as in the shared task."""
    out = {}
    for sp, folder in (("train", "training"), ("test", "testing")):
        votes = {}
        for q in ("l1", "l3"):
            d = pd.read_csv(f"{RAW}/tattle-made_uli_dataset/{folder}/{sp}_{lang}_{q}.csv", engine="python")
            ann = [c for c in d.columns if re.match(rf"{lang}_a\d", c)]
            a = d[ann].apply(pd.to_numeric, errors="coerce")
            maj = (((a == 1).sum(1) >= (a == 0).sum(1)) & (a.notna().sum(1) > 0)).astype(int)
            votes[q] = pd.Series(maj.values, index=d.text.astype(str).values).groupby(level=0).max()
        t = sorted(set(votes["l1"].index) & set(votes["l3"].index))
        out[sp] = frame(t, [int(votes["l1"][x] or votes["l3"][x]) for x in t], [f"Q1={votes['l1'][x]},Q3={votes['l3'][x]}" for x in t])
    return dev_from_train(out)


def L_mahahate():
    out = {}
    for sp, f in (("train", "hate_train"), ("dev", "hate_valid"), ("test", "hate_test")):
        d = pd.read_excel(f"{RAW}/l3cube-pune_MarathiNLP/L3Cube-MahaHate/4-class/{f}.xlsx")
        out[sp] = frame(d.text, (d.label != "NOT").astype(int), d.label)
    return out


def L_mold():
    out = {}
    for sp in ("train", "test"):
        d = pd.read_csv(f"{RAW}/tharindudr_MOLD/MOLD_1.0/MOLD_{sp}.csv")
        out[sp] = frame(d.Tweet, (d.Class.str.lower().str.strip() == "offensive").astype(int), d.Class)
    return dev_from_train(out)


def L_dravidian(cfg):
    ds = hf("community-datasets/offenseval_dravidian", cfg)
    names = ds["train"].features["label"].names
    out = {}
    for sp, hsp in (("train", "train"), ("test", "validation")):
        d = ds[hsp].to_pandas()
        lab = d.label.map(lambda i: names[i])
        b = lab.map(lambda s: np.nan if s.startswith("not-") else (0 if s == "Not_offensive" else 1))
        out[sp] = frame(d.text, b, lab)
    return dev_from_train(out)


def L_telugu_news():
    ds = hf("mounikaiiith/Telugu-Hatespeech")
    out = {}
    for sp, hsp in (("train", "train"), ("dev", "validation"), ("test", "test")):
        d = ds[hsp].to_pandas()
        out[sp] = frame(d.Sentence, (d["Hate-Speech"].astype(str).str.lower() == "yes").astype(int), d["Hate-Speech"])
    return out


def L_hasoc_hi():
    d = hf("nikitadesai/hasoc")["train"].to_pandas()
    f = frame(d.rawtweets, (d["binary class"] == "HOF").astype(int), d["binary class"] + "/" + d["Context class"].astype(str))
    return strat_split(f)


def L_hostility_hi():
    d = hf("necrosyth/hindi-social-media-hostility-detection")["train"].to_pandas()
    labs = d["Labels Set"].astype(str).str.lower()
    b = labs.map(lambda s: 1 if ("hate" in s or "offensive" in s) else (0 if s.strip() == "non-hostile" else np.nan))
    return strat_split(frame(d.Post, b, labs))


def L_romim_bn():
    d = hf("Yanjo/bengali-hate-speech-dataset")["train"].to_pandas()
    return strat_split(frame(d.sentence, d.hate.astype(int), d.hate.astype(str) + "/" + d.category.astype(str)))


def L_mlbh_bn():
    d = hf("sumaiya-afroze/Multi-Label_Bangla_Hate_Speech_Data")["train"].to_pandas()
    return strat_split(frame(d.Comment, d["Hate/NoHate"].astype(int), d["Hate/NoHate"].astype(str)))


def L_tamil_native():
    ds = hf("krishan-CSE/Tamil_Hate_Speech")
    out = {sp: frame(ds[sp]["text"], pd.Series(ds[sp]["labels"]).astype(int), pd.Series(ds[sp]["labels"]).astype(str)) for sp in ("train", "test")}
    return dev_from_train(out)


def L_hatecheck_hi():
    d = hf("Paul/hatecheck-hindi")["test"].to_pandas()
    f = frame(d.test_case, (d.label_gold == "hateful").astype(int), d.functionality)
    return {"train": f.iloc[:0], "dev": f.iloc[:0], "test": f}


def L_ruhsold_ur():
    ds = hf("community-datasets/roman_urdu_hate_speech")
    out = {}
    for sp, hsp in (("train", "train"), ("test", "validation")):
        d = ds[hsp].to_pandas()
        out[sp] = frame(d.tweet, (d.label == 0).astype(int), d.label.map({0: "Abusive/Offensive", 1: "Normal"}))  # 0 = abusive
    return dev_from_train(out)


UNITS = [  # unit, source, lang, loader, role
    ("MACD-hi", "MACD", "hi", lambda: L_macd("hindi"), "both"),
    ("MACD-ta", "MACD", "ta", lambda: L_macd("tamil"), "both"),
    ("MACD-te", "MACD", "te", lambda: L_macd("telugu"), "both"),
    ("MACD-kn", "MACD", "kn", lambda: L_macd("kannada"), "both"),
    ("MACD-ml", "MACD", "ml", lambda: L_macd("malyalam"), "both"),
    ("Uli-hi", "Uli", "hi", lambda: L_uli("hi"), "both"),
    ("Uli-ta", "Uli", "ta", lambda: L_uli("ta"), "both"),
    ("MahaHate-mr", "MahaHate", "mr", L_mahahate, "both"),
    ("MOLD-mr", "MOLD", "mr", L_mold, "both"),
    ("DCM-ta", "DravidianCodeMix", "ta", lambda: L_dravidian("tamil"), "both"),
    ("DCM-ml", "DravidianCodeMix", "ml", lambda: L_dravidian("malayalam"), "both"),
    ("DCM-kn", "DravidianCodeMix", "kn", lambda: L_dravidian("kannada"), "both"),
    ("HASOC-hi", "HASOC2021", "hi", L_hasoc_hi, "both"),
    ("Hostility-hi", "CONSTRAINT", "hi", L_hostility_hi, "both"),
    ("Romim-bn", "Romim", "bn", L_romim_bn, "both"),
    ("MLBH-bn", "MultiLabelBangla", "bn", L_mlbh_bn, "both"),
    ("TamilHS-ta", "TamilHS", "ta", L_tamil_native, "both"),
    ("TeluguNews-te", "TeluguNews", "te", L_telugu_news, "eval"),
    ("HateCheck-hi", "HateCheckHi", "hi", L_hatecheck_hi, "eval"),
    ("RUHSOLD-ur", "RUHSOLD", "ur", L_ruhsold_ur, "both"),
]


def build():
    os.makedirs(OUT, exist_ok=True)
    log, stats, frames = [], [], []
    seen_global = set()  # normalized texts of all previously built units (cross-source de-duplication)
    for unit, src, lang, fn, role in UNITS:
        try:
            sp = fn()
        except Exception as e:
            log.append({"unit": unit, "status": f"FAILED {type(e).__name__}: {str(e)[:200]}"}); print(unit, "FAILED", e); continue
        xdup = 0
        for k in list(sp):
            m = sp[k].text.map(norm).isin(seen_global)
            xdup += int(m.sum()); sp[k] = sp[k][~m].reset_index(drop=True)
        # remove test/dev items duplicating train items of the same unit
        trn = set(sp["train"].text.map(norm))
        leak = {k: int(sp[k].text.map(norm).isin(trn).sum()) for k in ("dev", "test")}
        for k in ("dev", "test"):
            sp[k] = sp[k][~sp[k].text.map(norm).isin(trn)].reset_index(drop=True)
        # cap the evaluation set (stratified) for cross-unit evaluation cost
        te = sp["test"]
        if len(te) > TEST_CAP:
            te = te.groupby("broad", group_keys=False).apply(lambda g: g.sample(max(1, round(TEST_CAP * len(g) / len(te))), random_state=SEED))
        sp["test_eval"] = te.reset_index(drop=True)
        for k in ("train", "dev", "test", "test_eval"):
            d = sp[k].copy()
            d["unit"], d["source"], d["lang"], d["role"], d["split"] = unit, src, lang, role, k
            d["script"] = d.text.map(script_of)
            d["uid"] = [hashlib.md5(f"{unit}|{k}|{t}".encode()).hexdigest()[:12] for t in d.text]
            frames.append(d)
        s = {"unit": unit, "source": src, "lang": lang, "role": role, **{f"n_{k}": len(sp[k]) for k in ("train", "dev", "test", "test_eval")},
             "pos_rate": round(float(pd.concat([sp["train"], sp["test"]]).broad.mean()), 3),
             "latin_share": round(float((pd.concat([sp["train"], sp["test"]]).text.map(script_of) == "latin").mean()), 3),
             "leak_dev": leak["dev"], "leak_test": leak["test"]}
        s["cross_source_dups_removed"] = xdup
        for k in ("train", "dev", "test"):
            seen_global.update(sp[k].text.map(norm))
        stats.append(s); log.append({"unit": unit, "status": "ok"}); print(s, flush=True)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_parquet(f"{OUT}/collection.parquet", index=False)
    pd.DataFrame(stats).to_csv(f"{OUT}/unit_stats.csv", index=False)
    pd.DataFrame(log).to_csv(f"{OUT}/build_log.csv", index=False)
    overlap_audit(allf)
    return allf


def overlap_audit(allf):
    """Exact (normalized) duplicates across units, and near-duplicates via MinHash LSH on char 5-gram shingles."""
    a = allf[allf.split.isin(["train", "test"])].copy()
    a["n"] = a.text.map(norm)
    a = a[a.n.str.len() >= 15]
    ex = a.groupby("n").unit.apply(lambda s: sorted(set(s)))
    pairs = {}
    for us in ex:
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                pairs[(us[i], us[j])] = pairs.get((us[i], us[j]), 0) + 1
    rows = [{"unit_a": k[0], "unit_b": k[1], "exact_dups": v} for k, v in pairs.items()]
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=0.8, num_perm=64); mh = {}
        sub = a.drop_duplicates("n")
        for i, (u, t) in enumerate(zip(sub.unit, sub.n)):
            m = MinHash(num_perm=64)
            for k in range(max(1, len(t) - 4)):
                m.update(t[k:k + 5].encode())
            key = f"{u}#{i}"; lsh.insert(key, m); mh[key] = m
        near = {}
        for key, m in mh.items():
            u = key.split("#")[0]
            for o in lsh.query(m):
                v = o.split("#")[0]
                if v != u:
                    p = tuple(sorted((u, v))); near[p] = near.get(p, 0) + 1
        for p, c in near.items():
            rows.append({"unit_a": p[0], "unit_b": p[1], "near_dups_x2": c})
    except Exception as e:
        print("MinHash skipped:", e)
    pd.DataFrame(rows).to_csv(f"{OUT}/overlap_audit.csv", index=False)
    print(pd.DataFrame(rows).groupby(["unit_a", "unit_b"]).sum().to_string())


if __name__ == "__main__":
    build()
