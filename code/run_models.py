"""Stage-1 experiments on the harmonized collection: TF-IDF and fine-tuned encoders, size-matched transfer matrix.
Saves per-item probabilities for every (model, seed, source unit) on every unit's test_eval set, plus the
threshold chosen on the source unit's dev set. All analyses are done afterwards from these files (CPU)."""
import json, os, time
import numpy as np, pandas as pd

N_TRAIN = 2000
MIN_POS = 100


def load(col_dir):
    c = pd.read_parquet(f"{col_dir}/collection.parquet")
    ev = c[c.split == "test_eval"].reset_index(drop=True)
    return c, ev


def train_units(c):
    us = []
    for u, g in c[(c.split == "train") & (c.role == "both")].groupby("unit"):
        exp_pos = g.broad.mean() * min(N_TRAIN, len(g))
        if exp_pos >= MIN_POS and (len(g) - g.broad.sum()) >= MIN_POS:
            us.append(u)
    return sorted(us)


def sample_train(c, unit, seed):
    g = c[(c.split == "train") & (c.unit == unit)]
    return g.sample(min(N_TRAIN, len(g)), random_state=seed)


def best_threshold(y, p):
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 37)
    return float(grid[int(np.argmax([f1_score(y, p > t, average="macro") for t in grid]))])


def save_preds(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_parquet(path, columns=["model", "seed", "src"])
    return set(map(tuple, d.drop_duplicates().values.tolist()))


def run_tfidf(c, ev, out, seeds=(0, 1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    path = f"{out}/preds_tfidf.parquet"; done = done_keys(path)
    for seed in seeds:
        for src in train_units(c):
            if ("tfidf", seed, src) in done:
                continue
            tr = sample_train(c, src, seed)
            dev = c[(c.split == "dev") & (c.unit == src)]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
            clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced").fit(vec.fit_transform(tr.text), tr.broad)
            thr = best_threshold(dev.broad.values, clf.predict_proba(vec.transform(dev.text))[:, 1]) if len(dev) else 0.5
            p = clf.predict_proba(vec.transform(ev.text))[:, 1]
            save_preds([{"model": "tfidf", "seed": seed, "src": src, "thr": thr, "uid": u, "tgt": t, "y": y, "p": float(q)}
                        for u, t, y, q in zip(ev.uid, ev.unit, ev.broad, p)], path)
            print(f"tfidf seed{seed} {src} done", flush=True)


def run_encoder(c, ev, out, model_name, tag, seeds=(0,), epochs=3, bs=32, lr=3e-5, max_len=128):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = f"{out}/preds_{tag}.parquet"; done = done_keys(path)
    tok = AutoTokenizer.from_pretrained(model_name)

    def predict(model, texts):
        ps = []
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(texts), 256):
                e = tok(list(texts[b:b + 256]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
                ps.append(torch.softmax(model(**e).logits.float(), -1)[:, 1].cpu())
        return torch.cat(ps).numpy()

    for seed in seeds:
        for src in train_units(c):
            if (tag, seed, src) in done:
                continue
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            tr = sample_train(c, src, seed)
            dev = c[(c.split == "dev") & (c.unit == src)]
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            enc = tok(list(tr.text), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            y = torch.tensor(tr.broad.values)
            n = len(y); steps = epochs * ((n + bs - 1) // bs)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
            scaler = torch.amp.GradScaler("cuda")
            model.train()
            for ep in range(epochs):
                perm = torch.randperm(n)
                for b in range(0, n, bs):
                    idx = perm[b:b + bs]
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(input_ids=enc["input_ids"][idx].cuda(), attention_mask=enc["attention_mask"][idx].cuda(),
                                     labels=y[idx].cuda()).loss
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                model.train()
            thr = best_threshold(dev.broad.values, predict(model, dev.text.values)) if len(dev) else 0.5
            p = predict(model, ev.text.values)
            save_preds([{"model": tag, "seed": seed, "src": src, "thr": thr, "uid": u, "tgt": t, "y": yy, "p": float(q)}
                        for u, t, yy, q in zip(ev.uid, ev.unit, ev.broad, p)], path)
            print(f"{tag} seed{seed} {src}: {(time.time()-t0)/60:.1f} min, thr {thr:.2f}", flush=True)
            del model; torch.cuda.empty_cache()



def run_pooled(c, ev, out, model_name, tag, per_unit=2000, seed=0, leave_out=None, epochs=2, bs=32, lr=3e-5, max_len=128):
    """Train one model on the union of size-matched samples from all training units (optionally leaving one SOURCE out).
    Saved with src = 'POOLED' or 'LOSO:<source>'."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "POOLED" if leave_out is None else f"LOSO:{leave_out}"
    path = f"{out}/preds_{tag}_pooled.parquet"
    if (tag, seed, name) in done_keys(path):
        return
    t0 = time.time()
    units = [u for u in train_units(c) if leave_out is None or c[c.unit == u].source.iloc[0] != leave_out]
    tr = pd.concat([sample_train(c, u, seed) for u in units]).sample(frac=1, random_state=seed)
    dev = pd.concat([c[(c.split == "dev") & (c.unit == u)].head(300) for u in units])
    tok = AutoTokenizer.from_pretrained(model_name)
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    y = torch.tensor(tr.broad.values); texts = list(tr.text); n = len(y)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler("cuda"); model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs].tolist()
            e = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**e, labels=y[idx].cuda()).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

    def predict(tx):
        ps = []; model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(tx), 256):
                e = tok(list(tx[b:b + 256]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
                ps.append(torch.softmax(model(**e).logits.float(), -1)[:, 1].cpu())
        return torch.cat(ps).numpy()
    thr = best_threshold(dev.broad.values, predict(dev.text.values))
    p = predict(ev.text.values)
    save_preds([{"model": tag, "seed": seed, "src": name, "thr": thr, "uid": u, "tgt": t, "y": yy, "p": float(q)}
                for u, t, yy, q in zip(ev.uid, ev.unit, ev.broad, p)], path)
    print(f"{tag} {name}: {len(tr)} train items, {(time.time()-t0)/60:.1f} min", flush=True)
    del model; torch.cuda.empty_cache()
