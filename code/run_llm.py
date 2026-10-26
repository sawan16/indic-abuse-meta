"""Zero-shot LLM scoring on every unit: P(Yes) vs P(No) from next-token log-probabilities (no free generation).
Scores up to N_EVAL test_eval items and N_DEV dev items per unit (dev items allow per-unit threshold calibration)."""
import os, time
import numpy as np, pandas as pd, torch

N_EVAL, N_DEV = 500, 200
PROMPT = ("Is the following social media comment offensive, abusive or hateful (including insults, slurs, profanity, "
          "threats or attacks on a person or group)? Answer with Yes or No only.\n\nComment: {t}")


def items(c, seed=0):
    ev = c[c.split == "test_eval"].groupby("unit", group_keys=False).apply(lambda g: g.sample(min(N_EVAL, len(g)), random_state=seed))
    dv = c[c.split == "dev"].groupby("unit", group_keys=False).apply(lambda g: g.sample(min(N_DEV, len(g)), random_state=seed))
    return pd.concat([ev, dv]).reset_index(drop=True)


def run_llm(c, out, model_name, tag, token=None, bs=16):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    path = f"{out}/preds_llm_{tag}.parquet"
    it = items(c)
    if os.path.exists(path):
        done = set(pd.read_parquet(path, columns=["uid"]).uid)
        it = it[~it.uid.isin(done)]
    print(tag, len(it), "items to score", flush=True)
    if not len(it):
        return
    tok = AutoTokenizer.from_pretrained(model_name, token=token); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, quantization_config=q, dtype=torch.float16, token=token).eval()
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")})
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(t=t[:1000])}], tokenize=False, add_generation_prompt=True)
             for t in it.text]
    order = np.argsort([len(t) for t in texts])
    rows, t0 = [], time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        e = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad():
            lg = model(**e, logits_to_keep=1).logits[:, -1, :].float()
        lp = torch.log_softmax(lg, -1)
        py = torch.logsumexp(lp[:, yes], -1); pn = torch.logsumexp(lp[:, no], -1)
        mass = (torch.logsumexp(torch.stack([py, pn], -1), -1)).exp()
        p = torch.sigmoid(py - pn).cpu().numpy(); mass = mass.cpu().numpy()
        for j, i in enumerate(idx):
            r = it.iloc[i]
            rows.append({"model": tag, "uid": r.uid, "unit": r.unit, "split": r.split, "y": int(r.broad), "p": float(p[j]), "mass": float(mass[j])})
        if (b // bs) % 50 == 0:
            print(f"  {b}/{len(order)}  {(time.time()-t0)/60:.1f} min", flush=True)
            pd.DataFrame(rows).pipe(lambda d: pd.concat([pd.read_parquet(path), d]) if os.path.exists(path) else d).to_parquet(path, index=False); rows = []
    if rows:
        pd.DataFrame(rows).pipe(lambda d: pd.concat([pd.read_parquet(path), d]) if os.path.exists(path) else d).to_parquet(path, index=False)
    del model; torch.cuda.empty_cache()
