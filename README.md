# IndicAbuseMeta

This repository holds the code, the per-item manifest and all model predictions for the paper *Language Shift or Dataset Shift? A De-duplicated Meta-Collection of Indic Abusive-Language Datasets and a Decomposition of Cross-Dataset Transfer* by Sawan Rai.

IndicAbuseMeta harmonises **20 units** (source dataset × language) from **13 public datasets** in eight languages into one binary *abusive-or-hateful* label. The languages are Hindi, Marathi, Bengali, Roman Urdu, Tamil, Telugu, Kannada and Malayalam.

Before cross-unit evaluation, the build removes two kinds of duplicate:
- test and dev items that duplicate a training item of the same unit;
- items that duplicate an item of another unit.

## What is here

| Path | Contents |
|---|---|
| `code/build_collection.py` | Downloads every source from its official GitHub or Hugging Face location, harmonises the labels, splits and de-duplicates, and runs the overlap audit |
| `code/run_models.py` | TF-IDF+LR, fine-tuned encoders (MuRIL, XLM-R, IndicBERTv2), pooled and leave-one-source-out training |
| `code/run_llm.py` | Zero-shot LLM scoring from next-token Yes/No probabilities |
| `notebooks/` | The four Kaggle notebooks that were run, each on one T4 GPU |
| `data/manifest.csv.gz` | One row per item: `uid`, `unit`, `source`, `lang`, `construct`, `role`, `split`, `original_label`, `label`, `script`. There is **no text** in this file. |
| `data/unit_stats.csv`, `data/overlap_audit_*.csv` | Unit statistics and the audit of overlaps between units |
| `s2/out`, `s3a/out`, `s3b/out` | Per-item predictions (`uid`, target unit, gold label, score) for every model, seed and source |
| `analysis/` | Scripts that compute every table and figure in the paper from the predictions (CPU only) |

## Reproducing the paper's numbers

```bash
pip install pandas pyarrow scikit-learn statsmodels scipy matplotlib
python analysis/make_all.py          # writes paper/out/tables/*.csv, paper/out/figures/*, paper/out/numbers.json
python analysis/make_tex_tables.py   # writes the LaTeX tables
python analysis/fig_design.py        # the design figure
```

Run the scripts from the repository root. On a laptop CPU they take about 10–15 minutes. Most of that time goes to the per-pair metrics, which are cached in `analysis/` after the first run.

## Rebuilding the collection

The texts are not redistributed. Four sources allow only non-commercial use: MACD and MahaHate (CC BY-NC-SA 4.0), and HASOC 2021 Hindi and Multi-label Bangla (CC BY-NC 4.0).

To rebuild, first clone the GitHub sources into a folder `raw/`, naming each folder `owner_repo` (for example `raw/ShareChatAI_MACD`):
- `ShareChatAI/MACD`
- `tattle-made/uli_dataset`
- `l3cube-pune/MarathiNLP`
- `tharindudr/MOLD`

Then run:

```bash
python code/build_collection.py raw collection
```

The remaining sources are downloaded from the Hugging Face Hub. Check the result against `data/manifest.csv.gz`. Each `uid` is the first 12 hex characters of `md5(unit|split|text)`.

## Licences

- Code: MIT.
- Manifest and predictions: CC BY-NC-SA 4.0. They are derived from sources that carry non-commercial and share-alike terms.
- Each source dataset keeps its own licence (Table 2 of the paper). Please cite the original dataset papers when you use their data.

## Label notes

- MACD codes the positive class as **0** (`abusive`), as its README states.
- The Hugging Face release of RUHSOLD also codes the positive class as **0** (`Abusive/Offensive`).
- MOLD 2.0 is not used, because all 260 offensive test items appear in its training split. MOLD 1.0 is used instead.
