import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
def box(x, y, w, h, t, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="#52514e", lw=0.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7, weight="bold" if bold else "normal")
# left: pipeline
ax.text(0, 38, "(a) Collection pipeline", weight="bold", fontsize=8)
steps = ["13 public sources\n(HF / GitHub)", "Harmonise to binary\n'abusive-or-hateful'", "Splits; remove\ntest/dev = train", "Cross-source\nde-duplication", "20 units\n(source x language)"]
for i, s in enumerate(steps):
    box(1, 30 - i * 7.2, 21, 5, s, "#f0efec")
    if i:
        ax.annotate("", xy=(11.5, 35 - i * 7.2 + 0.2), xytext=(11.5, 30 - (i - 1) * 7.2 - 0.4), arrowprops=dict(arrowstyle="->", lw=0.6, color="#52514e"))
# right: 2x2
ax.text(36, 38, "(b) Within-target 2x2 design (example target t = Uli-hi)", weight="bold", fontsize=8)
X0, Y0, W, H = 44, 4, 26, 13
ax.text(X0 + W / 2, Y0 + 2 * H + 3.2, "same source", ha="center"); ax.text(X0 + 1.5 * W, Y0 + 2 * H + 3.2, "different source", ha="center")
ax.text(X0 - 1.5, Y0 + 1.5 * H, "same\nlanguage", ha="right", va="center"); ax.text(X0 - 1.5, Y0 + 0.5 * H, "different\nlanguage", ha="right", va="center")
cells = [(0, 1, "In-dataset (reference)\nUli-hi -> Uli-hi\n17 pairs", "#dbe9fa"),
         (1, 1, "Dataset shift  $\\beta_D$\nMACD-hi, HASOC-hi -> Uli-hi\n30 pairs", "#fbe1d6"),
         (0, 0, "Language shift  $\\beta_L$\nUli-ta -> Uli-hi\n24 pairs (20 within MACD)", "#d6f2e7"),
         (1, 0, "Both shifts  $\\beta_L+\\beta_D+\\gamma$\nRomim-bn -> Uli-hi\n218 pairs", "#fbeccc")]
for c, r, t, fc in cells:
    box(X0 + c * W + 0.6, Y0 + r * H + 0.6, W - 1.2, H - 1.2, t, fc)
ax.text(X0 + W, 0.3, "AUROC$_{st} = \\mu_t - \\beta_L L_{st} - \\beta_D S_{st} - \\gamma L_{st}S_{st} + \\varepsilon_{st}$,  size-matched sources (2,000 items)", ha="center", fontsize=7.5)
fig.savefig("paper/out/figures/design.pdf", bbox_inches="tight"); fig.savefig("paper/out/figures/design.png", dpi=200, bbox_inches="tight")
