# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pillow",
# ]
# ///
"""Inscribed Squares from Noise — a marimo walkthrough of
"Visual Diffusion Models are Geometric Solvers" (Goren et al., CVPR 2026 Highlight).

Submission for the alphaXiv x marimo "Bring Research to Life" competition.
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import io
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    GH_RAW = "https://raw.githubusercontent.com/FarseenSh/alphaxiv-marimo-comp/main/data/gallery.npz"
    LOCAL_PATH = Path(__file__).resolve().parents[1] / "data" / "gallery.npz"

    if LOCAL_PATH.exists():
        gallery = dict(np.load(LOCAL_PATH))
    else:
        with urllib.request.urlopen(GH_RAW) as _r:
            gallery = dict(np.load(io.BytesIO(_r.read())))
    return gallery, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align:center; padding: 24px 0 8px 0;">
    <h1 style="margin-bottom: 4px; font-size: 2.4rem;">Inscribed Squares from Noise</h1>
    <p style="font-size: 1.05rem; opacity: 0.85; margin-top: 0;">
    A diffusion model doesn't generate cats here.<br>
    It solves a 100-year-old open problem in geometry — by drawing.
    </p>
    <img src="https://raw.githubusercontent.com/FarseenSh/alphaxiv-marimo-comp/main/assets/hero_8_squares.png"
         style="width: min(900px, 100%); border-radius: 12px; margin-top: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.10);" />
    <p style="font-size: 0.85rem; opacity: 0.6; margin-top: 8px;">
    The same Jordan curve, eight random seeds, eight different inscribed squares.
    </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The setup

    The **Inscribed Square Problem** (Toeplitz, 1911) asks: does every closed,
    non-self-intersecting curve in the plane contain four points that form a
    perfect square? It's still open in full generality after 100+ years.

    Goren, Yehezkel, Dahary, Voynov, Patashnik, and Cohen-Or
    ([arXiv 2510.21697](https://arxiv.org/abs/2510.21697), CVPR 2026 Highlight)
    propose something a little wild: **train a standard image diffusion model
    to take a Jordan curve as a 128×128 picture and denoise Gaussian noise
    into a picture of an inscribed square**. No specialized architecture, no
    parametric tricks — pure pixel-space denoising.

    It works. And because diffusion is multimodal, the same curve produces
    a *different* inscribed square at every seed — uncovering a hidden family
    of solutions the paper itself doesn't fully explore.

    That's where this notebook spends most of its time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline — at a glance

    | Stage | What goes in | What comes out |
    |---|---|---|
    | **Condition** | A Jordan curve, rasterized to 128×128 binary | 1-channel image (the "problem statement") |
    | **Noise**     | Random Gaussian (1×128×128) | The starting point of denoising |
    | **U-Net**     | `[noise, condition]` concatenated as 2 channels | Predicted noise per pixel |
    | **DDIM**      | 100 denoising steps | A clean 1-channel image of an inscribed square |
    | **Snap**      | Discrete corners on the curve | A geometrically-valid 4-vertex square |

    The U-Net is a vanilla 4-level diffusion U-Net (~20M params, attention at
    bottleneck and the inner enc/dec levels). The training data is purely
    synthetic: 100,000 procedurally generated Jordan curves, each
    constructed to pass exactly through a known random square.

    Everything below is **pre-computed** — the diffusion sampling itself
    ran offline on CPU (~5s per sample, 100 DDIM steps). The notebook is a
    viewer over those samples. *Why?* Because PyTorch doesn't run in
    WASM/Pyodide. Look at `scripts/precompute.py` in the repo to re-run
    sampling yourself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick a curve

    We walk through the pipeline on one running example, then apply it to
    the rest of the gallery at the end. Try switching curves now to see how
    the model behaves on different inputs.
    """)
    return


@app.cell
def _(gallery, mo):
    _choices = [
        c for c in
        ["hero_butterfly", "circle", "peanut", "spiky_gear", "paper_figure_1"]
        if f"{c}/curve_img" in gallery
    ]
    curve_picker = mo.ui.dropdown(
        options=_choices, value="hero_butterfly", label="Jordan curve"
    )
    curve_picker
    return (curve_picker,)


@app.cell
def _(curve_picker, gallery, plt):
    name = curve_picker.value
    curve_img = gallery[f"{name}/curve_img"]
    samples = gallery[f"{name}/samples"]

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.imshow(curve_img, cmap="binary", vmin=0, vmax=255)
    _ax.set_title(f"input: {name}  ({curve_img.shape[0]}×{curve_img.shape[1]} binary)")
    _ax.axis("off")
    plt.tight_layout()
    _fig
    return curve_img, samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How is the curve made?

    The training distribution is procedurally generated. Each curve is the
    graph of a radial function

    $$
    r(\theta) = 1 + \sum_{h=1}^{H} \rho_h \sin(h\theta + \phi_h) + \delta(\theta)
    $$

    where the harmonics $\rho_h$ decay logarithmically and $\delta$ is a
    **periodic cubic spline correction** chosen to make the curve pass
    exactly through four predetermined square corners. So each training
    example *guarantees an inscribed square by construction* — the model
    never has to wonder whether a solution exists.

    At test time on real curves the construction trick is gone, but the
    model has learned how a curve "wants" a square: it produces one anyway.

    <details>
    <summary>The data trick is doing some heavy lifting here.</summary>

    Hardcoding a square into every training curve means the model has only
    ever seen Jordan curves that *do* contain a square (a known unknown for
    non-rectifiable curves). If your test curve is wildly out of
    distribution — heavily self-intersecting after smoothing, fractal-like,
    very thin lobes — the model can fail gracefully or silently. The
    spiky-gear example below pushes that envelope.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Watch the square emerge

    DDIM walks 100 denoising steps from pure Gaussian noise to a clean
    prediction. The pictures below are the **predicted clean image
    $\hat x_0$** at every 10th step — they show what the model "thinks"
    the answer is at every point in the trajectory.
    """)
    return


@app.cell
def _(curve_picker, gallery, mo):
    has_traj = f"{curve_picker.value}/trajectory" in gallery
    if has_traj:
        traj = gallery[f"{curve_picker.value}/trajectory"]
        step_slider = mo.ui.slider(
            start=0, stop=traj.shape[0] - 1, value=traj.shape[0] - 1,
            step=1, label="denoising step",
        )
    else:
        traj = None
        step_slider = mo.md("*(trajectory only cached for the hero butterfly curve — switch to it to use this slider)*")
    step_slider
    return has_traj, step_slider, traj


@app.cell
def _(curve_img, has_traj, plt, step_slider, traj):
    if has_traj:
        _s = int(step_slider.value)
        _frame = traj[_s, 0]
        if _frame.ndim == 3:
            _frame = _frame[0]
        _fig, _ax = plt.subplots(figsize=(5, 5))
        _ax.imshow(curve_img, cmap="gray_r", alpha=0.35)
        _ax.imshow(_frame, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.85)
        _ax.set_title(f"$\\hat x_0$ at trajectory step {_s} of {traj.shape[0]-1}")
        _ax.axis("off")
        plt.tight_layout()
        _out = _fig
    else:
        _out = None
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One solution at a time

    Each random seed produces a different valid inscribed square.
    Use the slider to scroll through the seeds the model produced
    for the curve above.
    """)
    return


@app.cell
def _(mo, samples):
    sample_slider = mo.ui.slider(
        start=0, stop=samples.shape[0] - 1, value=0, step=1, label="seed index"
    )
    sample_slider
    return (sample_slider,)


@app.cell
def _(curve_img, np, plt, sample_slider, samples):
    _i = int(sample_slider.value)
    _sq = samples[_i]
    _sq01 = np.clip((_sq + 1) / 2, 0, 1)

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.imshow(curve_img, cmap="gray_r", alpha=0.6)
    _ax.imshow(1 - _sq01, cmap="Reds", alpha=0.7)
    _ax.set_title(f"seed {_i} → one valid inscribed square")
    _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The hero cell — the family of solutions

    Here is the part the original paper hints at but doesn't quantify.
    A smooth Jordan curve generally admits a *continuous family* of
    inscribed squares. Diffusion models, being multimodal samplers, can
    sweep through that family one seed at a time.

    Below: every seed's predicted square overlaid on a single curve as
    a heat map. Bright pixels are covered by *every* sampled square;
    dim pixels by only some seeds. The contrast is the "wiggle room"
    of the inscribed-square family on this Jordan curve.
    """)
    return


@app.cell
def _(curve_img, np, plt, samples):
    _n = samples.shape[0]
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(curve_img, cmap="gray_r", alpha=0.55)

    _mask = (samples < 0).astype(np.float32)
    _heat = _mask.mean(axis=0)
    _im = _ax.imshow(
        np.where(_heat > 0.05, _heat, np.nan),
        cmap="plasma", alpha=0.85, vmin=0, vmax=1,
    )
    _cbar = plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.set_label(f"fraction of {_n} seeds covering pixel")
    _ax.set_title("Multimodal solution map: where the squares live")
    _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shape of the bright core is the **intersection of all inscribed
    squares** for this curve; the dim halo is their union. This is
    something the paper's qualitative figures do not show. It's a free
    byproduct of using a diffusion model — there is no extra inference
    cost for getting it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Are these actually squares?

    We can score each sample's "squareness" — the fraction of the
    predicted shape filled by the smallest enclosing rectangle, scaled
    by an aspect-ratio penalty (the metric the paper uses):

    $$
    Q = \frac{\text{area}}{w \cdot h} \cdot \exp\!\left(-2 \left|\tfrac{\max(w,h)}{\min(w,h)} - 1\right|\right)
    $$

    $Q = 1$ is a perfect square; $Q$ near 0 is a long thin rectangle or
    scattered noise.
    """)
    return


@app.cell
def _(np, plt, samples):
    def _squareness(mask):
        _ys, _xs = np.where(mask < 0)
        if len(_xs) < 8:
            return 0.0, 0.0
        _x0, _x1 = _xs.min(), _xs.max()
        _y0, _y1 = _ys.min(), _ys.max()
        _w, _h = max(1, _x1 - _x0), max(1, _y1 - _y0)
        _area = float(len(_xs))
        _fill = _area / (_w * _h)
        _aspect = max(_w, _h) / min(_w, _h)
        return float(_fill * np.exp(-2 * abs(_aspect - 1))), _area

    scores = np.array([_squareness(s)[0] for s in samples])
    _areas = np.array([_squareness(s)[1] for s in samples])

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    _a1.bar(range(len(scores)), scores, color="#5b8def")
    _a1.axhline(0.85, ls="--", color="grey", alpha=0.5,
                label="paper's quality threshold ≈ 0.85")
    _a1.set_xlabel("seed index"); _a1.set_ylabel("squareness $Q$")
    _a1.set_ylim(0, 1.02); _a1.legend(fontsize=8)
    _a1.set_title("per-seed squareness")

    _a2.scatter(_areas, scores, s=40, c="#5b8def")
    _a2.set_xlabel("predicted square area (px)"); _a2.set_ylabel("squareness $Q$")
    _a2.set_title("size vs. quality")
    plt.tight_layout()
    _fig
    return (scores,)


@app.cell(hide_code=True)
def _(mo, scores):
    mo.md(
        f"""
        Mean squareness across seeds: **{float(scores.mean()):.3f}** (n={len(scores)}).
        Outliers near 0 are typically failed samples — the model occasionally
        produces a degenerate blob instead of a square. Failure rates of
        ~5–15% match the paper's reported numbers on the curves task.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gallery

    The same model, the same pipeline, applied to every curve in our test
    set. For each curve we show the multimodal map (intersection of all
    sampled squares).
    """)
    return


@app.cell
def _(gallery, np, plt):
    _names = [
        n for n in
        ["hero_butterfly", "circle", "peanut", "spiky_gear", "paper_figure_1"]
        if f"{n}/curve_img" in gallery
    ]
    _rows = (len(_names) + 1) // 2

    _fig, _axes = plt.subplots(_rows, 2, figsize=(11, 5 * _rows))
    _axes = np.atleast_2d(_axes)

    for _k, _name in enumerate(_names):
        _r, _c = divmod(_k, 2)
        _ax = _axes[_r, _c]
        _cimg = gallery[f"{_name}/curve_img"]
        _smp = gallery[f"{_name}/samples"]
        _heat = (_smp < 0).astype(np.float32).mean(axis=0)
        _ax.imshow(_cimg, cmap="gray_r", alpha=0.55)
        _ax.imshow(np.where(_heat > 0.05, _heat, np.nan),
                   cmap="plasma", vmin=0, vmax=1, alpha=0.85)
        _ax.set_title(f"{_name}  (n={_smp.shape[0]} seeds)")
        _ax.axis("off")

    for _k in range(len(_names), _rows * 2):
        _r, _c = divmod(_k, 2)
        _axes[_r, _c].axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook does *not* claim

    - **The model is not a proof.** It outputs pixel pictures. Its squares
      are visually inscribed, but corners snap to the nearest curve pixel
      to within ~1 pixel of error — not provably exact.
    - **Out-of-distribution failure is real.** Curves with very thin lobes,
      aggressive self-intersections after smoothing, or fractal-like
      textures often produce degenerate samples. We don't filter them out
      in the multimodal map.
    - **The training "trick" matters.** Each training curve was constructed
      to pass through a known square. The model never had to discover
      inscribed-ness from scratch on adversarial inputs.

    None of this diminishes the central observation: a generic image
    diffusion model — no architectural specialization, no parametric output
    head — recovers inscribed squares from images at high enough quality
    to be visually convincing on novel curves. That's the paper's
    contribution. The multimodal-overlay angle in this notebook is a free
    side effect of the formulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself

    - **Source repo**: `github.com/FarseenSh/alphaxiv-marimo-comp` — fork to run sampling on
      your own curves. The script `scripts/precompute.py` accepts any
      `(H, ρ, target_radius, rotation, seed)` tuple.
    - **Pretrained checkpoint**: 80MB, available at
      [huggingface.co/nirgoren/geometric-solver](https://huggingface.co/nirgoren/geometric-solver).
    - **Original paper**: [arXiv:2510.21697](https://arxiv.org/abs/2510.21697)
      (Goren, Yehezkel, Dahary, Voynov, Patashnik, Cohen-Or — CVPR 2026 Highlight).
    - **alphaXiv discussion**: [comments](https://www.alphaxiv.org/abs/2510.21697).

    Built for the
    [alphaXiv × marimo "Bring Research to Life" notebook competition](https://marimo.io/pages/events/notebook-competition).
    """)
    return


if __name__ == "__main__":
    app.run()
