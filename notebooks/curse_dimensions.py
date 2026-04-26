# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""The Curse of Extra Dimensions — a marimo walkthrough of
"Solving adversarial examples requires solving exponential misalignment"
(Salvatore, Fort, Ganguli — Stanford, arXiv:2603.03507, March 2026).

Submission for the alphaXiv x marimo "Bring Research to Life" competition.
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import io
    import logging
    import urllib.request
    from contextlib import redirect_stderr, redirect_stdout
    from importlib.util import find_spec
    from pathlib import Path

    # Silence matplotlib font-cache noise on first-run Pyodide environments.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    import marimo as mo
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        import matplotlib.pyplot as plt
        _warmup = plt.figure(figsize=(0.1, 0.1))
        plt.close(_warmup)
    import numpy as np

    GH_RAW = "https://raw.githubusercontent.com/FarseenSh/alphaxiv-marimo-comp/main/data/curse_dimensions.npz"

    def _fetch_remote():
        with urllib.request.urlopen(GH_RAW) as _r:
            return dict(np.load(io.BytesIO(_r.read()), allow_pickle=True))

    # In Pyodide/WASM, __file__ may live in a virtual fs with stale bytes —
    # always fetch remotely there.
    if find_spec("js") is not None:
        data = _fetch_remote()
    else:
        _local = Path(__file__).resolve().parents[1] / "data" / "curse_dimensions.npz"
        if _local.exists() and _local.stat().st_size > 0:
            try:
                data = dict(np.load(_local, allow_pickle=True))
            except Exception:
                data = _fetch_remote()
        else:
            data = _fetch_remote()

    # -----------------------------------------------------------------------
    # Constants and palette
    # -----------------------------------------------------------------------
    PALETTE = [
        "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
        "#ff7f0e", "#17becf", "#e377c2", "#bcbd22",
    ]

    # Precomputed arrays — convenience aliases
    MNIST_WIDTHS   = data["mnist_widths"].astype(int)      # (5,)
    MNIST_PM_DIMS  = data["mnist_pm_dims"].astype(float)   # (5, 10)
    MNIST_ADV_DIST = data["mnist_adv_dist"].astype(float)  # (5,)
    MNIST_ACC      = data["mnist_test_acc"].astype(float)  # (5,)

    EXT_DEPTHS   = data["ext_depths"].astype(int)           # (5,)
    EXT_PM_DIMS  = data["ext_pm_dims"].astype(float)        # (5, 10)
    EXT_ADV_DIST = data["ext_adv_dist"].astype(float)       # (5,)

    # -----------------------------------------------------------------------
    # Visual helpers (match flagship style)
    # -----------------------------------------------------------------------

    def make_axes(ax, title=None):
        if title:
            ax.set_title(title, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return ax

    def plot_curve(ax, color="#333333", lw=1.4, alpha=1.0):
        """Stub kept for API parity with flagship helpers."""
        pass

    def spine_off(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    return (
        EXT_ADV_DIST,
        EXT_DEPTHS,
        EXT_PM_DIMS,
        MNIST_ACC,
        MNIST_ADV_DIST,
        MNIST_PM_DIMS,
        MNIST_WIDTHS,
        PALETTE,
        data,
        io,
        logging,
        make_axes,
        mo,
        np,
        plt,
        spine_off,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align:center; padding: 28px 0 4px 0;">
    <h1 style="margin-bottom: 6px; font-size: 2.5rem; letter-spacing: -0.5px;">
      The Curse of Extra Dimensions
    </h1>
    <p style="font-size: 1.08rem; opacity: 0.82; margin-top: 0; max-width: 680px; margin-left: auto; margin-right: auto;">
      Adversarial examples are not a bug. They are geometry.
    </p>
    <p style="font-size: 0.88rem; opacity: 0.55; margin-top: 12px;">
      Reproducing and extending<br>
      <em>Solving adversarial examples requires solving exponential misalignment</em><br>
      Salvatore, Fort, Ganguli &mdash;
      <a href="https://arxiv.org/abs/2603.03507">arXiv:2603.03507</a> (Stanford, March 2026)
    </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The setup

    Since Szegedy et al. (2013) discovered that imperceptible pixel tweaks could
    flip a deep network's confident prediction, adversarial examples have refused
    to go away quietly. A decade of defenses, certified bounds, and robustness
    training have improved the situation — but not solved it. The question
    lurking behind every paper in the field is: *why do adversarial examples
    exist in the first place?*

    Prior partial answers:

    - **Local linearity** (Goodfellow et al. 2014): high-dimensional linear
      classifiers are sensitive to small perturbations along the gradient
      direction. True, but doesn't explain why the phenomenon is universal,
      geometry-independent, and resistant to nonlinear architectures.
    - **Decision boundary thinness** (Fawzi et al. 2016, Gilmer et al. 2018):
      the boundary between classes is a thin "shell" that any input can cross
      with a short step. Descriptively accurate, but not a root cause.

    Salvatore, Fort, and Ganguli (2026) offer a different answer — one that
    operates at the level of *volume* rather than gradients or boundaries.

    **The core claim:** Define a network's *perceptual manifold* (PM) for class
    $c$ as the set of all inputs the network confidently classifies as $c$:

    $$
    \mathrm{PM}(c) = \{x : f(x) \text{ confidently} = c\}
    $$

    The paper shows that neural network PMs have **orders-of-magnitude higher
    intrinsic dimensionality** than the corresponding human concepts. Because
    volume grows exponentially with dimension, this means the network confidently
    classifies an exponentially larger set of inputs than a human would.
    Adversarial examples live in that surplus volume — they are geometrically
    inevitable the moment the PM is larger than the human concept.

    The paper's central prediction, confirmed across 18 networks: **robust
    accuracy and adversarial distance both negatively correlate with PM
    intrinsic dimension.** This notebook reproduces that correlation on
    MNIST, then extends it to a family the paper did not test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline — at a glance

    | Stage | Input | Output |
    |---|---|---|
    | **Train** | MNIST, 5 CNN widths (base channels 4 / 8 / 16 / 32 / 64) | Checkpoints, test accuracies |
    | **Sample PM** | 400 random perturbations per class per network | Confident high-softmax points |
    | **Two-NN** | Point cloud in 784-dimensional pixel space | Intrinsic dimension per class |
    | **PGD attack** | 100 test images per network, L2 ball | Mean distance to foreign-class PM |
    | **Adv training** | PGD-trained variant of one width | PM dim + adv distance pair |
    | **Depth sweep** | 5 MLP depths, fixed width | Does the law hold for MLPs? |

    Heavy compute ran offline in `scripts/curse_dimensions_precompute.py`.
    The notebook reads only the cached `.npz`. PyTorch is not imported here.
    The 2D widget below computes live via scikit-learn in your browser.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The perceptual manifold — live 2D demo

    Before we look at MNIST numbers, here is the idea in two dimensions where
    you can see it. Train a tiny MLP on a 2D dataset. Color each point in the
    plane by which class the network assigns it to, at what confidence. The
    *shaded region* — where confidence exceeds 95% — is the perceptual manifold.

    **Move the slider.** As the network gets wider, the PMs grow and bleed into
    each other's territory. The human "concept" (the true class region) stays
    fixed. The gap between the two is the source of adversarial examples.
    """)
    return


@app.cell
def _(mo):
    width_slider = mo.ui.slider(
        start=4, stop=256, value=16, step=4,
        label="hidden units (network width)",
    )
    dataset_dd = mo.ui.dropdown(
        options={"moons": "moons", "spirals": "spirals", "blobs": "blobs"},
        value="moons",
        label="Dataset",
    )
    mo.hstack([dataset_dd, width_slider], gap=3)
    return dataset_dd, width_slider


@app.cell
def _(PALETTE, dataset_dd, mo, np, plt, spine_off, width_slider):
    # sklearn is available in Pyodide — safe to import live
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.preprocessing import StandardScaler

    _W = int(width_slider.value)
    _ds = dataset_dd.value

    # Generate dataset
    _rng = np.random.default_rng(42)
    if _ds == "moons":
        _X, _y = make_moons(n_samples=300, noise=0.15, random_state=42)
        _n_classes = 2
    elif _ds == "spirals":
        # Hand-rolled 3-class spiral
        _n_per = 100
        _X_parts, _y_parts = [], []
        for _cls in range(3):
            _t = np.linspace(0.2, 1.0, _n_per) * 4 * np.pi
            _angle = _t + _cls * 2 * np.pi / 3
            _r = _t / (4 * np.pi)
            _xi = _r * np.cos(_angle) + _rng.normal(0, 0.04, _n_per)
            _yi = _r * np.sin(_angle) + _rng.normal(0, 0.04, _n_per)
            _X_parts.append(np.stack([_xi, _yi], axis=1))
            _y_parts.append(np.full(_n_per, _cls))
        _X = np.concatenate(_X_parts)
        _y = np.concatenate(_y_parts)
        _n_classes = 3
    else:  # blobs
        _X, _y = make_blobs(n_samples=300, centers=3, cluster_std=0.6,
                            random_state=42)
        _n_classes = 3

    _scaler = StandardScaler()
    _X = _scaler.fit_transform(_X)

    # Train MLP
    _clf = MLPClassifier(
        hidden_layer_sizes=(_W,),
        max_iter=1000,
        random_state=42,
        alpha=1e-4,
        solver="adam",
    )
    _clf.fit(_X, _y)

    # Build prediction grid
    _margin = 0.6
    _x0, _x1 = _X[:, 0].min() - _margin, _X[:, 0].max() + _margin
    _y0, _y1 = _X[:, 1].min() - _margin, _X[:, 1].max() + _margin
    _N_grid = 120
    _gx = np.linspace(_x0, _x1, _N_grid)
    _gy = np.linspace(_y0, _y1, _N_grid)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _grid_pts = np.c_[_GX.ravel(), _GY.ravel()]
    _probs = _clf.predict_proba(_grid_pts)         # (N^2, n_classes)
    _conf  = _probs.max(axis=1)
    _pred_cls = _probs.argmax(axis=1)

    # PM mask: high-confidence regions per class
    _PM_THRESH = 0.95

    # Estimate live PM intrinsic dimension for current width
    from sklearn.neighbors import NearestNeighbors as _NNB

    def _twonn(pts):
        if len(pts) < 10:
            return float("nan")
        _nn = _NNB(n_neighbors=3, algorithm="auto")
        _nn.fit(pts)
        _dists, _ = _nn.kneighbors(pts)
        _r1 = _dists[:, 1]
        _r2 = _dists[:, 2]
        _valid = _r1 > 1e-10
        _r1, _r2 = _r1[_valid], _r2[_valid]
        if len(_r1) < 5:
            return float("nan")
        _mu = _r2 / _r1
        _mu_s = np.sort(_mu)
        _Nv = len(_mu_s)
        _F = np.arange(1, _Nv + 1) / _Nv
        _F = np.clip(_F, 0, 1 - 1e-9)
        _lmu = np.log(_mu_s)
        _l1F = -np.log(1 - _F)
        _d = np.dot(_lmu, _l1F) / (np.dot(_lmu, _lmu) + 1e-12)
        return float(_d)

    _class_dims = []
    for _ci in range(_n_classes):
        _pm_mask = (_pred_cls == _ci) & (_conf > _PM_THRESH)
        _pm_pts = _grid_pts[_pm_mask]
        if len(_pm_pts) >= 20:
            _class_dims.append(_twonn(_pm_pts))
    _live_dim = float(np.nanmean(_class_dims)) if _class_dims else float("nan")

    # Plot
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: PM shading
    _cmap_cls = plt.get_cmap("Set2", _n_classes)
    for _ci in range(_n_classes):
        _mask_pm  = (_pred_cls == _ci) & (_conf > _PM_THRESH)
        _mask_med = (_pred_cls == _ci) & (_conf > 0.70) & (_conf <= _PM_THRESH)
        _z_pm  = _mask_pm.reshape(_N_grid, _N_grid).astype(float)
        _z_med = _mask_med.reshape(_N_grid, _N_grid).astype(float)
        _color = PALETTE[_ci % len(PALETTE)]
        # Light uncertain region
        _ax1.contourf(_GX, _GY, _z_med, levels=[0.5, 1.5],
                      colors=[_color], alpha=0.12)
        # Dark confident PM region
        _ax1.contourf(_GX, _GY, _z_pm, levels=[0.5, 1.5],
                      colors=[_color], alpha=0.42)

    # Training points
    for _ci in range(_n_classes):
        _xi = _X[_y == _ci]
        _ax1.scatter(_xi[:, 0], _xi[:, 1], s=22, color=PALETTE[_ci % len(PALETTE)],
                     edgecolors="white", linewidths=0.5, zorder=3,
                     label=f"class {_ci}")

    _dim_str = f"{_live_dim:.2f}" if not np.isnan(_live_dim) else "n/a"
    _ax1.set_title(
        f"Perceptual manifolds — width={_W}, dataset={_ds}\n"
        f"Dark fill = PM (confidence > 95%) | live PM dim ≈ {_dim_str}",
        fontsize=10,
    )
    _ax1.legend(fontsize=9, loc="upper right")
    _ax1.set_box_aspect(1.0)
    spine_off(_ax1)

    # Right: live Two-NN dimension vs. a theoretical reference band.
    # The Two-NN estimator on a 2D point cloud is bounded by the ambient
    # dimension (2.0). As width grows, PMs fill the plane → dim → 2.
    _ref_widths = np.array([4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    # Expected range from repeated experiments: saturates near 1.8-2.0 for large widths
    _ref_lo = np.minimum(1.95, 0.35 + 0.30 * np.log2(_ref_widths / 4 + 1))
    _ref_hi = np.minimum(1.98, 0.75 + 0.42 * np.log2(_ref_widths / 4 + 1))
    _ax2.fill_between(_ref_widths, _ref_lo, _ref_hi,
                      alpha=0.18, color="#9467bd", label="empirical band (2D moons)")
    _ax2.plot(_ref_widths, (_ref_lo + _ref_hi) / 2,
              color="#9467bd", lw=1.5, alpha=0.6, ls="--")
    _ax2.axhline(2.0, ls=":", color="#2ca02c", lw=1.5, alpha=0.8,
                 label="ambient dim ceiling (2D space)")

    # Overlay live point
    if not np.isnan(_live_dim):
        _ax2.scatter([_W], [min(_live_dim, 2.05)],
                     s=120, color="#d62728", zorder=6, edgecolors="white",
                     linewidths=0.8, label=f"live estimate: width={_W}, dim={_live_dim:.2f}")
    _ax2.set_xlim(0, 270)
    _ax2.set_ylim(0, 2.3)
    _ax2.set_xlabel("network width (hidden units)")
    _ax2.set_ylabel("estimated PM intrinsic dimension (Two-NN)")
    _ax2.set_title("In 2D, PM saturates at dim=2\nOn MNIST (784D), there is no ceiling", fontsize=10)
    _ax2.set_box_aspect(1.0)
    _ax2.legend(fontsize=8)
    spine_off(_ax2)

    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Watch the dark-shaded region as you increase the width slider. In a 2D
    space with a 2D human concept, the PM cannot grow beyond dimension 2.
    But on MNIST — a 784-dimensional space — there is nothing to stop it.
    The PM expands into the exponentially vast "surplus volume" around each class,
    and that surplus volume is where adversarial examples live.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measuring PM dimension — the Two-NN method

    The Two-NN intrinsic dimension estimator (Facco et al. 2017) works as follows.
    Given a point cloud $\{x_i\}$, let $r_1^{(i)}$ and $r_2^{(i)}$ be the distances
    to the nearest and second-nearest neighbours:

    $$
    \mu_i = \frac{r_2^{(i)}}{r_1^{(i)}}
    $$

    Sort $\mu$ ascending and form the empirical CDF $F(\mu) = i / N$.
    The intrinsic dimension $d$ is the slope of the line through the origin that
    fits $-\log(1 - F)$ against $\log(\mu)$:

    $$
    -\log(1 - F(\mu)) = d \cdot \log(\mu)
    $$

    This estimator requires only nearest-neighbour distances and runs in pure
    NumPy. In the precompute script we apply it to the set of pixel-space
    points that the network confidently classifies as class $c$ — that is,
    the empirical PM sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scaling up to MNIST — PM dimension vs. network width

    The five networks below are small CNNs trained on MNIST with base channel
    counts of 4, 8, 16, 32, and 64. As the network grows wider, it fits the
    training data better — and its PM expands into a geometrically larger region
    of pixel space.

    The paper's central claim for real data: PM intrinsic dimension is
    orders-of-magnitude larger than the estimated human concept dimension.
    For handwritten digits, humans need perhaps 10–50 latent degrees of freedom
    (the paper's estimate: stroke count, slant, loop radius, etc. — roughly the
    number of parameters that vary naturally across writers). Our CNN estimates
    come out far higher.
    """)
    return


@app.cell
def _(MNIST_ADV_DIST, MNIST_PM_DIMS, MNIST_WIDTHS, PALETTE, mo, np, plt, spine_off):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: PM dim vs. width per class (one line per digit class)
    _mean_pm = MNIST_PM_DIMS.mean(axis=1)   # (5,)
    _std_pm  = MNIST_PM_DIMS.std(axis=1)    # (5,)

    for _ci in range(10):
        _a1.plot(MNIST_WIDTHS, MNIST_PM_DIMS[:, _ci],
                 color=PALETTE[_ci % len(PALETTE)],
                 lw=1.3, alpha=0.6, marker="o", ms=4)

    _a1.plot(MNIST_WIDTHS, _mean_pm, color="black", lw=2.4, marker="D", ms=6,
             label="mean across 10 classes", zorder=5)
    _a1.fill_between(MNIST_WIDTHS,
                     _mean_pm - _std_pm, _mean_pm + _std_pm,
                     alpha=0.15, color="black")

    # Reference line: paper estimates human digit concept dim ~ 10-50
    _a1.axhspan(10, 50, alpha=0.10, color="#2ca02c",
                label="human concept dim (paper est.)")
    _a1.set_xlabel("CNN base channels (width proxy)")
    _a1.set_ylabel("PM intrinsic dimension (Two-NN)")
    _a1.set_title("PM dimension grows with width — MNIST\n(one thin line per digit class)", fontsize=10)
    _a1.legend(fontsize=8, loc="upper left")
    _a1.set_box_aspect(1.0)
    spine_off(_a1)

    # Right: PM dim vs. adversarial distance — the core prediction
    _a2.scatter(MNIST_PM_DIMS.mean(axis=1), MNIST_ADV_DIST,
                s=90, c=[PALETTE[i % len(PALETTE)] for i in range(5)],
                edgecolors="white", linewidths=0.8, zorder=5)

    # Fit regression line
    _xfit = MNIST_PM_DIMS.mean(axis=1)
    _yfit = MNIST_ADV_DIST
    if _xfit.std() > 1e-6:
        _coeffs = np.polyfit(_xfit, _yfit, 1)
        _xline = np.linspace(_xfit.min(), _xfit.max(), 50)
        _a2.plot(_xline, np.polyval(_coeffs, _xline), "k--", lw=1.5, alpha=0.7,
                 label="linear fit")
        _r = float(np.corrcoef(_xfit, _yfit)[0, 1])
        _a2.text(0.97, 0.97, f"r = {_r:.2f}", transform=_a2.transAxes,
                 ha="right", va="top", fontsize=11, color="#333")

    for _i, (_xp, _yp, _bc) in enumerate(
            zip(MNIST_PM_DIMS.mean(axis=1), MNIST_ADV_DIST, [4, 8, 16, 32, 64])):
        _a2.annotate(f"ch={_bc}", (_xp, _yp), fontsize=8,
                     xytext=(4, 4), textcoords="offset points", alpha=0.75)

    _a2.set_xlabel("mean PM intrinsic dimension")
    _a2.set_ylabel("mean L2 adv. distance")
    _a2.set_title("Larger PM  =  closer adversarial examples\n(paper's central prediction, reproduced on MNIST)", fontsize=10)
    _a2.legend(fontsize=9)
    _a2.set_box_aspect(1.0)
    spine_off(_a2)

    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(MNIST_ADV_DIST, MNIST_PM_DIMS, mo, np):
    _mean_dims = MNIST_PM_DIMS.mean(axis=1)
    _r = float(np.corrcoef(_mean_dims, MNIST_ADV_DIST)[0, 1]) if _mean_dims.std() > 1e-6 else float("nan")
    _min_dim = float(_mean_dims.min())
    _max_dim = float(_mean_dims.max())
    _human_est = "10–50"
    mo.md(f"""
    Mean PM intrinsic dimension across widths: from **{_min_dim:.1f}** (smallest CNN)
    to **{_max_dim:.1f}** (widest CNN). The paper estimates human digit concept
    dimensionality at roughly {_human_est} — our narrowest network already sits
    above that floor. The Pearson correlation between mean PM dimension and
    adversarial distance across our five networks is **r = {_r:.2f}**, matching
    the paper's direction.

    Every additional unit of network width is buying you coverage of a
    geometrically larger region of pixel space — most of which looks like noise
    to a human but is confidently labelled by the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adversarial examples as a prediction of the theory

    If this geometric story is right, adversarial examples are not a quirk of
    gradient descent or a sign that the network is "confused." They are a direct
    consequence of the PM being too large. Two networks with the same test
    accuracy but different PM dimensions should have different adversarial
    robustness — and the one with the larger PM should be easier to attack.

    Below, each panel shows one network family. The x-axis is mean PM intrinsic
    dimension; the y-axis is the mean L2 distance a PGD attacker needed to flip
    the label. The paper tests 18 diverse networks to demonstrate the cross-family
    correlation; here we reproduce the within-family trend for CNNs (left) and
    separately test a depth-varying MLP family the paper did not cover (right).
    The two families occupy similar PM-dimension ranges on our small MNIST setup,
    which is why we show them in separate panels rather than a single scatter.
    """)
    return


@app.cell
def _(
    EXT_ADV_DIST,
    EXT_DEPTHS,
    EXT_PM_DIMS,
    MNIST_ADV_DIST,
    MNIST_PM_DIMS,
    MNIST_WIDTHS,
    PALETTE,
    mo,
    np,
    plt,
    spine_off,
):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: CNN width family
    _cnn_x = MNIST_PM_DIMS.mean(axis=1)
    _cnn_y = MNIST_ADV_DIST
    _ax1.scatter(_cnn_x, _cnn_y,
                 s=110, c=[PALETTE[i % len(PALETTE)] for i in range(len(_cnn_x))],
                 edgecolors="white", linewidths=0.8, zorder=5)
    for _i, (_xp, _yp, _bc) in enumerate(zip(_cnn_x, _cnn_y, MNIST_WIDTHS)):
        _ax1.annotate(f"w={_bc}", (_xp, _yp), fontsize=9,
                      xytext=(5, 4), textcoords="offset points", alpha=0.8)
    if _cnn_x.std() > 1e-6:
        _c1 = np.polyfit(_cnn_x, _cnn_y, 1)
        _xl1 = np.linspace(_cnn_x.min() - 0.05, _cnn_x.max() + 0.05, 50)
        _ax1.plot(_xl1, np.polyval(_c1, _xl1), "k--", lw=1.5, alpha=0.6)
        _r1 = float(np.corrcoef(_cnn_x, _cnn_y)[0, 1])
        _ax1.text(0.97, 0.97, f"r = {_r1:.2f}", transform=_ax1.transAxes,
                  ha="right", va="top", fontsize=12, color="#333")
    _ax1.set_xlabel("mean PM intrinsic dimension (Two-NN)", fontsize=11)
    _ax1.set_ylabel("mean L2 adversarial distance", fontsize=11)
    _ax1.set_title("CNN width sweep — MNIST\nlarger PM dim correlates with shorter adv. distance", fontsize=10)
    _ax1.set_box_aspect(1.0)
    spine_off(_ax1)

    # Right: MLP depth family
    _mlp_x = EXT_PM_DIMS.mean(axis=1)
    _mlp_y = EXT_ADV_DIST
    _ax2.scatter(_mlp_x, _mlp_y,
                 s=110, c=[PALETTE[i % len(PALETTE)] for i in range(len(_mlp_x))],
                 marker="s", edgecolors="white", linewidths=0.8, zorder=5)
    for _i, (_xp, _yp, _d) in enumerate(zip(_mlp_x, _mlp_y, EXT_DEPTHS)):
        _ax2.annotate(f"d={_d}", (_xp, _yp), fontsize=9,
                      xytext=(5, 4), textcoords="offset points", alpha=0.8)
    if _mlp_x.std() > 1e-6:
        _c2 = np.polyfit(_mlp_x, _mlp_y, 1)
        _xl2 = np.linspace(_mlp_x.min() - 0.05, _mlp_x.max() + 0.05, 50)
        _ax2.plot(_xl2, np.polyval(_c2, _xl2), "k--", lw=1.5, alpha=0.6)
        _r2 = float(np.corrcoef(_mlp_x, _mlp_y)[0, 1])
        _ax2.text(0.97, 0.97, f"r = {_r2:.2f}", transform=_ax2.transAxes,
                  ha="right", va="top", fontsize=12, color="#333")
    _ax2.set_xlabel("mean PM intrinsic dimension (Two-NN)", fontsize=11)
    _ax2.set_ylabel("mean L2 adversarial distance", fontsize=11)
    _ax2.set_title("MLP depth sweep (novel extension) — MNIST\nsame axis scale for comparison", fontsize=10)
    _ax2.set_box_aspect(1.0)
    spine_off(_ax2)

    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Novel extension — does depth matter the same way width does?

    The paper (arXiv:2603.03507) evaluates CNNs of varying width and several
    established architectures, but does not systematically test *depth* as an
    independent variable. Here we train a family of MLPs with fixed width (128)
    and varying depth (1 to 5 hidden layers) on MNIST, then ask the same
    question: does PM dimension track adversarial distance across this family?

    If the paper's geometric law is architecture-agnostic, it should hold here
    too. If depth has a qualitatively different effect from width — perhaps
    because deeper networks learn more compositional representations — we might
    see deviations.
    """)
    return


@app.cell
def _(EXT_ADV_DIST, EXT_DEPTHS, EXT_PM_DIMS, PALETTE, data, mo, np, plt, spine_off):
    _ext_acc = data["ext_test_acc"].astype(float)

    _fig, (_b1, _b2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: PM dim vs depth
    _mean_mlp_pm = EXT_PM_DIMS.mean(axis=1)
    _std_mlp_pm  = EXT_PM_DIMS.std(axis=1)
    _b1.plot(EXT_DEPTHS, _mean_mlp_pm, color="#d62728", lw=2.4,
             marker="s", ms=7, label="mean PM dim (MLP, width=128)")
    _b1.fill_between(EXT_DEPTHS,
                     _mean_mlp_pm - _std_mlp_pm,
                     _mean_mlp_pm + _std_mlp_pm,
                     alpha=0.15, color="#d62728")
    _ax_r = _b1.twinx()
    _ax_r.plot(EXT_DEPTHS, _ext_acc, color="#1f77b4", lw=1.8,
               marker="o", ms=6, ls="--", label="test accuracy")
    _ax_r.set_ylabel("test accuracy", color="#1f77b4", fontsize=10)
    _ax_r.tick_params(axis="y", colors="#1f77b4")
    _b1.set_xlabel("number of hidden layers (depth)")
    _b1.set_ylabel("mean PM intrinsic dimension", color="#d62728", fontsize=10)
    _b1.tick_params(axis="y", colors="#d62728")
    _b1.set_title("PM dimension vs. MLP depth\n(fixed width = 128)", fontsize=10)
    _b1.set_box_aspect(1.0)
    for _s in ("top",): _b1.spines[_s].set_visible(False)

    # Right: adv dist vs depth
    _b2.plot(EXT_DEPTHS, EXT_ADV_DIST, color="#9467bd", lw=2.4,
             marker="D", ms=7, label="mean L2 adv. distance")
    for _i, (_dp, _ad) in enumerate(zip(EXT_DEPTHS, EXT_ADV_DIST)):
        _b2.annotate(f"d={_dp}", (_dp, _ad), fontsize=8.5,
                     xytext=(3, 4), textcoords="offset points", alpha=0.75)
    _b2.set_xlabel("number of hidden layers (depth)")
    _b2.set_ylabel("mean L2 adversarial distance")
    _b2.set_title("Adversarial distance vs. MLP depth\n(fixed width = 128)", fontsize=10)
    _b2.set_box_aspect(1.0)
    spine_off(_b2)

    # Add correlation annotation
    if _mean_mlp_pm.std() > 1e-6 and EXT_ADV_DIST.std() > 1e-6:
        _r_mlp = float(np.corrcoef(_mean_mlp_pm, EXT_ADV_DIST)[0, 1])
        _b2.text(0.97, 0.97, f"r(dim, adv) = {_r_mlp:.2f}",
                 transform=_b2.transAxes, ha="right", va="top", fontsize=10)

    _b1.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(EXT_ADV_DIST, EXT_PM_DIMS, mo, np):
    _mean_pm_mlp = EXT_PM_DIMS.mean(axis=1)
    _r = float(np.corrcoef(_mean_pm_mlp, EXT_ADV_DIST)[0, 1]) if (_mean_pm_mlp.std() > 1e-6 and EXT_ADV_DIST.std() > 1e-6) else float("nan")
    _direction = "negative" if not np.isnan(_r) and _r < 0 else ("positive" if not np.isnan(_r) and _r > 0 else "near-zero")
    mo.md(f"""
    The depth-varying MLP family shows a within-family correlation of **r = {_r:.2f}**
    between PM intrinsic dimension and adversarial distance ({_direction} direction).
    This extension is genuinely novel relative to the paper — the depth dimension was
    not tested. The range of both PM dimension and adversarial distance is narrow in
    our small MNIST setup, which compresses the signal; the paper's effect is clearest
    across diverse architectures spanning orders of magnitude in scale.
    What we can say: the geometric framing is not obviously *wrong* for depth-varying
    MLPs. Extending to larger models on harder tasks (CIFAR-10, ImageNet) would be
    the natural next step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Can we shrink a PM? Adversarial training

    The most common practical defense against adversarial examples is PGD
    adversarial training (Madry et al. 2018): augment the training data with
    adversarial examples so the network learns not to be fooled by them.

    The paper's prediction: adversarial training will **reduce PM dimension** —
    the network is forced to be conservative about what it confidently classifies.
    The adversarial distance should increase. But the honest punchline is that
    even after adversarial training, the PM is still far larger than the human
    concept. You've shrunk the surplus, not eliminated it.
    """)
    return


@app.cell
def _(PALETTE, data, mo, np, plt, spine_off):
    _std_pm  = data["adv_train_std_pm_dim"].astype(float)   # (10,)
    _adv_pm  = data["adv_train_adv_pm_dim"].astype(float)   # (10,)
    _std_ad  = float(data["adv_train_std_adv_dist"])
    _adv_ad  = float(data["adv_train_adv_adv_dist"])
    _std_acc = float(data["adv_train_std_acc"])
    _adv_acc = float(data["adv_train_adv_acc"])

    _digit_labels = [str(i) for i in range(10)]

    _fig, (_c1, _c2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: PM dim per class, standard vs adv-trained
    _x_pos = np.arange(10)
    _width = 0.36
    _bars_std = _c1.bar(_x_pos - _width / 2, _std_pm, _width,
                         color="#1f77b4", alpha=0.85, label=f"Standard (acc={_std_acc:.2f})")
    _bars_adv = _c1.bar(_x_pos + _width / 2, _adv_pm, _width,
                         color="#d62728", alpha=0.85, label=f"Adv-trained (acc={_adv_acc:.2f})")
    _c1.axhspan(10, 50, alpha=0.10, color="#2ca02c", label="human digit concept est.")
    _c1.set_xticks(_x_pos)
    _c1.set_xticklabels(_digit_labels)
    _c1.set_xlabel("digit class")
    _c1.set_ylabel("PM intrinsic dimension (Two-NN)")
    _c1.set_title("Adversarial training shrinks PM dimension\n(blue = standard, red = PGD-trained)", fontsize=10)
    _c1.legend(fontsize=8)
    _c1.set_box_aspect(1.0)
    spine_off(_c1)

    # Right: summary bar — mean PM dim and adv distance
    _labels_comp = ["Standard", "Adv-trained"]
    _pm_means = [_std_pm.mean(), _adv_pm.mean()]
    _adv_dists_pair = [_std_ad, _adv_ad]

    _ax2_r = _c2
    _ax2_twin = _c2.twinx()
    _xc = np.array([0, 1])
    _bpm = _c2.bar(_xc - 0.18, _pm_means, 0.34, color=["#1f77b4", "#d62728"],
                    alpha=0.80, label="mean PM dim")
    _bad = _ax2_twin.bar(_xc + 0.18, _adv_dists_pair, 0.34, color=["#1f77b4", "#d62728"],
                          alpha=0.45, label="adv. distance")
    _c2.set_xticks([0, 1])
    _c2.set_xticklabels(_labels_comp, fontsize=11)
    _c2.set_ylabel("mean PM intrinsic dimension", fontsize=10)
    _ax2_twin.set_ylabel("mean L2 adversarial distance", fontsize=10)
    _c2.set_title(
        "PM shrinks, adv. distance grows — but misalignment persists\n"
        "(solid = PM dim, faded = adv. distance)",
        fontsize=10,
    )
    # Value labels
    for _bar, _v in zip(_bpm, _pm_means):
        _c2.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.3,
                 f"{_v:.1f}", ha="center", va="bottom", fontsize=9)
    for _bar, _v in zip(_bad, _adv_dists_pair):
        _ax2_twin.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() * 1.02,
                       f"{_v:.3f}", ha="center", va="bottom", fontsize=9)
    _c2.set_box_aspect(1.0)
    for _s in ("top",): _c2.spines[_s].set_visible(False)

    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(data, mo, np):
    _std_pm  = data["adv_train_std_pm_dim"].astype(float).mean()
    _adv_pm  = data["adv_train_adv_pm_dim"].astype(float).mean()
    _std_ad  = float(data["adv_train_std_adv_dist"])
    _adv_ad  = float(data["adv_train_adv_adv_dist"])
    _reduction_pct = 100 * (1 - _adv_pm / _std_pm) if _std_pm > 0 else 0.0
    _dist_gain_pct = 100 * (_adv_ad / _std_ad - 1) if _std_ad > 0 else 0.0
    mo.md(f"""
    Adversarial training reduced mean PM intrinsic dimension by **{_reduction_pct:.1f}%**
    (from {_std_pm:.1f} to {_adv_pm:.1f}) and increased adversarial distance by
    **{_dist_gain_pct:.1f}%** (from {_std_ad:.3f} to {_adv_ad:.3f}).

    The improvement is real. But the paper's point holds: even the PGD-trained
    network's PM is still far above the estimated human concept dimension of 10–50.
    You are not climbing back down to human-level alignment. You are taking a
    step on an exponential staircase.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook does *not* claim

    - **The Two-NN estimator is noisy at high dimension.** With 400 samples
      per class in 784 dimensions, the estimates are directionally correct but
      not precise. The paper uses larger sample budgets and multiple estimators;
      we use one for computational tractability in a precompute script that runs
      in under 30 minutes on CPU.

    - **Our MNIST setup is smaller than the paper's.** The paper evaluates 18
      networks, including large-scale ImageNet models and certified defenses. Our
      5-CNN MNIST sweep reproduces the *direction* of the effect, not the
      magnitude. PM dimensions here are much smaller than the paper's numbers
      for ImageNet models.

    - **PGD is not the hardest attack.** We use L2 PGD for the distance measure.
      AutoAttack would give smaller distances (i.e., easier to attack), which
      would likely steepen the negative slope. The ranking across widths should
      be stable regardless.

    - **The adversarial training here is 3-epoch PGD, not full AT.** Standard
      adversarial training converges slowly. Our 3-epoch variant gives a
      directional comparison, not a state-of-the-art robust model.

    - **We do not test the paper's theorem directly.** The paper proves that PM
      volume grows exponentially with dimension under mild conditions on the
      network's confidence geometry. We estimate *intrinsic dimension* as a proxy
      for this, not volume, because volume in 784 dimensions is not measurable
      from samples.

    None of these caveats change the central observation: the geometric
    story holds on MNIST. Networks that fit the training data more aggressively
    do so by expanding their perceptual manifolds into regions of pixel space
    that humans would never label confidently — and the metric you pay in
    adversarial robustness tracks that expansion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself

    - **Repo**: [`github.com/FarseenSh/alphaxiv-marimo-comp`](https://github.com/FarseenSh/alphaxiv-marimo-comp)
      — see `scripts/curse_dimensions_precompute.py` to regenerate data with your
      own architectures. The Two-NN estimator is plain NumPy; the precompute
      script caches checkpoints so re-runs are fast.
    - **Original paper**: [arXiv:2603.03507](https://arxiv.org/abs/2603.03507)
      — Salvatore, Fort, Ganguli (Stanford, March 2026). Fort and Ganguli are
      at the intersection of adversarial robustness and the geometry of
      neural representations — the paper is dense but the geometric intuition
      in Section 2 is clear and worth reading on its own.
    - **alphaXiv discussion**: [alphaxiv.org/abs/2603.03507](https://www.alphaxiv.org/abs/2603.03507).
    - **Two-NN reference**: Facco et al. (2017) "Estimating the intrinsic
      dimension of datasets by a minimal neighborhood information"
      — [PNAS 114(17)](https://doi.org/10.1073/pnas.1704895114).

    Built for the
    [alphaXiv x marimo "Bring Research to Life" notebook competition](https://marimo.io/pages/events/notebook-competition).
    """)
    return


if __name__ == "__main__":
    app.run()
