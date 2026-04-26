# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-image",
#     "pillow",
# ]
# ///
"""Dead Networks, Live Explanations — a marimo walkthrough of
"The Dead Salmons of AI Interpretability" (Meloux, Dirupo, Portet, Peyrard, 2025).

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

    # Suppress matplotlib font-cache noise on first-run Pyodide environments.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    import marimo as mo
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        import matplotlib.pyplot as plt
        _warmup = plt.figure(figsize=(0.1, 0.1))
        plt.close(_warmup)
    import numpy as np

    GH_RAW = "https://raw.githubusercontent.com/FarseenSh/alphaxiv-marimo-comp/main/data/dead_salmons.npz"

    def _fetch_remote():
        with urllib.request.urlopen(GH_RAW) as _r:
            return dict(np.load(io.BytesIO(_r.read()), allow_pickle=True))

    # In Pyodide/WASM, __file__ may exist in a virtual fs with stale bytes —
    # always fetch remotely there.
    if find_spec("js") is not None:
        data = _fetch_remote()
    else:
        _local = Path(__file__).resolve().parents[1] / "data" / "dead_salmons.npz"
        if _local.exists() and _local.stat().st_size > 0:
            try:
                data = dict(np.load(_local, allow_pickle=True))
            except Exception:
                data = _fetch_remote()
        else:
            data = _fetch_remote()

    # -----------------------------------------------------------------------
    # Helper constants derived from the data
    # -----------------------------------------------------------------------
    IMAGE_SIZE = 32
    CLASS_NAMES = [str(c) for c in data["class_names"]]
    METHOD_NAMES = [str(m) for m in data["method_names"]]
    METHOD_LABELS = {
        "vanilla_gradient": "Vanilla gradient",
        "gradient_times_input": "Gradient x input",
        "smoothgrad": "SmoothGrad",
        "integrated_gradients": "Integrated gradients",
    }
    # method_idx lookup
    METHOD_IDX = {name: i for i, name in enumerate(METHOD_NAMES)}
    STATE_IDX = {"trained": 0, "random": 1}

    PALETTE = [
        "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
        "#ff7f0e", "#17becf", "#e377c2", "#bcbd22",
    ]

    # -----------------------------------------------------------------------
    # Visual helpers
    # -----------------------------------------------------------------------

    def overlay_saliency(ax, img_u8, sal, cmap="magma", alpha=0.75):
        """Show grayscale image with saliency map overlaid."""
        gray = img_u8.mean(axis=-1).astype(np.float32)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        sal_f = sal.astype(np.float32)
        mn, mx = sal_f.min(), sal_f.max()
        if mx > mn:
            sal_f = (sal_f - mn) / (mx - mn)
        ax.imshow(sal_f, cmap=cmap, alpha=alpha, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def make_panel_ax(ax, title=None, color=None):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_box_aspect(1.0)
        if title:
            ax.set_title(title, fontsize=9, color=color or "black", pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def spine_off(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    return (
        CLASS_NAMES,
        GH_RAW,
        IMAGE_SIZE,
        METHOD_IDX,
        METHOD_LABELS,
        METHOD_NAMES,
        PALETTE,
        STATE_IDX,
        data,
        io,
        logging,
        make_panel_ax,
        mo,
        np,
        overlay_saliency,
        plt,
        spine_off,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align:center; padding: 28px 0 4px 0;">
    <h1 style="margin-bottom: 6px; font-size: 2.5rem; letter-spacing: -0.5px;">
      Dead Networks, Live Explanations
    </h1>
    <p style="font-size: 1.08rem; opacity: 0.82; margin-top: 0; max-width: 680px; margin-left: auto; margin-right: auto;">
      Your saliency maps, linear probes, and attribution scores look equally
      convincing on a network that has learned nothing. This notebook shows
      you the dead salmon — then hands you the statistical tool to tell the
      difference.
    </p>
    <p style="font-size: 0.88rem; opacity: 0.55; margin-top: 12px;">
      Reproducing and extending<br>
      <em>The Dead Salmons of AI Interpretability</em><br>
      Meloux, Dirupo, Portet, Peyrard &mdash;
      <a href="https://arxiv.org/abs/2512.18792">arXiv:2512.18792</a> (Dec 2025)
    </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The setup

    In 2009, neuroscientist Craig Bennett and colleagues put a dead Atlantic salmon
    in an fMRI scanner, showed it photographs of people in social situations, and
    asked (as a standard protocol) whether any voxels in its brain responded to the
    emotional content of the images. Several did — with *p* < 0.001.

    The paper was a deliberate provocation. The scanner was picking up noise; the
    multiple-comparison problem turned that noise into a "discovery." Neuroimaging
    responded by tightening its correction procedures and raising its standards.

    Meloux et al. (2025) argue the same failure is widespread in AI interpretability.
    The core claim:

    > Feature attribution methods, linear probes, sparse autoencoders, and causal
    > analyses **produce plausible-looking explanations even for randomly initialized
    > neural networks.** Without a statistical null model, you cannot distinguish
    > signal from structure-in-noise.

    The fix they propose is to treat interpretability as statistical estimation:
    compute an **alignment score** for your finding, then compare it against a
    permutation null to obtain a *p*-value. Results that survive this test are
    real; results that don't are dead salmons.

    This notebook gives you:
    - A side-by-side gut punch: saliency maps on trained vs. random networks.
    - An interactive method picker: all four standard attribution methods, both networks.
    - A linear probe comparison: above-chance accuracy on both networks.
    - A permutation null: the statistical tool that tells the two apart.
    - A false-positive-rate sweep: which methods are most susceptible.

    One small equation to frame the test. Let $s$ be the saliency map and $m$
    be a target mask. The alignment score is:

    $$
    r = \operatorname{corr}(s, m)
    $$

    We declare a finding "significant" if the observed $r$ exceeds the 95th percentile
    of the null distribution $\{r_\pi : \pi \in \text{label permutations}\}$.
    Without this test, a random network passes as often as a trained one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline — at a glance

    | Stage | Input | Output |
    |---|---|---|
    | **Trained model** | CIFAR-10 training set (50 k images) | Small CNN, ~70% test accuracy |
    | **Random twin** | Same architecture, fresh Kaiming initialization | ~10% test accuracy (chance) |
    | **Saliency** | 8 test images x 4 methods x 2 models | Normalized heat maps (32x32) |
    | **Linear probe** | Frozen features from both models | Animal-vs-vehicle accuracy |
    | **Permutation null** | Saliency scores + label shuffles | *p*-value per finding |
    | **FPR sweep** | All 4 methods on the random model | False-positive rate table |

    Everything heavy was computed offline in `scripts/dead_salmons_precompute.py`
    using PyTorch. The notebook reads only the 8 MB `.npz`. PyTorch is not in
    Pyodide and is not imported here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The hero comparison — two rows, one question

    Below: **eight CIFAR-10 test images**, each correctly classified by the trained
    network. For each image we show the saliency map from **vanilla gradient** attribution.

    Top row: the trained model. Bottom row: a fresh random-init network with the same
    architecture. Both rows look like they are highlighting something meaningful.
    """)
    return


@app.cell(hide_code=True)
def _(CLASS_NAMES, PALETTE, data, mo, np, overlay_saliency, plt):
    _imgs = data["imgs_uint8"]           # (8, 32, 32, 3) uint8
    _labels = data["labels"]             # (8,)
    _sal = data["saliency"]             # (4, 2, 8, 32, 32) float16
    _vanilla_idx = 0                     # vanilla_gradient is method 0

    _fig, _axes = plt.subplots(2, 8, figsize=(3.0 * 8, 3.4 * 2))
    _row_labels = ["Trained CNN\n(~70% accuracy)", "Random-init network\n(~10% accuracy)"]

    for _row in range(2):
        for _col in range(8):
            _ax = _axes[_row, _col]
            _sal_map = _sal[_vanilla_idx, _row, _col].astype(np.float32)
            overlay_saliency(_ax, _imgs[_col], _sal_map, cmap="magma")
            _ax.set_box_aspect(1.0)
            if _row == 0:
                _ax.set_title(CLASS_NAMES[_labels[_col]], fontsize=9,
                              color=PALETTE[_col], fontweight="bold", pad=3)
        # Row label on the left-most axis
        _axes[_row, 0].set_ylabel(_row_labels[_row], fontsize=9, labelpad=6)

    _fig.text(0.5, 1.01,
              "Saliency maps: vanilla gradient attribution. "
              "Both rows look equally \"interpretable\".",
              ha="center", fontsize=9, style="italic", color="#444")
    plt.tight_layout(h_pad=0.8)
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pause here. Look at both rows carefully. The top-row maps are brighter in some
    semantically reasonable places — the frog's outline, the automobile's body. But
    the bottom-row maps are not random scatter either. They show spatial structure,
    edge responses, contrast responses. An untrained reviewer would find both
    convincing.

    That is the dead salmon. The random network has learned nothing, but the
    attribution method fits noise into a plausible story.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method picker — four tools, two networks

    Switch the dropdown to see whether the pattern holds for all four standard
    attribution methods. It does. That is the paper's empirical claim.
    """)
    return


@app.cell
def _(METHOD_LABELS, METHOD_NAMES, mo):
    method_picker = mo.ui.dropdown(
        options={METHOD_LABELS[m]: m for m in METHOD_NAMES},
        value=METHOD_NAMES[0],
        label="Attribution method",
    )
    method_picker
    return (method_picker,)


@app.cell
def _(
    CLASS_NAMES,
    METHOD_IDX,
    METHOD_LABELS,
    PALETTE,
    data,
    make_panel_ax,
    method_picker,
    mo,
    np,
    overlay_saliency,
    plt,
):
    _method = method_picker.value
    _midx = METHOD_IDX[_method]
    _imgs = data["imgs_uint8"]
    _labels = data["labels"]
    _sal = data["saliency"]           # (4, 2, 8, 32, 32)
    _trained_acc = float(data["trained_acc"])
    _random_acc = float(data["random_acc"])

    _fig, _axes = plt.subplots(2, 8, figsize=(3.0 * 8, 3.4 * 2))
    _row_labels = [
        f"Trained CNN  (test acc {_trained_acc:.0%})",
        f"Random-init  (test acc {_random_acc:.0%})",
    ]

    for _row in range(2):
        for _col in range(8):
            _ax = _axes[_row, _col]
            _sal_map = _sal[_midx, _row, _col].astype(np.float32)
            overlay_saliency(_ax, _imgs[_col], _sal_map, cmap="magma")
            _ax.set_box_aspect(1.0)
            if _row == 0:
                _ax.set_title(CLASS_NAMES[_labels[_col]], fontsize=9,
                              color=PALETTE[_col], fontweight="bold", pad=3)
        _axes[_row, 0].set_ylabel(_row_labels[_row], fontsize=9, labelpad=6)

    _fig.text(0.5, 1.01,
              f"Method: {METHOD_LABELS[_method]}",
              ha="center", fontsize=9, style="italic", color="#444")
    plt.tight_layout(h_pad=0.8)
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice: SmoothGrad and integrated gradients produce smoother maps than vanilla
    gradient. They look *more* polished on both networks. Polish is not evidence of
    correctness. A method can be technically sophisticated and statistically
    uninformative at the same time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear probe — above chance on both networks

    The dead-salmon problem is not limited to gradient-based attribution. Linear
    probing suffers the same pathology. We train a simple logistic regression on the
    frozen features of each network to classify images as **animal** (bird, cat,
    deer, dog, frog, horse) vs. **vehicle** (airplane, automobile, ship, truck) —
    CIFAR-10's two natural superclasses.

    Both probes reach above-chance accuracy. The random network's features have
    non-trivial random structure; a linear classifier can extract signal from
    that structure. Without a baseline, you would conclude that both networks
    have learned the animal/vehicle distinction.
    """)
    return


@app.cell
def _(data, mo, np, plt, spine_off):
    _probe_trained_tr = float(data["probe_trained_train_acc"])
    _probe_trained_te = float(data["probe_trained_test_acc"])
    _probe_random_tr = float(data["probe_random_train_acc"])
    _probe_random_te = float(data["probe_random_test_acc"])

    _chance = 0.5   # binary classification baseline
    _models = ["Trained CNN", "Random-init"]
    _train_accs = [_probe_trained_tr, _probe_random_tr]
    _test_accs = [_probe_trained_te, _probe_random_te]

    _x = np.array([0, 1])
    _width = 0.32

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _bars_tr = _ax.bar(_x - _width / 2, _train_accs, _width, label="Train accuracy",
                       color=["#1f77b4", "#d62728"], alpha=0.85, edgecolor="white")
    _bars_te = _ax.bar(_x + _width / 2, _test_accs, _width, label="Test accuracy",
                       color=["#1f77b4", "#d62728"], alpha=0.50, edgecolor="white",
                       linewidth=1.2)
    _ax.axhline(_chance, ls="--", color="#888", lw=1.4, label="Chance (50%)")
    _ax.axhline(1.0, ls=":", color="#ccc", lw=0.8)

    # Value labels
    for _bar, _v in zip(list(_bars_tr) + list(_bars_te),
                        _train_accs + _test_accs):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.012,
                 f"{_v:.0%}", ha="center", va="bottom", fontsize=9)

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_models, fontsize=10)
    _ax.set_ylabel("Animal vs. vehicle probe accuracy")
    _ax.set_ylim(0, 1.15)
    _ax.set_title("Linear probe on frozen features — both networks beat chance")
    _ax.legend(fontsize=9)
    spine_off(_ax)
    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(data, mo):
    _tr_te = float(data["probe_trained_test_acc"])
    _rn_te = float(data["probe_random_test_acc"])
    mo.md(f"""
    Trained-model probe test accuracy: **{_tr_te:.1%}**.
    Random-model probe test accuracy: **{_rn_te:.1%}**.

    Both exceed 50% chance. The gap between the two is real — the trained
    network's features are more linearly separable — but the gap is smaller
    than intuition would suggest. If you only ran the probe on the random
    network and saw {_rn_te:.0%}, you might conclude that random features are
    informative. They are — but not for the reason you think. That is the
    salmon.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The statistical fix — the permutation null

    The paper's proposed remedy is to treat interpretability findings as statistical
    claims and test them against explicit null models. Here we implement the simplest
    version:

    1. Compute an **alignment score** $r = \operatorname{corr}(s, m)$ between each
       saliency map $s$ and a target mask $m$. We use a Gaussian center-bias mask —
       plausibly the kind of thing a correct saliency map should highlight.
    2. Generate a null distribution by permuting the mask pixels 500 times and
       recomputing $r$ each time. This breaks any signal while preserving the
       distribution of values in the mask.
    3. Compare the observed $r$ to the null distribution. If the observed score
       sits in the right tail, the finding survives; if it sits inside the null,
       it is a dead salmon.

    Below: histograms of the null distribution (gray) vs. the observed score
    for the trained model (blue vertical line) and the random model (red vertical
    line). Pick a method from the dropdown.
    """)
    return


@app.cell
def _(METHOD_LABELS, METHOD_NAMES, mo):
    perm_method_picker = mo.ui.dropdown(
        options={METHOD_LABELS[m]: m for m in METHOD_NAMES},
        value=METHOD_NAMES[0],
        label="Attribution method",
    )
    perm_method_picker
    return (perm_method_picker,)


@app.cell
def _(
    CLASS_NAMES,
    METHOD_IDX,
    METHOD_LABELS,
    PALETTE,
    data,
    mo,
    np,
    perm_method_picker,
    plt,
    spine_off,
):
    _method = perm_method_picker.value
    _midx = METHOD_IDX[_method]
    _labels = data["labels"]           # (8,)

    # perm_observed: (4, 2, 8)   perm_null: (4, 2, 8, 500)
    _obs = data["perm_observed"][_midx]   # (2, 8)
    _null = data["perm_null"][_midx].astype(np.float32)  # (2, 8, 500)

    # One row per network state, one column per image
    _fig, _axes = plt.subplots(2, 8, figsize=(3.0 * 8, 3.4 * 2), sharex=True, sharey="row")
    _state_labels = ["Trained CNN", "Random-init"]
    _state_colors = ["#1f77b4", "#d62728"]

    for _si in range(2):
        for _ci in range(8):
            _ax = _axes[_si, _ci]
            _null_vals = _null[_si, _ci]
            _obs_val = float(_obs[_si, _ci])
            _threshold = np.percentile(_null_vals, 95)
            _pval = float((_null_vals >= _obs_val).mean())

            # Null histogram
            _ax.hist(_null_vals, bins=30, color="#aaa", alpha=0.7,
                     edgecolor="none", density=True)
            # 95th-percentile threshold
            _ax.axvline(_threshold, color="#888", lw=1.0, ls="--", alpha=0.8)
            # Observed score
            _ax.axvline(_obs_val, color=_state_colors[_si], lw=2.2, alpha=0.95,
                        label=f"r={_obs_val:.3f}")

            _ax.set_box_aspect(1.0)
            _sig = "p<0.05" if _pval < 0.05 else "n.s."
            if _si == 0:
                _ax.set_title(CLASS_NAMES[_labels[_ci]], fontsize=8,
                              color=PALETTE[_ci], fontweight="bold", pad=3)
            _ax.text(0.97, 0.95, _sig, transform=_ax.transAxes,
                     ha="right", va="top", fontsize=7.5,
                     color=_state_colors[_si])
            _ax.set_yticks([])
            for _sp in _ax.spines.values(): _sp.set_visible(False)
        _axes[_si, 0].set_ylabel(_state_labels[_si], fontsize=9, labelpad=4)

    _fig.text(0.5, 1.02,
              f"Permutation null (gray) vs. observed alignment score (colored line) "
              f"— {METHOD_LABELS[_method]}. "
              "'p<0.05' means observed > 95th-percentile of null.",
              ha="center", fontsize=8.5, style="italic", color="#444")
    plt.tight_layout(h_pad=0.6)
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(data, mo, np):
    # Summarize which fraction of trained vs random pass the null test
    _obs = data["perm_observed"]    # (4, 2, 8)
    _null = data["perm_null"].astype(np.float32)  # (4, 2, 8, 500)
    _method_names_raw = [str(m) for m in data["method_names"]]
    _labels_display = {
        "vanilla_gradient": "Vanilla gradient",
        "gradient_times_input": "Gradient x input",
        "smoothgrad": "SmoothGrad",
        "integrated_gradients": "Integrated gradients",
    }

    _rows = []
    for _mi, _mn in enumerate(_method_names_raw):
        for _si, _sk in enumerate(["Trained", "Random"]):
            _pvals = (_null[_mi, _si] >= _obs[_mi, _si, :, None]).mean(axis=1)  # (8,)
            _pass_frac = float((_pvals < 0.05).mean())
            _rows.append(f"| {_labels_display[_mn]} | {_sk} | {_pass_frac:.0%} |")

    _table = "\n".join(_rows)
    mo.md(f"""
    ### How often does each model pass the test?

    | Method | Network | Fraction of images passing (p < 0.05 vs null) |
    |---|---|---|
    {_table}

    Ideally: trained model passes often, random model rarely. The gap between
    the two rows within each method is the signal-to-noise ratio of that method.
    Small gaps mean the method is particularly susceptible to false positives.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The `dead_salmon_check` framing

    The result above packages into a reusable function. Here is the interface:

    ```python
    def dead_salmon_check(saliency_maps, target_mask, n_perms=500):
        # saliency_maps : (N, H, W) float array
        #     Normalized saliency maps for N images.
        # target_mask   : (H, W) float array
        #     The mask you expect a correct attribution to highlight.
        # n_perms       : int
        #     Number of permutation samples for the null distribution.
        #
        # Returns
        # -------
        # observed      : (N,) float  -- alignment score per image
        # null_dist     : (N, n_perms) float  -- null distribution per image
        # p_values      : (N,) float  -- fraction of null >= observed
        ...
    ```

    **Usage:** run it on your model's saliency maps. If `p_values.mean() < 0.05`,
    your findings survive the test. If not, you may be looking at a dead salmon.

    The function is independent of architecture, dataset, and attribution library —
    it operates purely on the output arrays. This is the paper's prescription made
    concrete: treat your explanations as statistical claims, not visualizations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweep — false positive rates across all four methods

    The permutation null gives us a formal false-positive rate: the fraction of
    saliency maps from the **random model** that would be declared "significant"
    at *p* < 0.05 without multiple-comparison correction. A perfectly behaved method
    would give an FPR of 5% (matching the threshold). Rates well above 5% indicate
    a method that cannot distinguish a trained network from a random one.
    """)
    return


@app.cell
def _(data, mo, np, plt, spine_off):
    _fpr = data["fpr"].astype(np.float32)   # (4,)
    _method_names_raw = [str(m) for m in data["method_names"]]
    _labels_display = [
        "Vanilla\ngradient",
        "Gradient\nx input",
        "SmoothGrad",
        "Integrated\ngradients",
    ]
    _colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(range(4), _fpr, color=_colors, alpha=0.85, width=0.55,
                    edgecolor="white")
    _ax.axhline(0.05, ls="--", color="#888", lw=1.4, label="Nominal FPR = 5%")
    for _bar, _v in zip(_bars, _fpr):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.012,
                 f"{_v:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _ax.set_xticks(range(4))
    _ax.set_xticklabels(_labels_display, fontsize=10)
    _ax.set_ylabel("False positive rate (random-init model, p < 0.05)")
    _ax.set_ylim(0, min(1.1, _fpr.max() + 0.25))
    _ax.set_title(
        "How often does each attribution method produce a 'significant' result\n"
        "on a network that has learned nothing?"
    )
    _ax.legend(fontsize=9)
    spine_off(_ax)
    plt.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(data, mo, np):
    _fpr = data["fpr"].astype(np.float32)
    _method_names_raw = [str(m) for m in data["method_names"]]
    _labels_display = {
        "vanilla_gradient": "Vanilla gradient",
        "gradient_times_input": "Gradient x input",
        "smoothgrad": "SmoothGrad",
        "integrated_gradients": "Integrated gradients",
    }
    _worst_idx = int(np.argmax(_fpr))
    _best_idx = int(np.argmin(_fpr))
    _worst_name = _labels_display[_method_names_raw[_worst_idx]]
    _best_name = _labels_display[_method_names_raw[_best_idx]]
    mo.md(f"""
    The worst offender in this experiment: **{_worst_name}** ({_fpr[_worst_idx]:.0%} FPR).
    The most conservative: **{_best_name}** ({_fpr[_best_idx]:.0%} FPR).

    These numbers are specific to our CIFAR-10 setup and the Gaussian center mask
    we use as the target. The paper's analysis is broader — it covers probing,
    sparse autoencoders, and causal interventions across multiple datasets. Our
    gradient-based focus is an illustration, not a comprehensive audit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook does *not* claim

    - **This is not evidence that interpretability is useless.** The trained model
      consistently scores higher than the random model on the permutation test.
      The signal is real; the problem is that the noise floor is also real, and
      most published work does not measure it.

    - **The center-bias mask is a proxy, not a ground truth.** We cannot
      obtain a true semantic mask for CIFAR-10 without per-pixel annotations.
      The Gaussian center mask captures the known prior that important objects
      tend to be centered; it is not the same as a ground-truth segmentation.
      A more rigorous test would use pixel-level class activation maps or
      human eye-tracking data as the target.

    - **We tested a tiny CNN, not a large model.** The paper argues the problem
      is general — it covers ResNets, vision transformers, and language model
      probes. Whether the FPR gap between trained and random models narrows or
      widens with scale is an open question.

    - **The permutation test here uses pixel-shuffling, not label-shuffling.**
      The paper's proposed null is richer: permute the *labels* of the dataset,
      retrain the explanation method, compare. Pixel-shuffling is a faster proxy
      that tests whether the map's spatial distribution is non-trivially
      structured, but it is not identical to the paper's prescription.

    - **One threshold does not fit all claims.** We use *p* < 0.05 with no
      multiple-comparison correction across eight images. A more careful
      analysis would apply Bonferroni or Benjamini–Hochberg correction.

    None of these caveats change the central observation: standard attribution
    tools, applied to a network that has learned nothing, produce output that
    looks indistinguishable from a trained network's output — without a
    statistical test to tell them apart. That is the paper's contribution.
    The `dead_salmon_check` framing is a step toward making that test routine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself

    - **Repo**: [`github.com/FarseenSh/alphaxiv-marimo-comp`](https://github.com/FarseenSh/alphaxiv-marimo-comp)
      — see `scripts/dead_salmons_precompute.py` to regenerate the data with your
      own CNN and dataset. The permutation test is plain NumPy; no attribution
      library required.
    - **Original paper**: [arXiv:2512.18792](https://arxiv.org/abs/2512.18792)
      — Meloux, Dirupo, Portet, Peyrard (Universite Grenoble Alpes, Dec 2025).
    - **Precursor work**: Adebayo et al., "Sanity Checks for Saliency Maps"
      (NeurIPS 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)) —
      the direct ML precedent that this paper extends to a formal statistical
      framework.
    - **alphaXiv discussion**: [alphaxiv.org/abs/2512.18792](https://www.alphaxiv.org/abs/2512.18792).

    Built for the
    [alphaXiv x marimo "Bring Research to Life" notebook competition](https://marimo.io/pages/events/notebook-competition).
    """)
    return


if __name__ == "__main__":
    app.run()
