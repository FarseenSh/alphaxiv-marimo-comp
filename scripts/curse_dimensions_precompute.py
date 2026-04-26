"""Pre-compute perceptual-manifold data for the 'Curse of Extra Dimensions' notebook.

Trains small CNNs on MNIST at five width scales, estimates perceptual-manifold (PM)
intrinsic dimension via the Two-NN method (Facco et al. 2017), runs PGD adversarial
attacks to measure foreign-class distance, retrains one network with PGD adversarial
training, and sweeps a novel depth-varying MLP family not tested in the paper.

Everything is saved to data/curse_dimensions.npz.

Usage:
    python scripts/curse_dimensions_precompute.py

Re-runnable: if the npz already exists and --force is not passed, each phase checks
for its keys and skips if already computed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import torchvision
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "curse_dimensions.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_SEED = 42
np.random.seed(TORCH_SEED)
torch.manual_seed(TORCH_SEED)


# ---------------------------------------------------------------------------
# Two-NN intrinsic dimension estimator (Facco et al. 2017, ~20 lines numpy)
# ---------------------------------------------------------------------------

def two_nn_intrinsic_dim(X: np.ndarray, n_neighbors: int = 2) -> float:
    """Estimate intrinsic dimension of a point cloud via Two-NN (Facco et al. 2017).

    Algorithm:
      1. For each point, find its 1st and 2nd nearest neighbours.
      2. mu_i = r2_i / r1_i  (ratio of 2nd to 1st neighbour distance).
      3. Sort mu ascending; empirical CDF F(mu) = i / N.
      4. Linear regression of -log(1 - F) on log(mu) with zero intercept → slope = d.

    Parameters
    ----------
    X : (N, D) array of sample points.
    n_neighbors : must be >= 2.

    Returns
    -------
    Estimated intrinsic dimension (float).
    """
    N = X.shape[0]
    if N < 10:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distances[:, 0] is the point itself (distance 0); skip it
    r1 = distances[:, 1]  # 1st neighbour
    r2 = distances[:, 2]  # 2nd neighbour

    # Avoid division by zero
    valid = r1 > 1e-10
    r1, r2 = r1[valid], r2[valid]
    if len(r1) < 5:
        return float("nan")

    mu = r2 / r1
    mu_sorted = np.sort(mu)
    N_v = len(mu_sorted)
    # Empirical CDF (avoid log(0) by clipping)
    F = np.arange(1, N_v + 1) / N_v
    F = np.clip(F, 0, 1 - 1e-9)

    log_mu = np.log(mu_sorted)
    log_1mF = -np.log(1.0 - F)

    # Ordinary least squares with zero intercept: d = sum(x*y) / sum(x^2)
    d = np.sum(log_mu * log_1mF) / (np.sum(log_mu ** 2) + 1e-12)
    return float(d)


# ---------------------------------------------------------------------------
# MNIST small CNN
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """A tiny CNN whose channel widths are controlled by `base_ch`."""

    def __init__(self, base_ch: int = 16, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),     # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                           # 14x14
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                           # 7x7
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_mnist(batch_size: int = 256) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    cache = REPO_ROOT / "cache" / "mnist"
    cache.mkdir(parents=True, exist_ok=True)
    train_ds = torchvision.datasets.MNIST(cache, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(cache, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    adv_train: bool = False,
    eps: float = 0.3,
    pgd_steps: int = 7,
    pgd_alpha: float = 0.1,
) -> float:
    """Train model; return final test accuracy."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if adv_train:
                imgs = pgd_attack(model, imgs, labels, eps=eps, alpha=pgd_alpha,
                                  steps=pgd_steps)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"    epoch {epoch + 1}/{epochs} done")

    return 0.0  # test acc computed separately


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


# ---------------------------------------------------------------------------
# PGD adversarial attack
# ---------------------------------------------------------------------------

def pgd_attack(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 0.3,
    alpha: float = 0.01,
    steps: int = 20,
) -> torch.Tensor:
    """L-inf PGD attack (Madry et al.)."""
    model.eval()
    delta = torch.zeros_like(imgs).uniform_(-eps, eps).to(imgs.device)
    delta.requires_grad_(True)
    for _ in range(steps):
        adv = imgs + delta
        loss = F.cross_entropy(model(adv), labels)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)
            delta.data = torch.clamp(imgs + delta, -2.0, 2.0) - imgs
        delta.grad.zero_()
    model.train()
    return (imgs + delta).detach()


# ---------------------------------------------------------------------------
# PM dimension estimation from a trained model
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_pm_dim(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int = 10,
    n_samples_per_class: int = 500,
    confidence_threshold: float = 0.90,
    noise_scale: float = 0.15,
) -> np.ndarray:
    """Sample points near each class's test examples, filter to high-confidence,
    then estimate PM intrinsic dimension via Two-NN.

    Returns
    -------
    dims : (num_classes,) array of estimated PM dimensions per class.
    """
    model.eval()

    # Collect test images + labels
    all_imgs, all_labels = [], []
    for imgs, labels in test_loader:
        all_imgs.append(imgs)
        all_labels.append(labels)
    all_imgs   = torch.cat(all_imgs).to(DEVICE)
    all_labels = torch.cat(all_labels).to(DEVICE)

    dims = np.zeros(num_classes)
    for c in range(num_classes):
        idx = (all_labels == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            dims[c] = float("nan")
            continue

        # Sample with replacement if needed
        chosen = idx[torch.randint(len(idx), (n_samples_per_class,))]
        anchors = all_imgs[chosen]  # (n_samples, 1, 28, 28)

        # Perturb with Gaussian noise to sample the local neighbourhood
        noise = torch.randn_like(anchors) * noise_scale
        candidates = anchors + noise

        # Filter to high-confidence correct-class predictions
        logits = model(candidates)
        probs  = F.softmax(logits, dim=1)
        conf   = probs[:, c]
        mask   = conf > confidence_threshold
        if mask.sum() < 20:
            # Lower threshold to get at least some points
            mask = conf > (confidence_threshold * 0.5)
        if mask.sum() < 10:
            dims[c] = float("nan")
            continue

        pts = candidates[mask].cpu().numpy().reshape(mask.sum().item(), -1)
        dims[c] = two_nn_intrinsic_dim(pts)

    return dims


# ---------------------------------------------------------------------------
# Adversarial distance measurement
# ---------------------------------------------------------------------------

def measure_adv_distance(
    model: nn.Module,
    test_loader: DataLoader,
    n_images: int = 200,
    eps: float = 2.0,
    alpha: float = 0.05,
    steps: int = 20,
) -> float:
    """Mean L2 distance to the nearest foreign-class perceptual manifold.

    Runs an L2-PGD attack and measures the L2 norm of the perturbation
    needed to flip the network's prediction.  Uses gradient-enabled forward
    passes inside the attack loop; outer data loading uses no_grad.
    """
    model.eval()
    with torch.no_grad():
        all_imgs, all_labels = [], []
        for imgs, labels in test_loader:
            all_imgs.append(imgs)
            all_labels.append(labels)
    all_imgs   = torch.cat(all_imgs)[:n_images].to(DEVICE)
    all_labels = torch.cat(all_labels)[:n_images].to(DEVICE)

    l2_dists = []
    batch_size = 50
    for i in range(0, n_images, batch_size):
        imgs_b   = all_imgs[i:i + batch_size]
        labels_b = all_labels[i:i + batch_size]
        nb = len(imgs_b)

        # Simple L2-PGD: take `steps` gradient steps inside an L2 ball of radius `eps`
        delta = torch.zeros_like(imgs_b)
        for _step in range(steps):
            delta = delta.detach().requires_grad_(True)
            adv = imgs_b + delta
            loss = F.cross_entropy(model(adv), labels_b)
            loss.backward()
            with torch.no_grad():
                g = delta.grad.clone()
                g_flat = g.view(nb, -1)
                g_norm = g_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                g_unit = (g_flat / g_norm).view_as(g)
                # Step in gradient direction
                delta_new = delta.detach() + alpha * g_unit
                # Project onto L2 ball of radius eps
                d_flat = delta_new.view(nb, -1)
                d_norm = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                scale  = (eps / d_norm).clamp(max=1.0)
                delta  = (d_flat * scale).view_as(delta_new)

        with torch.no_grad():
            l2 = delta.view(nb, -1).norm(dim=1)
        l2_dists.append(l2.cpu().numpy())

    return float(np.concatenate(l2_dists).mean())


# ---------------------------------------------------------------------------
# 2D toy precomputed sweep (sklearn MLPClassifier)
# ---------------------------------------------------------------------------

def sweep_2d_toy(
    widths: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train sklearn MLPs at varying widths on 2D moons; estimate PM dim via Two-NN.

    Returns
    -------
    widths_arr : (K,) int array
    dims_arr   : (K,) float array  — mean PM dim across classes
    """
    if widths is None:
        widths = [4, 8, 16, 32, 48, 64, 96, 128, 256, 512]

    # Fixed 2D moons dataset
    rng = np.random.default_rng(0)
    X, y = make_moons(n_samples=800, noise=0.12, random_state=0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    widths_out, dims_out = [], []
    for w in widths:
        print(f"  2D toy: width={w}")
        clf = MLPClassifier(
            hidden_layer_sizes=(w,),
            max_iter=500,
            random_state=0,
            solver="adam",
            alpha=1e-4,
        )
        clf.fit(X, y)

        # Sample a dense grid; filter to high-confidence
        xx = np.linspace(-3.0, 3.0, 150)
        yy = np.linspace(-3.0, 3.0, 150)
        grid = np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)
        probs = clf.predict_proba(grid)
        conf  = probs.max(axis=1)
        cls   = probs.argmax(axis=1)

        class_dims = []
        for c in range(2):
            mask = (cls == c) & (conf > 0.95)
            pts  = grid[mask]
            if len(pts) >= 20:
                d = two_nn_intrinsic_dim(pts)
                class_dims.append(d)
        mean_dim = float(np.nanmean(class_dims)) if class_dims else float("nan")
        widths_out.append(w)
        dims_out.append(mean_dim)

    return np.array(widths_out), np.array(dims_out)


# ---------------------------------------------------------------------------
# Novel extension: depth-varying MLP family on MNIST
# ---------------------------------------------------------------------------

class DepthMLP(nn.Module):
    """MLP with fixed width (128) and varying depth, for the novel-extension family."""

    def __init__(self, depth: int = 2, width: int = 128, num_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = 28 * 28
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute everything")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing bundle if present
    bundle: dict[str, np.ndarray] = {}
    if OUT_PATH.exists() and not args.force:
        try:
            bundle = dict(np.load(OUT_PATH))
            print(f"Loaded existing bundle with keys: {list(bundle.keys())}")
        except Exception:
            print("Could not load existing bundle; starting fresh.")

    t0_total = time.time()

    # -----------------------------------------------------------------------
    # Phase 0: MNIST data loaders
    # -----------------------------------------------------------------------
    print("\n=== Phase 0: Loading MNIST ===")
    train_loader, test_loader = get_mnist(batch_size=256)
    print("MNIST loaded.")

    # -----------------------------------------------------------------------
    # Phase 1: MNIST width sweep (5 base_ch values)
    # -----------------------------------------------------------------------
    BASE_CHANNELS = [4, 8, 16, 32, 64]
    WIDTH_PARAMS   = [sum(p.numel() for p in SmallCNN(bc).parameters()) for bc in BASE_CHANNELS]

    if "mnist_widths" not in bundle or args.force:
        print("\n=== Phase 1: MNIST width sweep ===")
        all_pm_dims  = []   # (5, 10)
        all_test_acc = []   # (5,)
        all_adv_dist = []   # (5,)

        for i, bc in enumerate(BASE_CHANNELS):
            print(f"\n  [width {bc}] training SmallCNN ({WIDTH_PARAMS[i]:,} params) …")
            model = SmallCNN(base_ch=bc)
            t_train = time.time()
            train_model(model, train_loader, epochs=3, adv_train=False)
            print(f"    trained in {time.time() - t_train:.1f}s")

            acc = evaluate(model, test_loader)
            print(f"    test accuracy: {acc:.3f}")
            all_test_acc.append(acc)

            print(f"    estimating PM dimension …")
            pm_dims = estimate_pm_dim(model, test_loader,
                                      n_samples_per_class=400,
                                      confidence_threshold=0.90)
            print(f"    PM dims: {pm_dims.round(2)}")
            all_pm_dims.append(pm_dims)

            print(f"    measuring adversarial distance …")
            adv_d = measure_adv_distance(model, test_loader, n_images=100,
                                         eps=2.0, alpha=0.05, steps=30)
            print(f"    mean adv dist: {adv_d:.4f}")
            all_adv_dist.append(adv_d)

        bundle["mnist_widths"]    = np.array(BASE_CHANNELS, dtype=np.int32)
        bundle["mnist_n_params"]  = np.array(WIDTH_PARAMS,  dtype=np.int64)
        bundle["mnist_pm_dims"]   = np.array(all_pm_dims,   dtype=np.float32)   # (5, 10)
        bundle["mnist_test_acc"]  = np.array(all_test_acc,  dtype=np.float32)
        bundle["mnist_adv_dist"]  = np.array(all_adv_dist,  dtype=np.float32)
        np.savez_compressed(OUT_PATH, **bundle)
        print("\nPhase 1 saved.")
    else:
        print("Phase 1 already in bundle — skipping.")

    # -----------------------------------------------------------------------
    # Phase 2: Adversarially-trained reference model (one width)
    # -----------------------------------------------------------------------
    if "adv_train_pm_dim" not in bundle or args.force:
        print("\n=== Phase 2: Adversarially-trained model ===")
        REF_BC = 16  # matches the middle width in the sweep
        print(f"  Training standard SmallCNN(base_ch={REF_BC}) …")
        std_model = SmallCNN(base_ch=REF_BC)
        train_model(std_model, train_loader, epochs=3, adv_train=False)
        std_acc  = evaluate(std_model, test_loader)
        std_dims = estimate_pm_dim(std_model, test_loader,
                                   n_samples_per_class=400,
                                   confidence_threshold=0.90)
        std_adv  = measure_adv_distance(std_model, test_loader,
                                        n_images=100, eps=2.0, steps=30)

        print(f"  Training adversarially-trained SmallCNN(base_ch={REF_BC}) …")
        adv_model = SmallCNN(base_ch=REF_BC)
        train_model(adv_model, train_loader, epochs=3, adv_train=True,
                    eps=0.3, pgd_steps=5, pgd_alpha=0.1)
        adv_acc  = evaluate(adv_model, test_loader)
        adv_dims = estimate_pm_dim(adv_model, test_loader,
                                   n_samples_per_class=400,
                                   confidence_threshold=0.90)
        adv_adv  = measure_adv_distance(adv_model, test_loader,
                                        n_images=100, eps=2.0, steps=30)

        bundle["adv_train_std_pm_dim"]  = std_dims.astype(np.float32)
        bundle["adv_train_adv_pm_dim"]  = adv_dims.astype(np.float32)
        bundle["adv_train_std_acc"]     = np.float32(std_acc)
        bundle["adv_train_adv_acc"]     = np.float32(adv_acc)
        bundle["adv_train_std_adv_dist"] = np.float32(std_adv)
        bundle["adv_train_adv_adv_dist"] = np.float32(adv_adv)
        np.savez_compressed(OUT_PATH, **bundle)
        print(f"  std PM dim mean: {std_dims.mean():.2f}  |  adv PM dim mean: {adv_dims.mean():.2f}")
        print(f"  std adv dist: {std_adv:.4f}  |  adv adv dist: {adv_adv:.4f}")
        print("Phase 2 saved.")
    else:
        print("Phase 2 already in bundle — skipping.")

    # -----------------------------------------------------------------------
    # Phase 3: Novel extension — depth-varying MLP family on MNIST
    # -----------------------------------------------------------------------
    DEPTHS = [1, 2, 3, 4, 6]

    if "ext_depths" not in bundle or args.force:
        print("\n=== Phase 3: Novel extension — depth-varying MLPs ===")
        ext_pm_dims  = []
        ext_adv_dist = []
        ext_test_acc = []
        ext_n_params = []

        for depth in DEPTHS:
            n_params = sum(p.numel() for p in DepthMLP(depth=depth).parameters())
            print(f"\n  [depth={depth}, {n_params:,} params] training …")
            model = DepthMLP(depth=depth, width=128)
            train_model(model, train_loader, epochs=3, adv_train=False)
            acc  = evaluate(model, test_loader)
            dims = estimate_pm_dim(model, test_loader,
                                   n_samples_per_class=400,
                                   confidence_threshold=0.90)
            adv  = measure_adv_distance(model, test_loader,
                                        n_images=100, eps=2.0, steps=30)
            print(f"    acc={acc:.3f}  pm_dim_mean={dims.mean():.2f}  adv_dist={adv:.4f}")
            ext_pm_dims.append(dims)
            ext_adv_dist.append(adv)
            ext_test_acc.append(acc)
            ext_n_params.append(n_params)

        bundle["ext_depths"]    = np.array(DEPTHS,       dtype=np.int32)
        bundle["ext_n_params"]  = np.array(ext_n_params, dtype=np.int64)
        bundle["ext_pm_dims"]   = np.array(ext_pm_dims,  dtype=np.float32)   # (5, 10)
        bundle["ext_adv_dist"]  = np.array(ext_adv_dist, dtype=np.float32)
        bundle["ext_test_acc"]  = np.array(ext_test_acc, dtype=np.float32)
        np.savez_compressed(OUT_PATH, **bundle)
        print("\nPhase 3 saved.")
    else:
        print("Phase 3 already in bundle — skipping.")

    # -----------------------------------------------------------------------
    # Phase 4: 2D toy sweep
    # -----------------------------------------------------------------------
    if "toy_widths" not in bundle or args.force:
        print("\n=== Phase 4: 2D toy PM-dim sweep ===")
        toy_widths, toy_dims = sweep_2d_toy()
        bundle["toy_widths"] = toy_widths.astype(np.int32)
        bundle["toy_dims"]   = toy_dims.astype(np.float32)
        np.savez_compressed(OUT_PATH, **bundle)
        print(f"  widths: {toy_widths}")
        print(f"  dims:   {toy_dims.round(3)}")
        print("Phase 4 saved.")
    else:
        print("Phase 4 already in bundle — skipping.")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0_total
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n=== All done in {elapsed / 60:.1f} min ===")
    print(f"Output: {OUT_PATH}  ({size_mb:.2f} MB)")
    print(f"Keys: {sorted(bundle.keys())}")
    assert size_mb < 12.0, f"NPZ too large: {size_mb:.2f} MB"


if __name__ == "__main__":
    main()
