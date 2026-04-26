"""Pre-compute saliency maps, linear probe results, and permutation null distributions
for the dead_salmons.py marimo notebook.

Paper: "The Dead Salmons of AI Interpretability"
       Méloux, Dirupo, Portet, Peyrard (Grenoble, Dec 2025, arXiv:2512.18792)

What this script does:
  1. Train a small CNN on CIFAR-10 (~70% accuracy, ~100k params). Cache checkpoint.
  2. Build a same-architecture random-init network (Kaiming initialization).
  3. Select 8 representative test images (one per CIFAR-10 class).
  4. For each (image, method, model_state) compute a saliency map.
     Methods: vanilla gradient, gradient*input, SmoothGrad, integrated gradients.
  5. Train a linear probe on frozen features for "animal vs. vehicle" classification.
     Do this for both trained and random models.
  6. For each saliency map, compute an alignment score vs. a center-bias target mask.
     Build permutation null distributions (500 permutations per method/model).
  7. Sweep false-positive rates across all 4 methods.
  8. Save everything as data/dead_salmons.npz (target < 12 MB).

Re-run safety: the script caches the checkpoint; re-runs skip training.
PyTorch is required here — the notebook itself imports none of this.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Fail loudly if PyTorch is not available — do not catch silently.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
except ImportError as exc:
    print(
        f"\nFATAL: {exc}\n"
        "Install dependencies with:\n"
        "  uv pip install torch torchvision\n"
        "or activate the project venv first."
    )
    sys.exit(1)

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / "cache" / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CACHE_DIR / "dead_salmons_cnn.pt"
NPZ_PATH = DATA_DIR / "dead_salmons.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
IMAGE_SIZE = 32  # CIFAR-10 native size
NUM_CLASSES = 10
NUM_IMAGES = 8   # one per CIFAR class
NUM_PERMS = 500  # permutation null samples
SMOOTHGRAD_SAMPLES = 50
IG_STEPS = 20

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
# CIFAR superclasses: 0=airplane,1=automobile,8=ship,9=truck → vehicle;
# 2=bird,3=cat,4=deer,5=dog,6=frog,7=horse → animal
ANIMAL_LABEL = 1  # 1=animal, 0=vehicle in binary probe
SUPERCLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Three conv blocks + one FC head. ~95k parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),          # 16x16
            nn.Dropout2d(0.15),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),          # 8x8
            nn.Dropout2d(0.15),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),          # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        """Return flattened features from the last conv block."""
        return self.features(x).flatten(1)


def kaiming_reinit(model: nn.Module) -> nn.Module:
    """Return a copy of the model with fresh Kaiming initialization."""
    import copy
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_cifar_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=str(CACHE_DIR / "cifar"), train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(CACHE_DIR / "cifar"), train=False, download=True, transform=test_tf
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )
    return train_loader, test_loader, test_ds


def train_model(model: nn.Module, train_loader, test_loader) -> float:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader)
        print(f"  epoch {epoch:2d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.3f}")

    return evaluate(model, test_loader)


def evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Test image selection
# ---------------------------------------------------------------------------

def select_test_images(test_ds, model: nn.Module):
    """Pick one image per class that the trained model classifies correctly."""
    model.eval()
    selected_imgs = {}   # class_idx -> (tensor, uint8_arr)
    selected_labels = {}

    # Normalization constants (to recover raw uint8 for display)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu()
            for i, (img, lab, pred) in enumerate(zip(imgs, labels, preds)):
                c = lab.item()
                if c not in selected_imgs and pred.item() == c:
                    # Recover uint8 for display
                    raw = (img * std + mean).clamp(0, 1)
                    raw_u8 = (raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    selected_imgs[c] = (img.unsqueeze(0), raw_u8)
                    selected_labels[c] = c
                if len(selected_imgs) == NUM_CLASSES:
                    break
            if len(selected_imgs) == NUM_CLASSES:
                break

    # Sort by class index, then keep first 8 (notebook hero grids are 2x4 = 8).
    order = sorted(selected_imgs.keys())[:8]
    imgs_tensor = torch.cat([selected_imgs[c][0] for c in order])  # (8, 3, 32, 32)
    imgs_uint8 = np.stack([selected_imgs[c][1] for c in order])    # (8, 32, 32, 3)
    labels_arr = np.array([selected_labels[c] for c in order], dtype=np.int32)
    return imgs_tensor, imgs_uint8, labels_arr


# ---------------------------------------------------------------------------
# Saliency methods
# ---------------------------------------------------------------------------

def _prep(imgs_tensor: torch.Tensor):
    """Return a fresh leaf tensor with grad enabled, on DEVICE."""
    x = imgs_tensor.detach().clone().to(DEVICE).float().requires_grad_(True)
    return x


@torch.enable_grad()
def vanilla_gradient(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Gradient of the class logit w.r.t. input pixels. (N, 32, 32)"""
    model.eval()
    x = _prep(imgs)
    out = model(x)
    scores = out[range(len(labels)), labels.tolist()]
    scores.sum().backward()
    sal = x.grad.data.abs().max(dim=1)[0]  # max across channels
    return sal.detach().cpu().numpy().astype(np.float16)


@torch.enable_grad()
def gradient_times_input(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Gradient * input (element-wise). (N, 32, 32)"""
    model.eval()
    x = _prep(imgs)
    out = model(x)
    scores = out[range(len(labels)), labels.tolist()]
    scores.sum().backward()
    sal = (x.grad.data * x.detach()).abs().max(dim=1)[0]
    return sal.detach().cpu().numpy().astype(np.float16)


@torch.enable_grad()
def smoothgrad(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor,
               n_samples: int = SMOOTHGRAD_SAMPLES, noise_level: float = 0.15) -> np.ndarray:
    """SmoothGrad: average gradients over noisy copies of the input. (N, 32, 32)"""
    model.eval()
    accumulated = torch.zeros(imgs.shape[0], imgs.shape[2], imgs.shape[3], device=DEVICE)
    sigma = noise_level * (imgs.max() - imgs.min()).item()
    for _ in range(n_samples):
        x = _prep(imgs + torch.randn_like(imgs) * sigma)
        out = model(x)
        scores = out[range(len(labels)), labels.tolist()]
        scores.sum().backward()
        accumulated += x.grad.data.abs().max(dim=1)[0].detach()
    sal = accumulated / n_samples
    return sal.cpu().numpy().astype(np.float16)


@torch.enable_grad()
def integrated_gradients(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor,
                          steps: int = IG_STEPS) -> np.ndarray:
    """Integrated gradients from a zero baseline. (N, 32, 32)"""
    model.eval()
    baseline = torch.zeros_like(imgs).to(DEVICE)
    imgs_d = imgs.to(DEVICE)
    accumulated = torch.zeros(imgs.shape[0], imgs.shape[2], imgs.shape[3], device=DEVICE)
    for k in range(steps):
        alpha = k / (steps - 1)
        x = _prep(baseline + alpha * (imgs_d - baseline))
        out = model(x)
        scores = out[range(len(labels)), labels.tolist()]
        scores.sum().backward()
        accumulated += x.grad.data.abs().max(dim=1)[0].detach()
    # Multiply by (input - baseline) to get IG
    delta = (imgs_d - baseline).abs().max(dim=1)[0]
    sal = (accumulated / steps) * delta
    return sal.cpu().numpy().astype(np.float16)


METHODS = [
    ("vanilla_gradient", vanilla_gradient),
    ("gradient_times_input", gradient_times_input),
    ("smoothgrad", smoothgrad),
    ("integrated_gradients", integrated_gradients),
]


def compute_all_saliency(models_dict: dict, imgs_tensor: torch.Tensor, labels_tensor: torch.Tensor):
    """
    Returns dict: method_name -> array of shape (2, 8, 32, 32) float16.
    Axis 0: [trained_model, random_model].
    """
    out = {}
    for mname, mfn in METHODS:
        print(f"  [saliency] {mname}")
        rows = []
        for mkey in ["trained", "random"]:
            model = models_dict[mkey].to(DEVICE).eval()
            s = mfn(model, imgs_tensor, labels_tensor)
            # Normalize each map to [0, 1] so all 8 are on the same scale
            s_f = s.astype(np.float32)
            for i in range(len(s_f)):
                mn, mx = s_f[i].min(), s_f[i].max()
                if mx > mn:
                    s_f[i] = (s_f[i] - mn) / (mx - mn)
            rows.append(s_f.astype(np.float16))
        out[mname] = np.stack(rows, axis=0)  # (2, 8, 32, 32)
    return out


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def extract_features(model: nn.Module, loader, device: str):
    """Return (N, D) features and (N,) binary labels (animal=1, vehicle=0)."""
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            f = model.get_features(imgs.to(device))
            feats_list.append(f.cpu().numpy())
            # Map to binary superclass
            binary = np.array([SUPERCLASS[l.item()] for l in labs], dtype=np.int32)
            labels_list.append(binary)
    return np.vstack(feats_list), np.concatenate(labels_list)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def logistic_regression_numpy(X_train, y_train, X_test, y_test,
                               lr=0.05, epochs=200, reg=1e-3):
    """Minimal logistic regression in NumPy. Returns (train_acc, test_acc, weights)."""
    n, d = X_train.shape
    # Normalize
    mu = X_train.mean(0)
    sigma = X_train.std(0) + 1e-8
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma

    W = np.zeros((d, 2))   # (features, 2 classes)
    b = np.zeros(2)

    for ep in range(epochs):
        logits = Xtr @ W + b       # (n, 2)
        probs = softmax(logits)
        one_hot = np.eye(2)[y_train]
        grad_logits = (probs - one_hot) / n
        grad_W = Xtr.T @ grad_logits + reg * W
        grad_b = grad_logits.sum(0)
        W -= lr * grad_W
        b -= lr * grad_b
        if ep % 50 == 0:
            loss = -np.log(probs[range(n), y_train] + 1e-9).mean()

    train_pred = (Xtr @ W + b).argmax(1)
    test_pred = (Xte @ W + b).argmax(1)
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    return float(train_acc), float(test_acc), W.astype(np.float32)


# ---------------------------------------------------------------------------
# Permutation null distribution
# ---------------------------------------------------------------------------

def center_bias_mask(size: int = IMAGE_SIZE) -> np.ndarray:
    """Gaussian-weighted center mask: interpretability tools are expected to
    highlight central, semantically relevant regions. Shape (size, size)."""
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    mask = np.exp(-(x**2 + y**2) / (2 * 0.35**2))
    return (mask / mask.max()).astype(np.float32)


def alignment_score(sal_map: np.ndarray, target_mask: np.ndarray) -> float:
    """Pearson correlation between saliency map and target mask."""
    s = sal_map.astype(np.float32).ravel()
    t = target_mask.ravel()
    if s.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(s, t)[0, 1])


def compute_permutation_null(saliency_maps: np.ndarray, target_mask: np.ndarray,
                              n_perms: int = NUM_PERMS, rng_seed: int = 42) -> tuple:
    """
    saliency_maps: (N_images, H, W) for one method/model.
    Returns:
        observed_scores: (N_images,) float32
        null_dist: (N_images, n_perms) float32  — per-image permuted scores
    """
    rng = np.random.default_rng(rng_seed)
    n_img = saliency_maps.shape[0]
    observed = np.array([alignment_score(saliency_maps[i], target_mask)
                         for i in range(n_img)], dtype=np.float32)
    # Build null by permuting the saliency map pixels
    h, w = target_mask.shape
    null_dist = np.zeros((n_img, n_perms), dtype=np.float32)
    for p in range(n_perms):
        flat_mask = target_mask.ravel().copy()
        rng.shuffle(flat_mask)
        shuffled = flat_mask.reshape(h, w)
        for i in range(n_img):
            null_dist[i, p] = alignment_score(saliency_maps[i], shuffled)
    return observed, null_dist


# ---------------------------------------------------------------------------
# False positive rate sweep
# ---------------------------------------------------------------------------

def compute_false_positive_rate(saliency_maps_random: np.ndarray,
                                 target_mask: np.ndarray,
                                 threshold_percentile: float = 95.0,
                                 n_perms: int = 200) -> float:
    """
    Fraction of random-model saliency maps that score above the 'threshold_percentile'
    of the null distribution (i.e., would be declared 'significant' without correction).
    """
    obs, null = compute_permutation_null(saliency_maps_random, target_mask,
                                         n_perms=n_perms, rng_seed=99)
    # For each image, what fraction of null scores does the observed exceed?
    p_vals = (null >= obs[:, None]).mean(axis=1)  # small p-val → apparent signal
    # FPR: fraction of images with p < 0.05 (would be "significant" without correction)
    fpr = float((p_vals < 0.05).mean())
    return fpr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 60)
    print("Dead Salmons pre-compute  (arXiv:2512.18792)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 1. Data
    print("\n[1/7] Loading CIFAR-10 ...")
    train_loader, test_loader, test_ds = get_cifar_loaders()

    # 2. Trained model
    print("\n[2/7] Trained model ...")
    trained_model = SmallCNN()
    if CKPT_PATH.exists():
        print(f"  Loading checkpoint from {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        trained_model.load_state_dict(ckpt["state_dict"])
        trained_acc = ckpt.get("test_acc", 0.0)
        print(f"  Cached test accuracy: {trained_acc:.3f}")
    else:
        print("  No checkpoint found — training from scratch ...")
        trained_acc = train_model(trained_model, train_loader, test_loader)
        torch.save({"state_dict": trained_model.state_dict(), "test_acc": trained_acc},
                   CKPT_PATH)
        print(f"  Saved checkpoint. Final test accuracy: {trained_acc:.3f}")

    # 3. Random model
    print("\n[3/7] Random-init model ...")
    random_model = kaiming_reinit(trained_model)
    random_acc = evaluate(random_model.to(DEVICE), test_loader)
    print(f"  Random-init test accuracy: {random_acc:.3f} (expected ~0.10)")

    models = {"trained": trained_model.cpu(), "random": random_model.cpu()}

    # 4. Select test images
    print("\n[4/7] Selecting test images (one per class) ...")
    imgs_tensor, imgs_uint8, labels_arr = select_test_images(
        test_ds, trained_model.to(DEVICE)
    )
    trained_model.cpu()
    labels_tensor = torch.from_numpy(labels_arr)
    print(f"  Selected classes: {[CIFAR_CLASSES[c] for c in labels_arr]}")

    # 5. Saliency maps
    print("\n[5/7] Computing saliency maps (4 methods x 2 models x 8 images) ...")
    saliency = compute_all_saliency(models, imgs_tensor, labels_tensor)
    # saliency[method_name] = (2, 8, 32, 32) float16

    # 6. Linear probe
    print("\n[6/7] Linear probe (animal vs. vehicle) ...")
    probe_results = {}
    full_loader_train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=str(CACHE_DIR / "cifar"), train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        ),
        batch_size=512, shuffle=False, num_workers=0
    )
    for mkey in ["trained", "random"]:
        m = models[mkey].to(DEVICE)
        print(f"  Extracting features: {mkey} ...")
        X_tr, y_tr = extract_features(m, full_loader_train, DEVICE)
        X_te, y_te = extract_features(m, test_loader, DEVICE)
        m.cpu()
        print(f"  Training logistic regression probe ({X_tr.shape[0]} samples, {X_tr.shape[1]} dims) ...")
        tr_acc, te_acc, probe_W = logistic_regression_numpy(X_tr, y_tr, X_te, y_te)
        print(f"    {mkey}: train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")
        probe_results[mkey] = {"train_acc": tr_acc, "test_acc": te_acc, "weights": probe_W}

    # 7. Permutation null distributions
    print("\n[7/7] Permutation null distributions ...")
    target_mask = center_bias_mask(IMAGE_SIZE)
    perm_results = {}
    for mname, _ in METHODS:
        print(f"  [{mname}]")
        for mi, mkey in enumerate(["trained", "random"]):
            sal_maps = saliency[mname][mi].astype(np.float32)  # (8, 32, 32)
            obs, null = compute_permutation_null(sal_maps, target_mask,
                                                  n_perms=NUM_PERMS, rng_seed=42 + mi)
            perm_results[(mname, mkey)] = {"observed": obs, "null_dist": null}
            mean_obs = obs.mean()
            null_95 = np.percentile(null, 95, axis=1).mean()
            print(f"    {mkey}: mean_observed={mean_obs:.4f}  null_95pct={null_95:.4f}")

    # False positive rate sweep (use 200 permutations for speed)
    print("\n  False positive rate sweep ...")
    fpr_arr = np.zeros(4, dtype=np.float32)
    for i, (mname, _) in enumerate(METHODS):
        sal_rand = saliency[mname][1].astype(np.float32)
        fpr = compute_false_positive_rate(sal_rand, target_mask, n_perms=200)
        fpr_arr[i] = fpr
        print(f"    {mname}: FPR = {fpr:.3f}")

    # -----------------------------------------------------------------------
    # Pack and save
    # -----------------------------------------------------------------------
    print("\nPacking arrays ...")
    bundle = {}

    # Images
    bundle["imgs_uint8"] = imgs_uint8.astype(np.uint8)              # (8, 32, 32, 3)
    bundle["labels"] = labels_arr.astype(np.int32)                  # (8,)
    bundle["class_names"] = np.array(CIFAR_CLASSES)                 # (10,) str
    bundle["method_names"] = np.array([m[0] for m in METHODS])      # (4,) str
    bundle["target_mask"] = target_mask.astype(np.float32)          # (32, 32)

    # Model accuracies
    bundle["trained_acc"] = np.float32(trained_acc)
    bundle["random_acc"] = np.float32(random_acc)

    # Saliency: (4 methods, 2 states, 8 images, 32, 32) float16
    sal_stack = np.stack([saliency[m[0]] for m in METHODS], axis=0)  # (4, 2, 8, 32, 32)
    bundle["saliency"] = sal_stack.astype(np.float16)

    # Probe results
    for mkey in ["trained", "random"]:
        bundle[f"probe_{mkey}_train_acc"] = np.float32(probe_results[mkey]["train_acc"])
        bundle[f"probe_{mkey}_test_acc"] = np.float32(probe_results[mkey]["test_acc"])

    # Permutation null: (4, 2, 8) observed scores and (4, 2, 8, N_PERMS) null dists
    obs_arr = np.zeros((4, 2, 8), dtype=np.float32)
    null_arr = np.zeros((4, 2, 8, NUM_PERMS), dtype=np.float32)
    for i, (mname, _) in enumerate(METHODS):
        for j, mkey in enumerate(["trained", "random"]):
            obs_arr[i, j] = perm_results[(mname, mkey)]["observed"]
            null_arr[i, j] = perm_results[(mname, mkey)]["null_dist"]
    bundle["perm_observed"] = obs_arr         # (4, 2, 8)
    bundle["perm_null"] = null_arr.astype(np.float16)  # (4, 2, 8, 500)

    # False positive rates
    bundle["fpr"] = fpr_arr                   # (4,)

    np.savez_compressed(str(NPZ_PATH), **bundle)

    size_mb = NPZ_PATH.stat().st_size / 1e6
    elapsed = time.time() - t_start
    print(f"\nSaved {NPZ_PATH}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Sanity check
    loaded = np.load(str(NPZ_PATH), allow_pickle=True)
    print("\nStored arrays:")
    for k in sorted(loaded.keys()):
        arr = loaded[k]
        print(f"  {k:40s}  {str(arr.shape):25s}  {arr.dtype}")

    if size_mb > 12.0:
        print(f"\nWARNING: .npz is {size_mb:.2f} MB, exceeds 12 MB budget!")
    else:
        print(f"\nSize OK ({size_mb:.2f} MB < 12 MB budget).")
    print("Done.")


if __name__ == "__main__":
    main()
