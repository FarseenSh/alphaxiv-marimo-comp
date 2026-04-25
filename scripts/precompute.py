"""Pre-compute Jordan curves + diffusion solutions for the marimo notebook.

Loads the pretrained checkpoint from visual-geo-solver, generates a small
gallery of curves (circle, peanut, spiky, butterfly, paper-figure-1), runs
DDIM sampling with N seeds per curve, saves everything as a single .npz.

The notebook never runs torch — it just visualizes this artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_ROOT = REPO_ROOT / "visual-geo-solver"
sys.path.insert(0, str(SOLVER_ROOT))

from model.diffusion import UNet  # type: ignore
from schedulers.ddim import DDIM  # type: ignore


def remove_weight_prefixes(state_dict):
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in state_dict.items():
        parts = k.split(".")
        while parts and parts[0] in ("module", "_orig_mod"):
            parts = parts[1:]
        out[".".join(parts)] = v
    return out

import cv2
from scipy.interpolate import CubicSpline, splprep, splev


# ---------------------------------------------------------------------------
# Curve generation — distilled from visual-geo-solver/data/Curves.py
# ---------------------------------------------------------------------------

def generate_jordan_curve(
    H: int,
    rho_scale: float = 1.0,
    target_radius: float = 0.55,
    rotation: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
    num_points: int = 1000,
    pass_through_square: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Jordan curve in [-1, 1]^2 that passes through a known square.

    Returns:
        curve_xy: (N, 2) closed curve polyline.
        square_corners: (4, 2) inscribed square vertices.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (target_radius / np.sqrt(2))
    rot_mat = np.array([[np.cos(rotation), -np.sin(rotation)],
                        [np.sin(rotation),  np.cos(rotation)]])
    square = base @ rot_mat.T + np.array(center)

    rho = rng.random(H) * np.logspace(-0.5, -2.5, H) * rho_scale
    phi = rng.random(H) * 2 * np.pi
    r_base = np.ones_like(t)
    for h in range(1, H + 1):
        r_base += rho[h - 1] * np.sin(h * t + phi[h - 1])

    if pass_through_square:
        # Snap radial profile through the four square corners via periodic spline correction
        sx, sy = square[:, 0] - center[0], square[:, 1] - center[1]
        sq_radii = np.sqrt(sx ** 2 + sy ** 2)
        sq_angles = np.arctan2(sy, sx) % (2 * np.pi)
        idx = [int(np.argmin(np.abs(t - a))) for a in sq_angles]
        delta = sq_radii - r_base[idx]
        order = np.argsort(sq_angles)
        a_sorted = np.append(sq_angles[order], sq_angles[order][0] + 2 * np.pi)
        d_sorted = np.append(delta[order], delta[order][0])
        spline = CubicSpline(a_sorted, d_sorted, bc_type="periodic")
        r_final = r_base + spline(t)
    else:
        r_final = r_base

    x = r_final * np.cos(t) + center[0]
    y = r_final * np.sin(t) + center[1]

    # Smooth via periodic B-spline
    tck, _ = splprep([x, y], s=0, per=True)
    u = np.linspace(0, 1, num_points)
    x, y = splev(u, tck)

    if pass_through_square:
        # Snap nearest curve points exactly onto square corners
        for sx, sy in square:
            d2 = (x - sx) ** 2 + (y - sy) ** 2
            j = int(np.argmin(d2))
            x[j], y[j] = sx, sy

    return np.stack([x, y], axis=1).astype(np.float32), square.astype(np.float32)


def rasterize_curve(curve_xy: np.ndarray, image_size: int = 128, thickness: int = 1) -> np.ndarray:
    """Render polyline onto a binary 128x128 image. Convention: black curve on white."""
    img = np.full((image_size, image_size), 255, dtype=np.uint8)
    pts = (curve_xy * (image_size // 2) + image_size // 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=0, thickness=thickness)
    return img


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: str = "cpu") -> UNet:
    model = UNet(
        in_channels=2,
        out_channels=1,
        base_channels=64,
        num_levels=4,
        time_emb_dim=128,
        attention_heads=8,
        attention_locations=["bottleneck", "enc2", "enc3", "dec2", "dec3"],
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = remove_weight_prefixes(ckpt["state_dict"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  missing={len(missing)} unexpected={len(unexpected)} (tolerated)")
    model.eval()
    return model


@torch.no_grad()
def sample_squares(
    model: UNet,
    scheduler: DDIM,
    curve_image: np.ndarray,
    n_samples: int = 8,
    seed: int = 0,
    device: str = "cpu",
    return_trajectory: bool = False,
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Run DDIM sampling.

    curve_image: (H, W) uint8, black curve on white.
    Returns: (n_samples, H, W) float arrays in [-1, 1].
    """
    H, W = curve_image.shape
    cond = torch.from_numpy(curve_image).float().div(255.0).mul(2).sub(1)  # [-1, 1]
    cond = cond.unsqueeze(0).unsqueeze(0).repeat(n_samples, 1, 1, 1).to(device)

    g = torch.Generator(device=device).manual_seed(int(seed))
    x_t = torch.randn((n_samples, 1, H, W), generator=g, device=device)

    timesteps = list(range(scheduler.diffusion_steps - 1, 0, -1))
    trajectory = [] if return_trajectory else None
    for step_idx, t in enumerate(timesteps):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor.unsqueeze(1), cond)
        prev_t = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else 0
        x_t, x0_pred, _, _ = scheduler.denoise_ddim(x_t, t, noise_pred, prev_timestep=prev_t)
        if return_trajectory and step_idx % 10 == 0:
            trajectory.append(x0_pred.detach().cpu().numpy())

    out = x_t.detach().cpu().numpy().squeeze(1)  # (N, H, W)
    return out, trajectory


# ---------------------------------------------------------------------------
# Gallery: hand-curated curves that thread through the notebook
# ---------------------------------------------------------------------------

GALLERY = [
    # name, kwargs to generate_jordan_curve
    ("circle",        dict(H=1, rho_scale=0.0, target_radius=0.6, seed=42, pass_through_square=False)),
    ("hero_butterfly", dict(H=12, rho_scale=1.2, target_radius=0.55, rotation=0.4, seed=7)),
    ("peanut",        dict(H=2, rho_scale=2.5, target_radius=0.45, rotation=0.0, seed=3)),
    ("spiky_gear",    dict(H=20, rho_scale=0.6, target_radius=0.55, rotation=0.0, seed=11)),
    ("paper_figure_1", dict(H=8, rho_scale=1.0, target_radius=0.6, rotation=0.7, seed=21)),
]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ckpt_path = repo / "cache" / "checkpoints" / "checkpoint_curves.pth"
    out_path = repo / "data" / "gallery.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt_path} …")
    device = "cpu"
    model = load_model(ckpt_path, device=device)
    scheduler = DDIM(diffusion_steps=100, eta=0.0, device=device)

    bundle: dict[str, np.ndarray] = {}

    for name, kwargs in GALLERY:
        print(f"\n[{name}] generating curve …")
        curve_xy, square_corners = generate_jordan_curve(**kwargs)
        curve_img = rasterize_curve(curve_xy, image_size=128, thickness=1)
        bundle[f"{name}/curve_xy"] = curve_xy
        bundle[f"{name}/curve_img"] = curve_img
        bundle[f"{name}/gt_square"] = square_corners

        n_samples = 16 if name == "hero_butterfly" else 8
        return_traj = name == "hero_butterfly"
        print(f"  sampling {n_samples} solutions (DDIM 100 steps, CPU) …")
        import time
        t0 = time.time()
        squares, trajectory = sample_squares(
            model, scheduler, curve_img, n_samples=n_samples, seed=hash(name) & 0xFFFF,
            device=device, return_trajectory=return_traj,
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s ({elapsed / n_samples:.2f}s/sample)")
        bundle[f"{name}/samples"] = squares.astype(np.float32)
        if trajectory is not None:
            bundle[f"{name}/trajectory"] = np.stack(trajectory, axis=0).astype(np.float32)

    print(f"\nSaving → {out_path}")
    np.savez_compressed(out_path, **bundle)
    print("Done.")


if __name__ == "__main__":
    main()
