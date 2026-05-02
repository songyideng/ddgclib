"""Generate Gong (2025) comparison plots for Case 2.

This script reruns the stock Case 2 bridge-formation problem, then creates
literature comparison PNGs against Gong et al. (2025), using:

1. A manually transcribed approximation of the corrected experimental
   F* - V* curve from Gong Fig. 6(b) for water, 5-5 mm particles, S*=0.2.
2. Direct figure-image extraction from the supplied Gong PDF to provide the
   original literature charts alongside the current Case 2 results.

The current Stage-2 solver is dynamic and bridge-onset driven, so the
comparison is informative rather than a strict one-to-one validation.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_coupled_collision.src._core import OUT_ROOT, run_case
from cases_dynamic.liquid_bridge_coupled_collision.src._params import bridge_formation_config


OUT_DIR = OUT_ROOT / "Case_2"
COMPARE_DIR = OUT_DIR / "fig_gong2025"
GONG_PDF = Path(
    "/Users/songyideng/Downloads/Gong2025_Liquid bridge morphology and capillary bridge force between particles.pdf"
)


def _extract_gong_images(pdf_path: Path, out_dir: Path) -> dict[str, Path]:
    """Extract the embedded Gong figure images needed for the comparison."""

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))

    # These page/image indices were inspected once from the supplied PDF:
    # page 6 image 0 -> Fig. 4 (water morphology + F*-V*)
    # page 7 image 0 -> Fig. 6 (water contact-angle correction + corrected F*-V*)
    # page 9 image 0 -> Fig. 8 (F*-S* panels for different V*)
    wanted = {
        "fig4_water_volume": (6, 0),
        "fig6_water_corrected": (7, 0),
        "fig8_separation_sweep": (9, 0),
    }

    paths: dict[str, Path] = {}
    for key, (page_no, image_idx) in wanted.items():
        page = reader.pages[page_no - 1]
        images = list(page.images)
        if image_idx >= len(images):
            raise RuntimeError(
                f"Missing expected image {image_idx} on Gong PDF page {page_no}."
            )
        image = images[image_idx]
        suffix = image.name.split(".")[-1].lower()
        path = out_dir / f"{key}.{suffix}"
        path.write_bytes(image.data)
        paths[key] = path

    return paths


def _load_history() -> list[dict]:
    """Rerun stock Case 2 and return the fresh history."""

    return run_case(
        bridge_formation_config(),
        out_dir=OUT_DIR,
        save_fig=True,
        save_results=True,
        verbose=True,
    )


def _case2_dimensionless(history: list[dict]) -> dict[str, np.ndarray]:
    cfg = bridge_formation_config()
    R = cfg.particle_radius
    gamma = cfg.gamma

    bridge = [row for row in history if row["bridge_formed"]]
    if not bridge:
        raise RuntimeError("Case 2 did not form a bridge, so Gong comparison cannot proceed.")

    s_star = np.array([row["sep"] / R for row in bridge], dtype=float)
    v_star = np.array([row["mesh_volume"] / (R**3) for row in bridge], dtype=float)
    f_star = np.array(
        [row["capillary_force_mag"] / (math.pi * R * gamma) for row in bridge],
        dtype=float,
    )
    t_ms = np.array([row["t"] * 1.0e3 for row in bridge], dtype=float)

    return {
        "s_star": s_star,
        "v_star": v_star,
        "f_star": f_star,
        "t_ms": t_ms,
    }


def _gong_fig6b_reference() -> tuple[np.ndarray, np.ndarray]:
    """Approximate water experimental points from Gong Fig. 6(b), S*=0.2.

    The values below are a light manual transcription from the figure image
    embedded in the supplied PDF. They are used only for a qualitative
    comparison against the current dynamic Stage-2 case.
    """

    v_star = np.array([0.01, 0.03, 0.05, 0.08, 0.11, 0.15, 0.20, 0.25, 0.35, 0.43, 0.48])
    f_star = np.array([0.50, 0.92, 1.28, 1.78, 1.85, 1.95, 2.08, 2.12, 2.18, 2.07, 1.98])
    return v_star, f_star


def _save_force_volume_compare(
    metrics: dict[str, np.ndarray],
    lit_image_path: Path,
    out_path: Path,
) -> Path:
    lit_v, lit_f = _gong_fig6b_reference()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    lit_image = mpimg.imread(lit_image_path)
    axes[0].imshow(lit_image)
    axes[0].axis("off")
    axes[0].set_title("Gong 2025: water, corrected Fig. 6", pad=10)

    v_case = metrics["v_star"]
    f_case = metrics["f_star"]
    axes[1].plot(lit_v, lit_f, "o-", color="#111111", linewidth=1.5, markersize=5, label="Gong 2025 experiment (approx.)")
    axes[1].axvspan(
        float(np.min(v_case)),
        float(np.max(v_case)),
        color="#d95f02",
        alpha=0.18,
        label="Current Case 2 V* band",
    )
    axes[1].vlines(
        float(np.mean(v_case)),
        float(np.min(f_case)),
        float(np.max(f_case)),
        colors="#d95f02",
        linewidth=2.6,
        label="Current Case 2 F* band",
    )
    axes[1].plot(
        float(np.mean(v_case)),
        float(np.max(f_case)),
        marker="*",
        markersize=14,
        color="#d95f02",
    )
    axes[1].set_xlim(0.0, 0.52)
    axes[1].set_ylim(0.0, 3.0)
    axes[1].set_xlabel(r"$V^*$")
    axes[1].set_ylabel(r"$F^* = F/(\pi R \gamma)$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Dimensionless Force vs Volume", pad=10)
    axes[1].text(
        0.03,
        0.95,
        (
            f"Case 2 mean V* = {np.mean(v_case):.3f}\n"
            f"Case 2 F* range = {np.min(f_case):.3f} - {np.max(f_case):.3f}"
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Case 2 vs Gong 2025: Force-Volume Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_force_gap_compare(
    metrics: dict[str, np.ndarray],
    lit_image_path: Path,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    lit_image = mpimg.imread(lit_image_path)
    axes[0].imshow(lit_image)
    axes[0].axis("off")
    axes[0].set_title("Gong 2025: Fig. 8 separation sweep", pad=10)

    s_star = metrics["s_star"]
    f_star = metrics["f_star"]
    t_ms = metrics["t_ms"]
    sc = axes[1].scatter(s_star, f_star, c=t_ms, cmap="viridis", s=34)
    axes[1].plot(s_star, f_star, color="#1f3b4d", linewidth=1.2, alpha=0.8)
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"$S^* = S/R$")
    axes[1].set_ylabel(r"$F^* = F/(\pi R \gamma)$")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Current Case 2 bridge segment", pad=10)
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("time [ms]")
    axes[1].text(
        0.03,
        0.95,
        (
            f"S* range = {np.min(s_star):.3f} - {np.max(s_star):.3f}\n"
            f"F* range = {np.min(f_star):.3f} - {np.max(f_star):.3f}\n"
            "Trend: force rises as the particles approach."
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Case 2 vs Gong 2025: Force-Separation Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_regime_map(metrics: dict[str, np.ndarray], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))

    # Simplified regime map from Gong abstract / discussion:
    # large V* and small S* move the bridge away from the simple concave shape.
    ax.axvspan(0.0, 0.10, color="#f4d35e", alpha=0.26, label=r"$S^* < 0.1$")
    ax.axhspan(0.10, 0.70, color="#ee964b", alpha=0.16, label=r"$V^* > 0.1$")
    ax.axvline(0.10, color="#b56576", linestyle="--", linewidth=1.3)
    ax.axhline(0.10, color="#b56576", linestyle="--", linewidth=1.3)

    ax.plot(metrics["s_star"], metrics["v_star"], color="#114b5f", linewidth=1.4)
    ax.scatter(metrics["s_star"], metrics["v_star"], c=metrics["t_ms"], cmap="viridis", s=36)

    ax.set_xlabel(r"$S^* = S/R$")
    ax.set_ylabel(r"$V^* = V/R^3$")
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(0.0, max(0.55, float(np.max(metrics['v_star']) + 0.05)))
    ax.grid(alpha=0.22)
    ax.set_title("Gong 2025 morphology map with current Case 2 path")
    ax.text(
        0.13,
        0.47,
        "Current Case 2 stays at\nlarge V* but moderate S*.\nIt does not enter the\nS* < 0.1 corner.",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_morphology_compare(
    lit_image_path: Path,
    bridge_png: Path,
    final_png: Path,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    axes[0].imshow(mpimg.imread(lit_image_path))
    axes[0].axis("off")
    axes[0].set_title("Gong 2025 water morphology")

    axes[1].imshow(mpimg.imread(bridge_png))
    axes[1].axis("off")
    axes[1].set_title("Current Case 2 bridge onset")

    axes[2].imshow(mpimg.imread(final_png))
    axes[2].axis("off")
    axes[2].set_title("Current Case 2 final bridge")

    fig.suptitle("Morphology Comparison: Gong 2025 vs current Case 2", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    history = _load_history()
    metrics = _case2_dimensionless(history)
    gong_images = _extract_gong_images(GONG_PDF, COMPARE_DIR / "gong_ref")

    force_volume_png = _save_force_volume_compare(
        metrics,
        gong_images["fig6_water_corrected"],
        COMPARE_DIR / "gong2025_force_volume_compare.png",
    )
    force_gap_png = _save_force_gap_compare(
        metrics,
        gong_images["fig8_separation_sweep"],
        COMPARE_DIR / "gong2025_force_gap_compare.png",
    )
    regime_png = _save_regime_map(
        metrics,
        COMPARE_DIR / "gong2025_regime_map.png",
    )
    morphology_png = _save_morphology_compare(
        gong_images["fig4_water_volume"],
        OUT_DIR / "fig" / "mesh_bridge.png",
        OUT_DIR / "fig" / "mesh_final.png",
        COMPARE_DIR / "gong2025_morphology_compare.png",
    )

    print("Wrote comparison PNGs:")
    for path in [force_volume_png, force_gap_png, regime_png, morphology_png]:
        print(path)


if __name__ == "__main__":
    main()
