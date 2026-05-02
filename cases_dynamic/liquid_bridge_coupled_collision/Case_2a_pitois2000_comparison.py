"""Generate preliminary Pitois (2000) comparison plots for the catenoid-start case.

This comparison is built on top of the new pre-bridged `Case_2a` path:

1. Two particles start from a bridge mesh seeded by the equilibrium catenoid.
2. Particle motion is prescribed at constant speed, matching the stage-driven
   experiment more closely than the free-DEM collision cases.
3. We compare signed axial bridge force histories against the experimental
   figure panels from Pitois et al. (2000).

Important limitation:
    The current Stage-2 solver does not yet implement topological bridge
    rupture. The fourth plot therefore uses a neck-growth separation proxy
    instead of a true rupture distance.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_coupled_collision.src._core import OUT_ROOT, run_case
from cases_dynamic.liquid_bridge_coupled_collision.src._params import catenoid_initialized_case2_config


PITOIS_PDF = Path(
    "/Users/songyideng/Downloads/Olivier Pitois 2000, Pascal Moucheront, Xavier Chateau. Liquid Bridge between Two Moving Spheres- An Experimental Study of Viscosity Effects.pdf"
)
OUT_DIR = OUT_ROOT / "Case_2a_pitois2000"
RUNS_DIR = OUT_DIR / "runs"
FIG_DIR = OUT_DIR / "fig"
REF_DIR = FIG_DIR / "pitois_ref"
PITOIS_FALLBACK_DIR = Path("/tmp/pitois_figs")


def _extract_pitois_images(pdf_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback_map = {
        "fig5_separation": PITOIS_FALLBACK_DIR / "p5_img1.tiff",
        "fig6_approach": PITOIS_FALLBACK_DIR / "p6_img0.tiff",
        "fig7_fmax": PITOIS_FALLBACK_DIR / "p6_img1.tiff",
        "fig8_rupture": PITOIS_FALLBACK_DIR / "p7_img0.jpg",
    }

    if all(path.exists() for path in fallback_map.values()):
        copied_paths: dict[str, Path] = {}
        for key, src_path in fallback_map.items():
            dst_path = out_dir / f"{key}{src_path.suffix.lower()}"
            shutil.copyfile(src_path, dst_path)
            copied_paths[key] = dst_path
        return copied_paths

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        missing = ", ".join(str(path) for path in fallback_map.values() if not path.exists())
        raise RuntimeError(
            "Could not extract the Pitois reference panels because `pypdf` is not installed "
            f"and the local fallback images are missing: {missing}"
        ) from exc

    reader = PdfReader(str(pdf_path))
    wanted = {
        "fig5_separation": (5, 1),
        "fig6_approach": (6, 0),
        "fig7_fmax": (6, 1),
        "fig8_rupture": (7, 0),
    }

    paths: dict[str, Path] = {}
    for key, (page_no, image_idx) in wanted.items():
        page = reader.pages[page_no - 1]
        images = list(page.images)
        if image_idx >= len(images):
            raise RuntimeError(f"Missing expected image {image_idx} on page {page_no}.")
        image = images[image_idx]
        suffix = image.name.split(".")[-1].lower()
        path = out_dir / f"{key}.{suffix}"
        path.write_bytes(image.data)
        paths[key] = path
    return paths


def _pitois_base_config(*, initial_sep: float, v_particle: float, n_steps: int, record_every: int) -> object:
    return replace(
        catenoid_initialized_case2_config(),
        particle_radius=4.0e-3,
        gamma=2.10e-2,
        mu_f=1.0e-1,
        rho_f=965.0,
        film_thickness=1.0e-3,
        initial_sep=initial_sep,
        v_approach=v_particle,
        dt=1.0e-3,
        fluid_substeps=1,
        n_steps=n_steps,
        record_every=record_every,
        mesh_snapshot_every=0,
        equilibrium_catenoid_refinement=2,
        prescribed_particle_motion=True,
        film_damping=1.0e-1,
        max_surface_acceleration=5.0,
        enable_volume_projection=False,
    )


def _run_cached(case_name: str, config) -> list[dict]:
    run_dir = RUNS_DIR / case_name
    history_path = run_dir / "results" / "history.json"
    if history_path.exists():
        payload = json.loads(history_path.read_text())
        return payload["history"]

    history = run_case(
        config,
        out_dir=run_dir,
        save_fig=False,
        save_results=False,
        verbose=False,
    )

    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"config": asdict(config), "history": history}, indent=2))
    return history


def _arrays(history: list[dict], *, radius: float) -> dict[str, np.ndarray]:
    rows = history
    d_over_r = np.array([float(row["sep"]) / radius for row in rows], dtype=float)
    force_mn = np.array([float(row["total_force_axial"]) * 1.0e3 for row in rows], dtype=float)
    neck_um = np.array([float(row["neck_radius"]) * 1.0e6 for row in rows], dtype=float)
    time_s = np.array([float(row["t"]) for row in rows], dtype=float)
    return {
        "d_over_r": d_over_r,
        "force_mn": force_mn,
        "neck_um": neck_um,
        "time_s": time_s,
    }


def _force_decay_proxy(history: list[dict], *, radius: float, peak_fraction: float = 0.10) -> float | None:
    arr = _arrays(history, radius=radius)
    force = arr["force_mn"]
    d_over_r = arr["d_over_r"]
    peak = float(np.max(force))
    if peak <= 0.0:
        return None

    threshold = peak_fraction * peak
    hit = np.where(force <= threshold)[0]
    if hit.size == 0:
        return None
    return float(d_over_r[int(hit[0])])


def _neck_growth_proxy(history: list[dict], *, radius: float, threshold_um: float = 20.0) -> float | None:
    arr = _arrays(history, radius=radius)
    neck_um = arr["neck_um"]
    d_over_r = arr["d_over_r"]
    hit = np.where(neck_um >= threshold_um)[0]
    if hit.size == 0:
        return None
    return float(d_over_r[int(hit[0])])


def _rupture_proxy_distance(history: list[dict], *, radius: float) -> float | None:
    force_proxy = _force_decay_proxy(history, radius=radius, peak_fraction=0.10)
    if force_proxy is not None:
        return force_proxy
    return _neck_growth_proxy(history, radius=radius, threshold_um=20.0)


def _save_force_compare(
    *,
    lit_image_path: Path,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    summary_text: str,
    out_path: Path,
    xscale: str = "linear",
    yscale: str = "linear",
    ylabel: str = "Signed axial force [mN]",
    use_abs_force: bool = False,
    show_zero_line: bool = False,
    clip_note: str | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    y_plot = np.abs(curve_y) if use_abs_force else np.array(curve_y, dtype=float)

    axes[0].imshow(mpimg.imread(lit_image_path))
    axes[0].axis("off")
    axes[0].set_title("Pitois 2000 experimental panel", pad=10)

    axes[1].plot(curve_x, y_plot, color="#114b5f", linewidth=1.8)
    axes[1].scatter(
        curve_x[:: max(1, len(curve_x) // 18)],
        y_plot[:: max(1, len(curve_x) // 18)],
        s=20,
        color="#114b5f",
    )
    axes[1].set_xscale(xscale)
    axes[1].set_yscale(yscale)
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_xlabel(r"$D / R$")
    axes[1].set_ylabel(ylabel)
    axes[1].grid(alpha=0.28)
    if show_zero_line and yscale == "linear":
        axes[1].axhline(0.0, color="#888888", linewidth=0.9, linestyle="--", alpha=0.8)
    axes[1].set_title(title, pad=10)
    axes[1].text(
        0.03,
        0.96,
        summary_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )
    if clip_note:
        axes[1].text(
            0.03,
            0.08,
            clip_note,
            transform=axes[1].transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_fmax_compare(
    *,
    lit_image_path: Path,
    particle_speeds_um_s: np.ndarray,
    fmax_mn: np.ndarray,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

    axes[0].imshow(mpimg.imread(lit_image_path))
    axes[0].axis("off")
    axes[0].set_title("Pitois 2000 Fig. 7", pad=10)

    axes[1].plot(particle_speeds_um_s, fmax_mn, "o-", color="#8c2f39", linewidth=1.8, markersize=6)
    axes[1].set_xlabel("Prescribed particle speed [um/s]")
    axes[1].set_ylabel(r"$F_{\max}$ [mN]")
    axes[1].set_xlim(0.0, 20.0)
    axes[1].set_ylim(0.0, 0.5)
    axes[1].grid(alpha=0.28)
    axes[1].set_title("Current catenoid-start case: approach-force peak", pad=10)
    axes[1].text(
        0.03,
        0.96,
        "Symmetric two-particle run\nrelative speed = 2 x particle speed",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )
    if float(np.max(fmax_mn)) > 0.5:
        axes[1].text(
            0.03,
            0.08,
            "Pitois axis window kept fixed.\nValues above 0.5 mN are clipped.",
            transform=axes[1].transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_rupture_proxy_compare(
    *,
    lit_image_path: Path,
    particle_speeds_um_s: np.ndarray,
    proxy_d_over_r: np.ndarray,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

    axes[0].imshow(mpimg.imread(lit_image_path))
    axes[0].axis("off")
    axes[0].set_title("Pitois 2000 Fig. 8", pad=10)

    valid = np.isfinite(proxy_d_over_r)
    if np.any(valid):
        axes[1].plot(
            particle_speeds_um_s[valid],
            proxy_d_over_r[valid],
            "o-",
            color="#d95f02",
            linewidth=1.8,
            markersize=6,
        )
    axes[1].set_xlabel("Prescribed particle speed [um/s]")
    axes[1].set_ylabel(r"proxy distance $D / R$")
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.1, 15.0)
    axes[1].grid(alpha=0.28)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))
    axes[1].set_title("Force-decay / neck-growth proxy, not true rupture", pad=10)
    axes[1].text(
        0.03,
        0.96,
        (
            "Current Stage-2 solver does not yet\n"
            "perform topological bridge rupture.\n"
            "Plotted here: first separation distance where\n"
            "the force falls below 10% of its peak.\n"
            "If that never happens, fall back to the first\n"
            "distance where neck radius exceeds 20 um.\n"
            "Only the x-range matches Fig. 8."
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ref_images = _extract_pitois_images(PITOIS_PDF, REF_DIR)

    # Representative force-distance curves.
    sep_cfg = _pitois_base_config(
        initial_sep=8.0e-5,
        v_particle=-2.5e-6,
        n_steps=5000,
        record_every=50,
    )
    app_cfg = _pitois_base_config(
        initial_sep=4.0e-4,
        v_particle=5.0e-6,
        n_steps=5000,
        record_every=50,
    )

    sep_history = _run_cached("separation_v5um_s_rel", sep_cfg)
    app_history = _run_cached("approach_v10um_s_rel", app_cfg)

    sep_arr = _arrays(sep_history, radius=sep_cfg.particle_radius)
    app_arr = _arrays(app_history, radius=app_cfg.particle_radius)

    separation_png = _save_force_compare(
        lit_image_path=ref_images["fig5_separation"],
        curve_x=sep_arr["d_over_r"],
        curve_y=sep_arr["force_mn"],
        title="Current catenoid-start case: separation",
        xlim=(0.01, 0.20),
        ylim=(0.02, 2.0),
        summary_text=(
            "Plotted as |F_axial| to match Pitois Fig. 5\n"
            f"relative speed = 5 um/s\n"
            f"D/R range = {np.min(sep_arr['d_over_r']):.3f} - {np.max(sep_arr['d_over_r']):.3f}\n"
            f"|F| range = {np.min(np.abs(sep_arr['force_mn'])):.3f} - {np.max(np.abs(sep_arr['force_mn'])):.3f} mN"
        ),
        out_path=FIG_DIR / "pitois2000_separation_force_compare.png",
        xscale="log",
        yscale="log",
        ylabel="Axial force magnitude [mN]",
        use_abs_force=True,
    )

    approach_png = _save_force_compare(
        lit_image_path=ref_images["fig6_approach"],
        curve_x=app_arr["d_over_r"],
        curve_y=app_arr["force_mn"],
        title="Current catenoid-start case: approach",
        xlim=(0.0, 0.20),
        ylim=(-0.40, 0.40),
        summary_text=(
            f"relative speed = 10 um/s\n"
            f"D/R range = {np.min(app_arr['d_over_r']):.3f} - {np.max(app_arr['d_over_r']):.3f}\n"
            f"F range = {np.min(app_arr['force_mn']):.3f} - {np.max(app_arr['force_mn']):.3f} mN"
        ),
        out_path=FIG_DIR / "pitois2000_approach_force_compare.png",
        show_zero_line=True,
        clip_note=(
            "Pitois axis window kept fixed.\n"
            "Part of the current curve extends below -0.4 mN."
        ),
    )

    # Velocity sweep for the approach-force peak.
    velocity_cases = [
        (2.5, 2.5e-6),
        (5.0, 5.0e-6),
        (7.5, 7.5e-6),
    ]
    particle_speed_um_s = []
    relative_speed_um_s = []
    fmax_vals = []
    rupture_proxy = []

    for vel_um_s_particle, v_particle in velocity_cases:
        rel_speed = 2.0 * vel_um_s_particle
        cfg = _pitois_base_config(
            initial_sep=4.0e-4,
            v_particle=v_particle,
            n_steps=5000,
            record_every=50,
        )
        history = _run_cached(f"approach_rel_{rel_speed:.1f}um_s", cfg)
        arr = _arrays(history, radius=cfg.particle_radius)
        particle_speed_um_s.append(vel_um_s_particle)
        relative_speed_um_s.append(rel_speed)
        fmax_vals.append(float(np.max(arr["force_mn"])))

        sep_cfg_v = _pitois_base_config(
            initial_sep=8.0e-5,
            v_particle=-v_particle,
            n_steps=5000,
            record_every=50,
        )
        sep_hist_v = _run_cached(f"separation_rel_{rel_speed:.1f}um_s", sep_cfg_v)
        rupture_proxy.append(_rupture_proxy_distance(sep_hist_v, radius=sep_cfg_v.particle_radius))

    particle_speed_um_s = np.array(particle_speed_um_s, dtype=float)
    relative_speed_um_s = np.array(relative_speed_um_s, dtype=float)
    fmax_vals = np.array(fmax_vals, dtype=float)
    rupture_proxy = np.array([np.nan if value is None else value for value in rupture_proxy], dtype=float)

    fmax_png = _save_fmax_compare(
        lit_image_path=ref_images["fig7_fmax"],
        particle_speeds_um_s=particle_speed_um_s,
        fmax_mn=fmax_vals,
        out_path=FIG_DIR / "pitois2000_fmax_compare.png",
    )

    rupture_png = _save_rupture_proxy_compare(
        lit_image_path=ref_images["fig8_rupture"],
        particle_speeds_um_s=particle_speed_um_s,
        proxy_d_over_r=rupture_proxy,
        out_path=FIG_DIR / "pitois2000_rupture_proxy_compare.png",
    )

    summary = {
        "separation_curve": {
            "d_over_r_min": float(np.min(sep_arr["d_over_r"])),
            "d_over_r_max": float(np.max(sep_arr["d_over_r"])),
            "force_mn_min": float(np.min(sep_arr["force_mn"])),
            "force_mn_max": float(np.max(sep_arr["force_mn"])),
        },
        "approach_curve": {
            "d_over_r_min": float(np.min(app_arr["d_over_r"])),
            "d_over_r_max": float(np.max(app_arr["d_over_r"])),
            "force_mn_min": float(np.min(app_arr["force_mn"])),
            "force_mn_max": float(np.max(app_arr["force_mn"])),
        },
        "fmax_vs_velocity": {
            "particle_speed_um_s": particle_speed_um_s.tolist(),
            "relative_speed_um_s": relative_speed_um_s.tolist(),
            "fmax_mn": fmax_vals.tolist(),
        },
        "rupture_proxy_vs_velocity": {
            "particle_speed_um_s": particle_speed_um_s.tolist(),
            "relative_speed_um_s": relative_speed_um_s.tolist(),
            "proxy_d_over_r": rupture_proxy.tolist(),
        },
    }
    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results" / "pitois2000_summary.json").write_text(json.dumps(summary, indent=2))

    print("Wrote comparison PNGs:")
    for path in [separation_png, approach_png, fmax_png, rupture_png]:
        print(path)


if __name__ == "__main__":
    main()
