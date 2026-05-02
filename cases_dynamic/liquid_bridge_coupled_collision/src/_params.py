"""Shared configuration for the Stage-2 coupled liquid-bridge cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class CoupledCaseConfig:
    """Configuration bundle for a coupled DEM-fluid bridge case."""

    name: str
    title: str
    initial_sep: float
    v_approach: float
    n_steps: int
    record_every: int
    mesh_snapshot_every: int
    bridge_connection_distance: float
    initial_geometry: str = "disconnected_caps"
    equilibrium_catenoid_refinement: int = 2
    enable_volume_projection: bool = True
    prescribed_particle_motion: bool = False
    particle_radius: float = 1.0e-3
    rho_s: float = 2500.0
    gamma: float = 0.072
    mu_f: float = 8.9e-4
    rho_f: float = 1000.0
    film_thickness: float = 1.0e-5
    cap_refinement: int = 2
    cap_phi_max: float = 0.92 * math.pi
    anchor_gap_alignment: float = -0.25
    dt: float = 2.0e-6
    fluid_substeps: int = 2
    film_damping: float = 5.0e-4
    max_surface_acceleration: float = 5.0e4
    contact_E: float = 5.0e8
    contact_nu: float = 0.22
    contact_gamma_n: float = 5.0e-2

    @property
    def dt_fluid(self) -> float:
        return self.dt / max(1, self.fluid_substeps)

    @property
    def total_time(self) -> float:
        return self.dt * self.n_steps


BASE_CONFIG = CoupledCaseConfig(
    name="base",
    title="Base coupled liquid-bridge collision case",
    initial_sep=2.2e-4,
    v_approach=1.2e-2,
    n_steps=2500,
    record_every=50,
    mesh_snapshot_every=100,
    bridge_connection_distance=3.35e-4,
)


def smoke_config() -> CoupledCaseConfig:
    """Non-bridging smoke test with visually noticeable approach motion."""

    return replace(
        BASE_CONFIG,
        name="Case_1_smoke",
        title="Case 1: Coupled moving-particle smoke test",
        initial_sep=6.0e-4,
        v_approach=2.0e-2,
        n_steps=2500,
        record_every=50,
        bridge_connection_distance=3.35e-4,
    )


def bridge_formation_config() -> CoupledCaseConfig:
    """Bridge-formation proof of concept with moving particles."""

    return replace(
        BASE_CONFIG,
        name="Case_2_bridge_formation",
        title="Case 2: Coupled dynamic bridge formation",
        initial_sep=2.2e-4,
        v_approach=1.2e-2,
        n_steps=2500,
        record_every=50,
        bridge_connection_distance=2.0e-4,
    )


def collision_config() -> CoupledCaseConfig:
    """Agglomerating collision with the coupled fluid bridge."""

    return replace(
        BASE_CONFIG,
        name="Case_3_collision_agglomeration",
        title="Case 3: Coupled collision and agglomeration",
        initial_sep=1.0e-4,
        v_approach=2.2e-2,
        n_steps=3500,
        record_every=50,
        bridge_connection_distance=1.3e-4,
        film_damping=7.5e-4,
    )


def catenoid_initialized_case2_config() -> CoupledCaseConfig:
    """Pre-bridged Case-2 variant seeded from the equilibrium catenoid mesh."""

    return replace(
        BASE_CONFIG,
        name="Case_2a_catenoid_initialized",
        title="Case 2a: Catenoid-initialized coupled bridge",
        initial_sep=2.2e-4,
        v_approach=1.2e-2,
        n_steps=2500,
        record_every=50,
        bridge_connection_distance=0.0,
        initial_geometry="equilibrium_catenoid",
        equilibrium_catenoid_refinement=2,
        enable_volume_projection=False,
    )
