# Coupled Liquid-Bridge Collision Cases

This folder contains the Stage-2 coupled DEM-fluid proof-of-concept cases.

## Cases

1. `Case_1_coupled_moving_particles_smoke.py`
   - short smoke test for the coupled time-step loop
   - verifies synchronized DEM + fluid stepping without bridge formation

2. `Case_2_coupled_bridge_formation.py`
   - two wetted particles approach each other
   - rigid wetting caps stitch into a bridge when the rim gap is small enough
   - after stitching, free bridge vertices evolve with `symplectic_euler(...)`

3. `Case_2a_catenoid_initialized_bridge.py`
   - starts from the equilibrium benchmark catenoid mesh instead of disconnected caps
   - useful for pre-bridged dynamic studies and Pitois-style approach/separation cases
   - keeps the same coupled DEM + fluid time-step loop, but the bridge exists from `t = 0`

4. `Case_2b_pitois2000_separation.py`
   - reuses the volumetric catenoid-like bridge mesh from `liquid_bridge_equilibrium/Case_5`
   - moves only the top and bottom cap boundaries with prescribed velocities
   - evaluates axial bridge force through built-in `multiphase_stress.py` on the volume
   - writes the Fig. 5 comparison PNGs itself, so there is only one `Case_2b` script and one `out/Case_2b` result folder
   - this is the stricter pre-bridged moving-boundary path for Pitois-style force-distance studies

5. `Case_3_coupled_collision_agglomeration.py`
   - moving particles form a bridge, collide, and settle qualitatively
   - capillary and Stokes-integral drag coupling are applied inside the same macro step as DEM

## Solver layout

Each macro time step performs:

1. sync the particle-attached wetting surface with the DEM particles
2. stitch a bridge band when the facing rim loops are close enough
3. advance the free fluid surface with `ddgclib.dynamic_integrators.symplectic_euler(...)`
4. extract capillary and Stokes-integral drag forces from the interface
5. advance DEM with `ddgclib.dem.dem_step(...)`
6. resync the anchored fluid vertices to the updated particle positions

The bridge force is therefore not taken from `LiquidBridgeManager`; it comes from the evolving interface mesh itself.

## Output

Each case writes to its own output folder under `out/Case_1`, `out/Case_2`, `out/Case_2a`, `out/Case_2b`, or `out/Case_3`:

- `results/history.json`
- `fig/separation.png`
- `fig/forces.png`
- `fig/bridge_metrics.png`
- `fig/summary.png`
- `fig/mesh_initial.png`
- `fig/mesh_final.png`
- `fig/mesh_bridge.png` for cases where a bridge forms during the run

`Case_2b` also writes:
- `fig/approach/mesh_iter****.png`
- `fig/separation/mesh_iter****.png`
- `fig/pitois2000_volumetric_separation_digitized_compare.png`

## Practical note

This is a qualitative Stage-2 deliverable. The bridge is evolved with the dynamic integrator and coupled to DEM in one loop, but the validation target is stability and plausible bridge formation/agglomeration rather than an analytical force benchmark.

`Case_2b_pitois2000_separation.py` is the exception to the surface-only Stage-2 solver path: it is a volumetric moving-boundary case built on the Case-5 mesh family. It is physically closer to the Pitois experiments because the force is extracted through the volumetric stress operator, but it still remains an intermediate validation step rather than a final matched experiment.
