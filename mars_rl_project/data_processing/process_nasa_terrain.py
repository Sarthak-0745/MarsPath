"""
process_nasa_terrain.py

Step 5 of the MarsPath timeline: convert a real NASA Digital Elevation
Model (DEM) of Jezero Crater into a terrain grid compatible with
MarsTerrainEnv (Section 2.3), for validating the trained agents on
real Mars topography instead of synthetic terrain.

Data source (public domain, USGS Astrogeology):
    https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/
    ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif
Download that file into data/raw/ before running this script.

How elevation becomes terrain codes
------------------------------------
Real Mars DEMs don't come pre-labeled as "safe ground / rock / boulder /
crater" -- that labeling is specific to your synthetic environment. To
validate on real data, we need a defensible, documented rule for
turning continuous elevation into your four discrete categories. This
script uses SLOPE (rate of elevation change between neighboring cells)
as the hazard signal, which mirrors how real rover autonomy systems
(e.g. NASA's AutoNav) assess traversability -- steep local slope is a
genuine proxy for boulders, crater walls, and other obstacles a rover
physically cannot cross, even though the DEM itself doesn't label
individual rocks.

Classification rule (documented assumption -- state this in your
methods/limitations section):
    slope < ROCK_THRESHOLD               -> SAFE
    ROCK_THRESHOLD <= slope < BOULDER_TH  -> ROCK       (passable, rough)
    BOULDER_TH <= slope < CRATER_TH       -> BOULDER    (impassable)
    slope >= CRATER_TH                    -> CRATER     (impassable)
Thresholds are in degrees and tuned so the resulting terrain-type
proportions are roughly comparable to your "hard" synthetic preset --
adjust ROCK_THRESHOLD/BOULDER_THRESHOLD/CRATER_THRESHOLD below and
re-run if you want a different real-terrain difficulty balance.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import zoom

from mars_rl_project.environments.terrain_generator import (
    SAFE, ROCK, BOULDER, CRATER, IMPASSABLE
)
from mars_rl_project.environments.terrain_generator import _bfs_reachable  # reuse solvability check

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "M20_JezeroCrater_CTXDEM_20m.tif"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Slope thresholds in degrees -- see module docstring.
ROCK_THRESHOLD = 3.0
BOULDER_THRESHOLD = 8.0
CRATER_THRESHOLD = 15.0


def load_elevation(path: Path) -> tuple[np.ndarray, float]:
    """Returns (elevation array, pixel size in meters)."""
    with rasterio.open(path) as src:
        elevation = src.read(1).astype(np.float32)
        pixel_size_m = abs(src.transform.a)  # size of one pixel in map units (meters for this CRS)
        nodata = src.nodata
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)
    return elevation, pixel_size_m


def crop_region(elevation: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    """Crop a size x size window starting at (row, col). Raises if it runs off the array."""
    if row + size > elevation.shape[0] or col + size > elevation.shape[1]:
        raise ValueError(
            f"Requested crop [{row}:{row+size}, {col}:{col+size}] exceeds "
            f"DEM shape {elevation.shape}. Pick a smaller size or different offset."
        )
    return elevation[row:row + size, col:col + size]


def compute_slope_degrees(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """
    Slope magnitude in degrees at each cell, from the elevation gradient.
    Uses np.gradient (central differences) -- standard, simple slope estimate;
    good enough for a coarse traversability classification like this one.
    """
    dz_dy, dz_dx = np.gradient(elevation, pixel_size_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    return np.degrees(slope_rad)


def classify_terrain(slope_deg: np.ndarray) -> np.ndarray:
    grid = np.full(slope_deg.shape, SAFE, dtype=np.int8)
    grid[slope_deg >= ROCK_THRESHOLD] = ROCK
    grid[slope_deg >= BOULDER_THRESHOLD] = BOULDER
    grid[slope_deg >= CRATER_THRESHOLD] = CRATER
    # NaN (nodata) cells -- treat as impassable rather than guessing.
    grid[np.isnan(slope_deg)] = BOULDER
    return grid


def resample_to_grid(terrain: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downsample the cropped terrain grid to target_size x target_size
    (matching MarsTerrainEnv's grid_size) using nearest-neighbor zoom,
    so terrain codes stay integers (0-3), never interpolated fractions.
    """
    zoom_factor = target_size / terrain.shape[0]
    resized = zoom(terrain.astype(np.float32), zoom_factor, order=0)  # order=0 = nearest neighbor
    resized = np.clip(np.round(resized), 0, 3).astype(np.int8)
    if resized.shape != (target_size, target_size):
        # zoom can be off by a pixel due to rounding -- pad/crop to exact size
        out = np.full((target_size, target_size), SAFE, dtype=np.int8)
        r = min(target_size, resized.shape[0])
        c = min(target_size, resized.shape[1])
        out[:r, :c] = resized[:r, :c]
        resized = out
    return resized


def find_solvable_start_goal(
    grid: np.ndarray, rng: np.random.Generator, min_distance: int = 20, max_attempts: int = 500
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Same solvability contract as synthetic terrain_generator.py: BFS-verified reachable pair."""
    size = grid.shape[0]
    safe_cells = np.argwhere(grid != BOULDER)
    safe_cells = [tuple(c) for c in safe_cells if grid[tuple(c)] not in IMPASSABLE]
    if len(safe_cells) < 2:
        raise RuntimeError("Not enough passable terrain in this region to place start/goal.")

    for _ in range(max_attempts):
        start = tuple(safe_cells[rng.integers(len(safe_cells))])
        goal = tuple(safe_cells[rng.integers(len(safe_cells))])
        cheby = max(abs(start[0] - goal[0]), abs(start[1] - goal[1]))
        if cheby >= min_distance and _bfs_reachable(grid, start, goal):
            return start, goal
    raise RuntimeError(
        f"Could not find a solvable start/goal pair with min_distance={min_distance} "
        f"in this region after {max_attempts} attempts. Try a different crop or lower min_distance."
    )


def scan_for_rough_regions(
    elevation: np.ndarray, pixel_size_m: float, tile_size: int, top_k: int = 10
) -> list[tuple[int, int, float]]:
    """
    Tiles the DEM into non-overlapping tile_size x tile_size windows, computes
    mean slope per tile, and returns the top_k roughest tiles as (row, col, mean_slope_deg),
    sorted steepest-first. Used to locate crater rim / delta scarp regions instead of
    guessing pixel coordinates blindly.
    """
    results = _scan_all_tiles(elevation, pixel_size_m, tile_size)
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


def scan_stratified_regions(
    elevation: np.ndarray, pixel_size_m: float, tile_size: int, n_bins: int = 5
) -> list[tuple[int, int, float]]:
    """
    Same tiling as scan_for_rough_regions, but instead of returning only the
    roughest tiles, splits ALL tiles into n_bins groups by mean slope
    (flattest -> roughest) and returns one representative tile per bin.
    This gives a spread across the DEM's real difficulty range -- flat
    crater floor, gently rolling terrain, moderate slopes, rough patches,
    and the roughest areas -- rather than only the two extremes (which is
    what picking just the flattest default crop + the single roughest
    --scan result gives you). Used to build a real-terrain evaluation set
    that's representative of the crater as a whole, not cherry-picked.
    """
    results = _scan_all_tiles(elevation, pixel_size_m, tile_size)
    results.sort(key=lambda x: x[2])  # flattest first
    n = len(results)
    if n == 0:
        return []
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    picks = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if lo >= hi:
            continue
        mid_idx = (lo + hi) // 2  # representative (median) tile within this difficulty bin
        picks.append(results[mid_idx])
    return picks


def _scan_all_tiles(elevation: np.ndarray, pixel_size_m: float, tile_size: int) -> list[tuple[int, int, float]]:
    results = []
    rows, cols = elevation.shape
    for r in range(0, rows - tile_size, tile_size):
        for c in range(0, cols - tile_size, tile_size):
            tile = elevation[r:r + tile_size, c:c + tile_size]
            if np.all(np.isnan(tile)):
                continue
            slope = compute_slope_degrees(tile, pixel_size_m)
            mean_slope = float(np.nanmean(slope))
            results.append((r, c, mean_slope))
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert a NASA DEM into a MarsTerrainEnv-compatible grid")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help=f"Path to the downloaded DEM .tif (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crop-row", type=int, default=None,
                         help="Row offset for crop window (default: center of DEM)")
    parser.add_argument("--crop-col", type=int, default=None,
                         help="Column offset for crop window (default: center of DEM)")
    parser.add_argument("--crop-size", type=int, default=250,
                         help="Size of the square region to crop from the DEM before resampling (in DEM pixels)")
    parser.add_argument("--grid-size", type=int, default=50,
                         help="Output grid size, must match MarsTerrainEnv's grid_size (default: 50)")
    parser.add_argument("--n-maps", type=int, default=5,
                         help="How many different solvable start/goal maps to generate from this one crop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan", action="store_true",
                         help="Instead of processing, scan the whole DEM and print the roughest "
                              "(steepest) tile locations -- use these as --crop-row/--crop-col "
                              "to target harder terrain (e.g. crater rim/delta scarp)")
    parser.add_argument("--scan-top-k", type=int, default=10)
    parser.add_argument("--scan-stratified", action="store_true",
                         help="Like --scan, but instead of the roughest tiles, prints one "
                              "representative tile from each of --stratified-bins difficulty "
                              "bins spanning the DEM's full slope range (flat -> rough). Use "
                              "this to build a representative real-terrain evaluation set "
                              "instead of only sampling extremes.")
    parser.add_argument("--stratified-bins", type=int, default=5)
    parser.add_argument("--all-regions", action="store_true",
                         help="Automatically process ALL --stratified-bins regions in one run "
                              "(combines --scan-stratified + processing each region), saving "
                              "--n-maps maps per region into --output-dir with unique filenames.")
    parser.add_argument("--name-prefix", default="jezero_map",
                         help="Prefix for saved map filenames (default: jezero_map). Useful when "
                              "combining maps from multiple regions/runs into one output-dir "
                              "without filename collisions.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"DEM not found at {args.input}\n"
            f"Download it from:\n"
            f"  https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/ScienceInvestigationMaps_JPL/"
            f"M20_JezeroCrater_CTXDEM_20m.tif\n"
            f"and save it to that path."
        )

    print(f"Loading DEM: {args.input}")
    elevation, pixel_size_m = load_elevation(args.input)
    print(f"DEM shape: {elevation.shape}, pixel size: {pixel_size_m:.1f} m")

    if args.scan:
        print(f"Scanning for the {args.scan_top_k} roughest {args.crop_size}x{args.crop_size} regions...")
        rough = scan_for_rough_regions(elevation, pixel_size_m, args.crop_size, args.scan_top_k)
        print("\nRank  Row    Col    Mean slope (deg)")
        for i, (r, c, mean_slope) in enumerate(rough):
            print(f"{i+1:>4}  {r:<6} {c:<6} {mean_slope:.2f}")
        print(
            f"\nRe-run with e.g.:\n"
            f"  python -m mars_rl_project.data_processing.process_nasa_terrain "
            f"--crop-row {rough[0][0]} --crop-col {rough[0][1]} --crop-size {args.crop_size} "
            f"--output-dir data/processed_hard"
        )
        return

    if args.scan_stratified:
        strat = scan_stratified_regions(elevation, pixel_size_m, args.crop_size, args.stratified_bins)
        print(f"\n{len(strat)} representative regions across the DEM's difficulty range "
              f"(flattest bin -> roughest bin):")
        print("Bin   Row    Col    Mean slope (deg)")
        for i, (r, c, mean_slope) in enumerate(strat):
            print(f"{i+1:>4}  {r:<6} {c:<6} {mean_slope:.2f}")
        print(
            f"\nRe-run with --all-regions to automatically process every one of these regions, "
            f"or process one manually with --crop-row/--crop-col."
        )
        return

    if args.all_regions:
        strat = scan_stratified_regions(elevation, pixel_size_m, args.crop_size, args.stratified_bins)
        print(f"Processing all {len(strat)} stratified regions, {args.n_maps} maps each...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        total_saved = 0
        for bin_i, (row, col, mean_slope) in enumerate(strat):
            print(f"\n--- Region {bin_i+1}/{len(strat)}: row={row} col={col} mean_slope={mean_slope:.1f} deg ---")
            cropped = crop_region(elevation, row, col, args.crop_size)
            slope = compute_slope_degrees(cropped, pixel_size_m)
            terrain = classify_terrain(slope)
            resampled = resample_to_grid(terrain, args.grid_size)

            unique, counts = np.unique(resampled, return_counts=True)
            names = {SAFE: "safe", ROCK: "rock", BOULDER: "boulder", CRATER: "crater"}
            total = resampled.size
            for code, count in zip(unique, counts):
                print(f"  {names.get(int(code), code)}: {100*count/total:.1f}%", end="  ")
            print()

            for i in range(args.n_maps):
                try:
                    start, goal = find_solvable_start_goal(resampled, rng)
                except RuntimeError as e:
                    print(f"    Map {i}: skipped ({e})")
                    continue
                out_path = args.output_dir / f"{args.name_prefix}_region{bin_i:02d}_{i:02d}.npz"
                np.savez(out_path, grid=resampled, start=start, goal=goal)
                total_saved += 1
        print(f"\nDone. Saved {total_saved} solvable real-terrain maps across "
              f"{len(strat)} regions to {args.output_dir}")
        return



    row = args.crop_row if args.crop_row is not None else elevation.shape[0] // 2 - args.crop_size // 2
    col = args.crop_col if args.crop_col is not None else elevation.shape[1] // 2 - args.crop_size // 2
    row, col = max(0, row), max(0, col)

    print(f"Cropping {args.crop_size}x{args.crop_size} region at (row={row}, col={col})")
    cropped = crop_region(elevation, row, col, args.crop_size)

    print("Computing slope...")
    slope = compute_slope_degrees(cropped, pixel_size_m)
    print(f"Slope stats (deg): min={np.nanmin(slope):.1f} max={np.nanmax(slope):.1f} "
          f"mean={np.nanmean(slope):.1f}")

    terrain = classify_terrain(slope)
    resampled = resample_to_grid(terrain, args.grid_size)

    unique, counts = np.unique(resampled, return_counts=True)
    names = {SAFE: "safe", ROCK: "rock", BOULDER: "boulder", CRATER: "crater"}
    total = resampled.size
    print("Terrain composition after resampling:")
    for code, count in zip(unique, counts):
        print(f"  {names.get(int(code), code)}: {count} ({100*count/total:.1f}%)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    saved = 0
    for i in range(args.n_maps):
        try:
            start, goal = find_solvable_start_goal(resampled, rng)
        except RuntimeError as e:
            print(f"  Map {i}: skipped ({e})")
            continue
        out_path = args.output_dir / f"{args.name_prefix}_{i:02d}.npz"
        np.savez(out_path, grid=resampled, start=start, goal=goal)
        print(f"  Saved {out_path.name}: start={start}, goal={goal}")
        saved += 1

    print(f"\nDone. Saved {saved}/{args.n_maps} solvable real-terrain maps to {args.output_dir}")
    if saved == 0:
        print(
            "No solvable maps were found in this crop region. Try a different --crop-row/--crop-col "
            "(e.g. somewhere flatter), a larger --crop-size, or loosen the slope thresholds at the "
            "top of this script."
        )


if __name__ == "__main__":
    main()
