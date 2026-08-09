"""
validate_against_rover_path.py

Real ground-truth validation for Step 5: checks whether our slope-based
terrain classifier (process_nasa_terrain.py) marks the terrain
Perseverance ACTUALLY drove across as passable.

This is the missing piece flagged during Step 5: process_nasa_terrain.py
derives "passable/impassable" purely from slope thresholds we chose
(3 / 8 / 15 degrees), with no confirmation those thresholds are
realistic. Perseverance's real, published driven path is ground truth:
if our classifier marks real-driven terrain as impassable, our
thresholds are miscalibrated -- this script measures exactly that.

Data needed (download first):
  1. The same DEM used in process_nasa_terrain.py:
     https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/
     ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif
  2. Perseverance's real driven path, from the Mars 2020 Rover PLACES
     Bundle (NASA/PDS Geosciences Node):
     https://pds-geosciences.wustl.edu/missions/mars2020/places.htm
     -> Data Localizations collection -> best_interp.csv
     Save both into data/raw/.

Usage:
    python -m mars_rl_project.data_processing.validate_against_rover_path
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from mars_rl_project.data_processing.process_nasa_terrain import (
    load_elevation, compute_slope_degrees, classify_terrain,
    ROCK_THRESHOLD, BOULDER_THRESHOLD, CRATER_THRESHOLD,
)
from mars_rl_project.environments.terrain_generator import SAFE, ROCK, BOULDER, CRATER, IMPASSABLE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DEM = PROJECT_ROOT / "data" / "raw" / "M20_JezeroCrater_CTXDEM_20m.tif"
DEFAULT_ROVER_CSV = PROJECT_ROOT / "data" / "raw" / "best_interp.csv"

NAMES = {SAFE: "safe", ROCK: "rock", BOULDER: "boulder", CRATER: "crater"}


def load_rover_path(csv_path: Path, lat_col_override: str | None = None,
                     lon_col_override: str | None = None) -> pd.DataFrame:
    """
    Loads the PLACES best_interp.csv. Column names in NASA's PLACES bundle
    can vary slightly by release -- this looks for the most likely
    lat/lon column names and fails loudly with the actual columns found
    if none match, rather than silently guessing wrong.
    """
    df = pd.read_csv(csv_path)

    print(f"\nAll numeric columns in {csv_path.name} with 'lat'/'lon'/'north'/'east' in the name, "
          f"and their value range (use this to spot the right one if auto-detection picks wrong):")
    keyword_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["lat", "lon", "north", "east", "x", "y"]
    )]
    for c in keyword_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            print(f"  {c:<35} min={df[c].min():.5f}  max={df[c].max():.5f}  range={df[c].max()-df[c].min():.5f}")

    if lat_col_override and lon_col_override:
        print(f"\nUsing manually specified columns: lat='{lat_col_override}', lon='{lon_col_override}'")
        return df.rename(columns={lat_col_override: "lat", lon_col_override: "lon"})

    lat_candidates = [
        "planetodetic_latitude", "PlanetodeticLatitude",
        "planetocentric_latitude", "PlanetocentricLatitude",
        "latitude", "lat",
    ]
    lon_candidates = [
        "planetodetic_longitude", "PlanetodeticLongitude",
        "planetocentric_longitude", "PlanetocentricLongitude",
        "longitude", "lon",
    ]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not find latitude/longitude columns automatically.\n"
            f"Columns found in {csv_path.name}: {list(df.columns)}\n"
            f"Look at the column list printed above (with ranges) and re-run with "
            f"--lat-col <name> --lon-col <name> pointing at whichever columns actually vary."
        )

    print(f"\nAuto-selected columns: lat='{lat_col}' (range={df[lat_col].max()-df[lat_col].min():.5f}), "
          f"lon='{lon_col}' (range={df[lon_col].max()-df[lon_col].min():.5f})")
    print("If this is the wrong column, re-run with --lat-col/--lon-col using one of the "
          "names listed above instead.")

    lat_range = df[lat_col].max() - df[lat_col].min()
    if lat_range < 1e-6:
        print(
            f"\nWARNING: auto-selected latitude column '{lat_col}' is constant (range={lat_range:.8f}). "
            f"This is almost certainly the WRONG column -- look at the list above for a column "
            f"that actually varies, then re-run with --lat-col <correct name>."
        )

    return df.rename(columns={lat_col: "lat", lon_col: "lon"})


def latlon_to_pixel(lat: np.ndarray, lon: np.ndarray, dem_path: Path, debug: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts rover lat/lon (Mars areographic coordinates) into DEM pixel
    (row, col) indices, using the DEM's own CRS and transform so this
    works correctly regardless of which specific Mars projection the
    DEM happens to use.
    """
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        transform = src.transform
        dem_shape = (src.height, src.width)

    if debug:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC 0: Raw coordinate values before pixel conversion")
        print("=" * 60)
        print(f"Raw lat: min={np.nanmin(lat):.5f} max={np.nanmax(lat):.5f} (range={np.nanmax(lat)-np.nanmin(lat):.5f} deg)")
        print(f"Raw lon: min={np.nanmin(lon):.5f} max={np.nanmax(lon):.5f} (range={np.nanmax(lon)-np.nanmin(lon):.5f} deg)")
        print(f"DEM CRS: {dem_crs}")
        print(f"DEM transform: a={transform.a} b={transform.b} c={transform.c} "
              f"d={transform.d} e={transform.e} f={transform.f}")

    source_crs = "+proj=longlat +a=3396190 +b=3396190 +no_defs"  # Mars sphere, IAU2000
    transformer = Transformer.from_crs(source_crs, dem_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    x, y = np.asarray(x), np.asarray(y)

    if debug:
        print(f"Projected x: min={np.nanmin(x):.1f} max={np.nanmax(x):.1f} (range={np.nanmax(x)-np.nanmin(x):.1f} m)")
        print(f"Projected y: min={np.nanmin(y):.1f} max={np.nanmax(y):.1f} (range={np.nanmax(y)-np.nanmin(y):.1f} m)")
        if (np.nanmax(y) - np.nanmin(y)) < 1.0:
            print(
                "  --> y range collapsed to ~0 HERE, before the pixel conversion step. "
                "This means the bug is in the lat->y projection itself (CRS mismatch, "
                "wrong source_crs assumption, or lat values genuinely not varying in "
                "the input CSV), NOT in the row/col pixel math below."
            )
        else:
            print("  --> y range looks fine here -- if rows still collapse below, the bug "
                  "is in the affine inversion step, not the CRS projection.")

    cols, rows = ~transform * (x, y)
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)

    in_bounds = (rows >= 0) & (rows < dem_shape[0]) & (cols >= 0) & (cols < dem_shape[1])
    return rows, cols, in_bounds


def projected_xy_to_pixel(x: np.ndarray, y: np.ndarray, dem_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts rover coordinates that are ALREADY in the DEM's own projected
    CRS (e.g. easting/northing in meters, as found directly in NASA's PLACES
    CSV) into pixel (row, col) indices. No pyproj/CRS transform needed --
    this is the preferred, more robust path whenever those columns are
    available, since it avoids any lat/lon datum or projection mismatch risk.
    """
    with rasterio.open(dem_path) as src:
        transform = src.transform
        dem_shape = (src.height, src.width)

    print("\n" + "=" * 60)
    print("Using pre-projected easting/northing columns directly (bypassing lat/lon + pyproj)")
    print("=" * 60)
    print(f"Raw easting (x):  min={np.nanmin(x):.1f} max={np.nanmax(x):.1f} (range={np.nanmax(x)-np.nanmin(x):.1f} m)")
    print(f"Raw northing (y): min={np.nanmin(y):.1f} max={np.nanmax(y):.1f} (range={np.nanmax(y)-np.nanmin(y):.1f} m)")

    cols, rows = ~transform * (x, y)
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)

    in_bounds = (rows >= 0) & (rows < dem_shape[0]) & (cols >= 0) & (cols < dem_shape[1])
    return rows, cols, in_bounds


def diagnose_projection(rows: np.ndarray, cols: np.ndarray, in_bounds: np.ndarray,
                          dem_shape: tuple[int, int], pixel_size_m: float) -> None:
    """
    Sanity-checks that the lat/lon -> pixel reprojection landed somewhere
    sensible, BEFORE trusting any slope statistics computed from it.
    A silently-wrong reprojection (e.g. swapped lat/lon, wrong Mars radius,
    wrong CRS assumption) would still produce numbers -- just meaningless
    ones -- so this is checked explicitly rather than assumed correct.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: Is the rover path correctly projected onto the DEM?")
    print("=" * 60)

    valid_rows = rows[in_bounds]
    valid_cols = cols[in_bounds]
    n_total, n_valid = len(rows), len(valid_rows)
    print(f"In-bounds: {n_valid}/{n_total} ({100*n_valid/max(n_total,1):.1f}%)")

    if n_valid == 0:
        print("No points in bounds -- cannot run further projection diagnostics.")
        return

    # Real-world extent this driven path covers, in meters/km, derived purely
    # from pixel spread -- Perseverance's real traverse (as of mid-2026) is on
    # the order of several km across. If this comes out as only a few pixels
    # wide, or as basically the entire DEM, that's a strong signal something
    # is wrong with the reprojection rather than "yes it works."
    row_span_px = valid_rows.max() - valid_rows.min()
    col_span_px = valid_cols.max() - valid_cols.min()
    row_span_km = row_span_px * pixel_size_m / 1000
    col_span_km = col_span_px * pixel_size_m / 1000
    print(f"Rover path pixel spread: rows {valid_rows.min()}-{valid_rows.max()} "
          f"({row_span_km:.2f} km), cols {valid_cols.min()}-{valid_cols.max()} ({col_span_km:.2f} km)")
    print(f"DEM full extent: {dem_shape[0]} x {dem_shape[1]} px "
          f"({dem_shape[0]*pixel_size_m/1000:.1f} x {dem_shape[1]*pixel_size_m/1000:.1f} km)")

    if row_span_km > 0.5 * dem_shape[0] * pixel_size_m / 1000:
        print(
            "  WARNING: rover path spans a large fraction of the entire DEM's height. "
            "Perseverance's real driven path is a few km across, not tens of km -- this "
            "could indicate a coordinate/projection mismatch rather than a genuine result."
        )
    elif row_span_km < 0.05:
        print(
            "  WARNING: rover path is suspiciously tiny (near a single pixel). This "
            "could mean the reprojection is collapsing all points to one location -- "
            "check the source CRS / lat-lon column assumptions."
        )
    else:
        print("  Spread looks plausible for a real rover traverse (few km scale). Looks OK.")

    # Duplicate/clustered-pixel check -- real GPS traces shouldn't have an
    # enormous fraction of points landing on the exact same pixel unless the
    # rover was stationary for a long time (possible, but worth flagging).
    coords = list(zip(valid_rows.tolist(), valid_cols.tolist()))
    n_unique = len(set(coords))
    dup_pct = 100 * (1 - n_unique / len(coords))
    print(f"Unique pixel locations: {n_unique}/{len(coords)} ({dup_pct:.1f}% duplicate/overlapping)")
    if dup_pct > 80:
        print(
            "  WARNING: over 80% of rover positions collapse onto the same handful of "
            "pixels. At 20 m/pixel this can legitimately happen (many GPS fixes per "
            "stationary science stop), but also happens if reprojection is degenerate -- "
            "worth a manual spot-check of a few (lat, lon) -> (row, col) conversions."
        )
    else:
        print("  Looks OK -- positions are spread across many distinct pixels.")


def compare_slope_distributions(slope: np.ndarray, slope_at_rover: np.ndarray) -> None:
    """
    Diagnostic 2+3: descriptive stats of slope along the real rover path,
    compared against the slope distribution of the WHOLE DEM. If the rover
    only ever drove through the DEM's gentlest terrain, that's expected and
    supports "thresholds too strict." If rover-path slope looks similar to
    (or rougher than) the full-DEM distribution, high mismatch % is more
    likely a real signature of legitimately rough terrain, and/or a
    projection issue, rather than purely a threshold problem.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2+3: Rover-path slope vs. whole-DEM slope distribution")
    print("=" * 60)

    dem_slope_flat = slope[~np.isnan(slope)]
    rover_slope = slope_at_rover[~np.isnan(slope_at_rover)]

    def stats(arr: np.ndarray) -> dict:
        p = np.percentile(arr, [50, 90, 95, 98, 99])
        return {
            "mean": float(np.mean(arr)), "median": p[0], "p90": p[1],
            "p95": p[2], "p98": p[3], "p99": p[4], "max": float(np.max(arr)),
        }

    dem_stats = stats(dem_slope_flat)
    rover_stats = stats(rover_slope)

    print(f"{'Metric':<12}{'Whole DEM':>12}{'Rover path':>14}")
    for key in ["mean", "median", "p90", "p95", "p98", "p99", "max"]:
        print(f"{key:<12}{dem_stats[key]:>12.1f}{rover_stats[key]:>14.1f}")

    if rover_stats["median"] < dem_stats["median"]:
        print(
            "\nRover-driven terrain is GENTLER (lower median slope) than the DEM overall -- "
            "consistent with the rover actively avoiding rough terrain, as expected. This "
            "supports the idea that a high impassable-mismatch % reflects overly strict "
            "thresholds rather than a projection error, since the rover-path slope "
            "distribution behaves sensibly relative to the full map."
        )
    else:
        print(
            "\nRover-driven terrain is NOT gentler than the DEM overall -- unexpected, since "
            "a real rover should avoid the roughest terrain. This is worth investigating "
            "before trusting recalibrated thresholds: check the projection diagnostic above, "
            "and consider spot-checking a few known rover waypoints (e.g. named landmarks "
            "with published coordinates) against this DEM manually."
        )


def main():
    parser = argparse.ArgumentParser(description="Validate terrain classifier against Perseverance's real path")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--rover-csv", type=Path, default=DEFAULT_ROVER_CSV)
    parser.add_argument("--lat-col", default=None, help="Manually specify the latitude column name if auto-detection picks the wrong one")
    parser.add_argument("--lon-col", default=None, help="Manually specify the longitude column name if auto-detection picks the wrong one")
    parser.add_argument("--easting-col", default="easting",
                         help="Column with rover position already in the DEM's projected CRS (meters). "
                              "Preferred over lat/lon when present, since it avoids any datum/CRS mismatch risk. "
                              "Set to '' to force lat/lon + pyproj instead.")
    parser.add_argument("--northing-col", default="northing")
    args = parser.parse_args()

    for path, label in [(args.dem, "DEM"), (args.rover_csv, "rover path CSV")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at {path}. See this script's docstring for download links.")

    print(f"Loading DEM: {args.dem}")
    elevation, pixel_size_m = load_elevation(args.dem)
    print(f"DEM shape: {elevation.shape}, pixel size: {pixel_size_m:.1f} m")

    print(f"Computing slope + classifying full DEM "
          f"(thresholds: rock>={ROCK_THRESHOLD} deg, boulder>={BOULDER_THRESHOLD} deg, "
          f"crater>={CRATER_THRESHOLD} deg)...")
    slope = compute_slope_degrees(elevation, pixel_size_m)
    terrain = classify_terrain(slope)

    print(f"Loading rover path: {args.rover_csv}")
    raw_df = pd.read_csv(args.rover_csv)
    print(f"Loaded {len(raw_df)} rover position records")

    use_projected = (
        args.easting_col and args.northing_col
        and args.easting_col in raw_df.columns and args.northing_col in raw_df.columns
    )

    if use_projected:
        rover_df = raw_df
        rows, cols, in_bounds = projected_xy_to_pixel(
            rover_df[args.easting_col].values, rover_df[args.northing_col].values, args.dem
        )
    else:
        rover_df = load_rover_path(args.rover_csv, args.lat_col, args.lon_col)
        rows, cols, in_bounds = latlon_to_pixel(rover_df["lat"].values, rover_df["lon"].values, args.dem)

    n_in_bounds = int(in_bounds.sum())
    print(f"{n_in_bounds}/{len(rover_df)} rover positions fall within this DEM's coverage")

    if n_in_bounds == 0:
        print(
            "\nNo rover positions overlap this DEM. This can happen if the CSV's "
            "coordinate columns weren't what we expected, or the rover's real "
            "path genuinely doesn't overlap this specific DEM tile. Double-check "
            "the lat/lon column names and DEM coverage before trusting the "
            "'no overlap' result."
        )
        return

    valid_rows = rows[in_bounds]
    valid_cols = cols[in_bounds]
    terrain_at_rover = terrain[valid_rows, valid_cols]

    diagnose_projection(rows, cols, in_bounds, elevation.shape, pixel_size_m)

    slope_at_rover = slope[valid_rows, valid_cols]
    compare_slope_distributions(slope, slope_at_rover)

    unique, counts = np.unique(terrain_at_rover, return_counts=True)
    total = len(terrain_at_rover)
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: Terrain classification at rover-driven locations")
    print("=" * 60)
    for code, count in zip(unique, counts):
        print(f"  {NAMES.get(int(code), code)}: {count} ({100*count/total:.1f}%)")

    impassable_count = int(np.isin(terrain_at_rover, list(IMPASSABLE)).sum())
    impassable_pct = 100 * impassable_count / total
    print(f"\n{'='*60}")
    print(f"Real-driven terrain classified as IMPASSABLE by our thresholds: {impassable_pct:.1f}%")
    print(f"{'='*60}")

    if impassable_pct < 2:
        print(
            "This is low -- our slope thresholds look well-calibrated: the classifier "
            "rarely marks terrain the real rover actually crossed as impassable."
        )
    elif impassable_pct < 10:
        print(
            "Some mismatch. Plausible causes: DEM/rover coordinate misalignment (a few "
            "pixels off), pixel-averaging smoothing out small safe gaps the real rover "
            "found, or thresholds slightly too strict. Worth a sentence in your "
            "limitations section either way."
        )
    else:
        print(
            "Meaningful mismatch -- your thresholds are likely too strict relative to "
            "what Perseverance's real engineering team considered traversable."
        )
        p50, p90, p95, p98, p99 = np.percentile(slope_at_rover[~np.isnan(slope_at_rover)], [50, 90, 95, 98, 99])
        print(
            f"\nData-driven recalibration suggestion, based on the ACTUAL slope "
            f"distribution at every point Perseverance really drove across:\n"
            f"  median slope driven : {p50:.1f} deg\n"
            f"  90th percentile     : {p90:.1f} deg\n"
            f"  95th percentile     : {p95:.1f} deg\n"
            f"  98th percentile     : {p98:.1f} deg\n"
            f"  99th percentile     : {p99:.1f} deg\n\n"
            f"Suggested new thresholds in process_nasa_terrain.py (rounded):\n"
            f"  ROCK_THRESHOLD = {p50:.1f}      # roughly the median real-driven slope\n"
            f"  BOULDER_THRESHOLD = {p95:.1f}   # only the roughest 5% the rover actually crossed\n"
            f"  CRATER_THRESHOLD = {p99:.1f}    # only the roughest 1% -- near the rover's real limit\n\n"
            f"These make 'impassable' mean 'steeper than the rover has almost ever actually "
            f"driven', anchored to real mission data instead of a guess. Update the three "
            f"threshold constants near the top of process_nasa_terrain.py to these values, "
            f"then re-run both process_nasa_terrain.py and this validation script to confirm "
            f"the impassable-mismatch percentage drops close to ~1-2%."
        )


if __name__ == "__main__":
    main()
