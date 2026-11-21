"""
Helpers for building the AlphaEarth (AEF) cosine-medoid layer on a WildfireSpreadTS (WSTS) grid.

Two parts:
  1. NumPy reference implementation of the cosine medoid (used to validate the Earth Engine result).
  2. Earth Engine builders that aggregate the ~10 m AEF embeddings into the exact 375 m WSTS pixels.

Cosine medoid of a cell with unit vectors v_1..v_n:
    argmax_i  sum_j v_i . v_j  =  argmax_i  v_i . S,   S = sum_j v_j
so it can be computed in O(n) per cell instead of O(n^2).
"""
import numpy as np

AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
N_SUB = 38          # fine pixels per WSTS pixel side: 375 m / 38 ~= 9.87 m (close to AEF's 10 m)
MAX_PIXELS = 4096   # reduceResolution cap; must exceed N_SUB**2 = 1444
UNIT_NORM_ATOL = 1e-2


# ----------------------------------------------------------------------------------------
# NumPy reference
# ----------------------------------------------------------------------------------------
def _unit(V):
    V = np.asarray(V, dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    keep = (norms[:, 0] > 0) & np.isfinite(V).all(axis=1)
    return V[keep] / norms[keep]


def cosine_medoid_bruteforce(V):
    """O(n^2) definition. V: (n, d). Returns (index into the filtered vectors, unit vector)."""
    Vn = _unit(V)
    if not len(Vn):
        raise ValueError("Cannot compute a medoid without valid vectors.")
    i = int(np.argmax((Vn @ Vn.T).sum(axis=1)))
    return i, Vn[i]


def cosine_medoid_fast(V):
    """O(n) version using sum_j v_i.v_j = v_i . S. Returns (index, unit vector)."""
    Vn = _unit(V)
    if not len(Vn):
        raise ValueError("Cannot compute a medoid without valid vectors.")
    S = Vn.sum(axis=0)
    i = int(np.argmax(Vn @ S))
    return i, Vn[i]


def block_medoids(fine, n=N_SUB):
    """
    fine: (C, n*H, n*W) array, NaN where invalid. Returns (C, H, W) float32 medoids, NaN where a
    cell has no valid pixel. Each n x n block of fine pixels is one WSTS pixel.
    """
    fine = np.asarray(fine)
    if fine.ndim != 3:
        raise ValueError(f"Expected fine embeddings with shape (bands, height, width); got {fine.shape}")
    if n <= 0:
        raise ValueError(f"Block size must be positive; got {n}")
    c, hn, wn = fine.shape
    if hn % n or wn % n:
        raise ValueError(f"Fine grid shape {(hn, wn)} is not divisible by block size {n}")
    H, W = hn // n, wn // n
    out = np.full((c, H, W), np.nan, dtype=np.float32)
    for r in range(H):
        for q in range(W):
            blk = fine[:, r * n:(r + 1) * n, q * n:(q + 1) * n].reshape(c, -1).T
            valid = _unit(blk)
            if len(valid):
                S = valid.sum(axis=0)
                out[:, r, q] = valid[int(np.argmax(valid @ S))]
    return out


# ----------------------------------------------------------------------------------------
# Grid helpers (pure Python, no Earth Engine needed)
# ----------------------------------------------------------------------------------------
def grid_from_tif(path):
    """Read CRS / affine transform / size of a WSTS GeoTIFF."""
    import rasterio
    with rasterio.open(path) as ds:
        t = ds.transform
        return {"crs": ds.crs.to_string(), "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                "width": ds.width, "height": ds.height}


def validate_grid(grid):
    """Reject malformed or non-north-up grids before building an Earth Engine graph."""
    if not grid.get("crs"):
        raise ValueError("Grid CRS must be set")
    if grid["width"] <= 0 or grid["height"] <= 0:
        raise ValueError(f"Grid dimensions must be positive; got {grid['width']}x{grid['height']}")
    transform = np.asarray(grid["transform"], dtype=np.float64)
    if transform.shape != (6,) or not np.isfinite(transform).all():
        raise ValueError("Grid transform must contain six finite affine coefficients")
    a, b, _, d, e, _ = transform
    if a == 0 or e == 0 or a * e - b * d == 0:
        raise ValueError(f"Grid transform is degenerate: {transform.tolist()}")
    if b != 0 or d != 0:
        raise ValueError("Rotated/sheared grids are not supported (expected a north-up GeoTIFF).")


def fine_grid(grid, n=N_SUB):
    validate_grid(grid)
    if n <= 0:
        raise ValueError(f"Fine-grid subdivision must be positive; got {n}")
    a, b, c, d, e, f = grid["transform"]
    return {"crs": grid["crs"], "transform": [a / n, b / n, c, d / n, e / n, f],
            "width": grid["width"] * n, "height": grid["height"] * n}


def grid_extent(grid):
    validate_grid(grid)
    a, b, c, d, e, f = grid["transform"]
    xs = sorted([c, c + a * grid["width"]])
    ys = sorted([f, f + e * grid["height"]])
    return xs[0], ys[0], xs[1], ys[1]          # xmin, ymin, xmax, ymax in the grid's CRS


# ----------------------------------------------------------------------------------------
# Earth Engine builders
# ----------------------------------------------------------------------------------------
def region_geometry(ee, grid, inset_px=0.0):
    """Rectangle covering the grid. inset_px > 0 shrinks it by that many pixels per side
    (use 0.5 for exports so no neighbouring row/column is included); < 0 grows it."""
    a, _, _, _, e, _ = grid["transform"]
    xmin, ymin, xmax, ymax = grid_extent(grid)
    dx, dy = abs(a) * inset_px, abs(e) * inset_px
    return ee.Geometry.Rectangle([xmin + dx, ymin + dy, xmax - dx, ymax - dy],
                                 proj=grid["crs"], geodesic=False)


def build_fine_embedding_image(ee, grid, aef_year, n=N_SUB):
    """Build unit AEF vectors on a fine grid aligned exactly to the WSTS grid."""
    fg = fine_grid(grid, n)
    fine_proj = ee.Projection(fg["crs"], fg["transform"])
    tiles = (ee.ImageCollection(AEF_COLLECTION)
             .filterDate(f"{aef_year}-01-01", f"{aef_year + 1}-01-01")
             .filterBounds(region_geometry(ee, grid, inset_px=-2))
             .mosaic().select(BANDS))
    return tiles.reproject(fine_proj)


def build_medoid_image(ee, grid, aef_year, n=N_SUB):
    """
    Returns (fine, coarse):
      fine   : unit-length 64-band embedding on a cell-aligned ~10 m grid (for validation)
      coarse : 64-band cosine medoid on the exact WSTS grid (bands A00..A63)
    """
    coarse_proj = ee.Projection(grid["crs"], grid["transform"])
    fine = build_fine_embedding_image(ee, grid, aef_year, n)
    fine_proj = fine.projection()
    # S = sum of unit vectors in each WSTS pixel, copied back to every fine pixel of that cell
    S = fine.reduceResolution(reducer=ee.Reducer.sum(), bestEffort=False,
                              maxPixels=MAX_PIXELS).reproject(coarse_proj)
    S_fine = S.reproject(fine_proj)

    # medoid score of each fine pixel = v . S  (argmax of this is the exact cosine medoid)
    score = fine.multiply(S_fine).reduce(ee.Reducer.sum()).rename("score")

    # Select one winning fine pixel so every output vector remains a real embedding.
    max_score = (score.reduceResolution(reducer=ee.Reducer.max(), bestEffort=False,
                                        maxPixels=MAX_PIXELS)
                 .reproject(coarse_proj)
                 .reproject(fine_proj))
    winners = fine.updateMask(score.eq(max_score))
    coarse = (winners.reduceResolution(reducer=ee.Reducer.firstNonNull(), bestEffort=False,
                                       maxPixels=MAX_PIXELS)
              .reproject(coarse_proj)
              .rename(BANDS))
    return fine, coarse


def download_fine_embeddings(ee, grid, aef_year, n=N_SUB):
    """Fetch unit AEF vectors for the event grid using geemap's Xarray reader."""
    import geemap

    fg = fine_grid(grid, n)
    fine_proj = ee.Projection(fg["crs"], fg["transform"])
    region = region_geometry(ee, fg, inset_px=0.5)
    fine = build_fine_embedding_image(ee, grid, aef_year, n)
    dataset = geemap.ee_to_xarray(
        dataset=fine,
        geometry=region,
        projection=fine_proj,
    )
    x_dim = next((dim for dim in ("x", "longitude", "lon") if dim in dataset.dims), None)
    y_dim = next((dim for dim in ("y", "latitude", "lat") if dim in dataset.dims), None)
    if x_dim is None or y_dim is None:
        raise ValueError(f"Xarray result must have x/y dimensions; got {dataset.dims}")
    if set(dataset.dims) != {x_dim, y_dim}:
        raise ValueError(f"Expected only spatial dimensions; got {dataset.dims}")

    missing = set(BANDS).difference(dataset.data_vars)
    if missing:
        raise ValueError(f"Xarray result is missing AEF bands: {sorted(missing)}")
    expected = fg
    a, _, c, _, e, f = expected["transform"]
    expected_x = c + a * (np.arange(expected["width"]) + 0.5)
    expected_y = f + e * (np.arange(expected["height"]) + 0.5)
    if not np.allclose(dataset.coords[x_dim].values, expected_x, rtol=0, atol=abs(a) * 1e-6):
        raise ValueError("Downloaded Xarray x coordinates are not aligned with the WSTS fine grid")
    if not np.allclose(dataset.coords[y_dim].values, expected_y, rtol=0, atol=abs(e) * 1e-6):
        raise ValueError("Downloaded Xarray y coordinates are not aligned with the WSTS fine grid")

    fine_array = np.asarray(
        dataset[BANDS].to_array(dim="band").transpose("band", y_dim, x_dim).values,
        dtype=np.float32,
    )
    expected_shape = (expected["height"], expected["width"])
    if fine_array.shape[1:] != expected_shape:
        raise ValueError(f"Downloaded fine grid has shape {fine_array.shape[1:]}; "
                         f"expected {expected_shape}")
    return fine_array


def write_medoid_geotiff(path, medoids, grid):
    """Write (bands, height, width) medoids to a GeoTIFF on the WSTS grid."""
    import rasterio
    from affine import Affine

    expected_shape = (len(BANDS), grid["height"], grid["width"])
    if medoids.shape != expected_shape:
        raise ValueError(f"Medoid array has shape {medoids.shape}; expected {expected_shape}")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid["width"],
        height=grid["height"],
        count=len(BANDS),
        dtype="float32",
        crs=grid["crs"],
        transform=Affine(*grid["transform"]),
        nodata=np.nan,
    ) as dataset:
        dataset.write(medoids.astype(np.float32, copy=False))


def download_medoid_image(ee, coarse, grid, name, path):
    """Download a cloud-computed medoid GeoTIFF directly to a local path."""
    import os
    import tempfile
    from pathlib import Path

    import rasterio
    import requests
    from affine import Affine
    from rasterio.crs import CRS

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "name": name,
        "crs": grid["crs"],
        "crs_transform": grid["transform"],
        "region": region_geometry(ee, grid, inset_px=0.5),
        "format": "GEO_TIFF",
    }
    url = coarse.toFloat().getDownloadURL(params)
    temp_path = None
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=output.parent, suffix=".tif", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temp_file.write(chunk)

        with rasterio.open(temp_path) as dataset:
            if (dataset.width, dataset.height) != (grid["width"], grid["height"]):
                raise ValueError(
                    f"Downloaded medoid has dimensions {(dataset.width, dataset.height)}; "
                    f"expected {(grid['width'], grid['height'])}"
                )
            if dataset.count != len(BANDS):
                raise ValueError(
                    f"Downloaded medoid has {dataset.count} bands; expected {len(BANDS)}"
                )
            if dataset.crs != CRS.from_user_input(grid["crs"]):
                raise ValueError(f"Downloaded medoid CRS {dataset.crs} != {grid['crs']}")
            expected_transform = Affine(*grid["transform"])
            if not np.allclose(tuple(dataset.transform), tuple(expected_transform),
                               rtol=0, atol=max(abs(expected_transform.a),
                                                abs(expected_transform.e)) * 1e-8):
                raise ValueError("Downloaded medoid transform does not match the WSTS grid")
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def export_to_drive(ee, coarse, grid, name, folder):
    task = ee.batch.Export.image.toDrive(
        image=coarse.toFloat(),
        description=name,
        folder=folder,
        fileNamePrefix=name,
        region=region_geometry(ee, grid, inset_px=0.5),
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=int(1e10),
        shardSize=64,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def validate_medoid_image(ee, image, grid):
    """Check band order, output georeferencing, and a sampled medoid before export."""
    validate_grid(grid)
    bands = image.bandNames().getInfo()
    if bands != BANDS:
        raise ValueError(f"Medoid image bands do not match AEF bands: {bands}")

    projection = image.projection().getInfo()
    if projection.get("crs") != grid["crs"]:
        raise ValueError(
            f"Medoid image CRS {projection.get('crs')} does not match WSTS CRS {grid['crs']}"
        )
    expected_transform = np.asarray(grid["transform"], dtype=np.float64)
    actual_transform = np.asarray(projection.get("transform"), dtype=np.float64)
    tolerance = max(abs(expected_transform[0]), abs(expected_transform[4])) * 1e-8
    if (actual_transform.shape != (6,) or not np.isfinite(actual_transform).all()
            or not np.allclose(actual_transform, expected_transform, rtol=0, atol=tolerance)):
        raise ValueError(
            f"Medoid image transform {actual_transform.tolist()} does not match "
            f"WSTS transform {expected_transform.tolist()}"
        )

    feature = (image.sample(
        region=region_geometry(ee, grid),
        numPixels=1,
        seed=0,
        dropNulls=True,
        geometries=False,
    ).first().getInfo())
    if feature is None:
        raise ValueError("Medoid image has no valid pixels in the WSTS grid")
    properties = feature.get("properties", {})
    if any(band not in properties for band in BANDS):
        raise ValueError("Sampled medoid is missing one or more AEF bands")
    vector = np.asarray([properties[band] for band in BANDS], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Sampled medoid contains non-finite values")
    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0, rtol=0, atol=UNIT_NORM_ATOL):
        raise ValueError(f"Sampled medoid is not unit-length (L2 norm is {norm:.6f})")
