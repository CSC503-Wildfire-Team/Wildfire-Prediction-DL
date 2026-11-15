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
    i = int(np.argmax((Vn @ Vn.T).sum(axis=1)))
    return i, Vn[i]


def cosine_medoid_fast(V):
    """O(n) version using sum_j v_i.v_j = v_i . S. Returns (index, unit vector)."""
    Vn = _unit(V)
    S = Vn.sum(axis=0)
    i = int(np.argmax(Vn @ S))
    return i, Vn[i]


def block_medoids(fine, n=N_SUB):
    """
    fine: (C, n*H, n*W) array, NaN where invalid. Returns (C, H, W) float32 medoids, NaN where a
    cell has no valid pixel. Each n x n block of fine pixels is one WSTS pixel.
    """
    c, hn, wn = fine.shape
    H, W = hn // n, wn // n
    out = np.full((c, H, W), np.nan, dtype=np.float32)
    for r in range(H):
        for q in range(W):
            blk = fine[:, r * n:(r + 1) * n, q * n:(q + 1) * n].reshape(c, -1).T
            if len(_unit(blk)) > 0:
                _, m = cosine_medoid_fast(blk)
                out[:, r, q] = m
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


def fine_grid(grid, n=N_SUB):
    a, b, c, d, e, f = grid["transform"]
    return {"crs": grid["crs"], "transform": [a / n, b / n, c, d / n, e / n, f],
            "width": grid["width"] * n, "height": grid["height"] * n}


def grid_extent(grid):
    a, b, c, d, e, f = grid["transform"]
    if abs(b) > 0 or abs(d) > 0:
        raise ValueError("Rotated/sheared grids are not supported (expected a north-up GeoTIFF).")
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


def build_medoid_image(ee, grid, aef_year, n=N_SUB):
    """
    Returns (fine, coarse):
      fine   : unit-length 64-band embedding on a cell-aligned ~10 m grid (for validation)
      coarse : 64-band cosine medoid on the exact WSTS grid (bands A00..A63)
    """
    coarse_proj = ee.Projection(grid["crs"], grid["transform"])
    fg = fine_grid(grid, n)
    fine_proj = ee.Projection(fg["crs"], fg["transform"])

    tiles = (ee.ImageCollection(AEF_COLLECTION)
             .filterDate(f"{aef_year}-01-01", f"{aef_year + 1}-01-01")
             .filterBounds(region_geometry(ee, grid, inset_px=-2))
             .mosaic().select(BANDS))
    unit = tiles.divide(tiles.pow(2).reduce(ee.Reducer.sum()).sqrt())   # guard against rounding
    fine = unit.reproject(fine_proj)                                    # nearest-neighbour resample

    # S = sum of unit vectors in each WSTS pixel, copied back to every fine pixel of that cell
    S = fine.reduceResolution(reducer=ee.Reducer.sum(), bestEffort=False,
                              maxPixels=MAX_PIXELS).reproject(coarse_proj)
    S_fine = S.reproject(fine_proj)

    # medoid score of each fine pixel = v . S  (argmax of this is the exact cosine medoid)
    score = fine.multiply(S_fine).reduce(ee.Reducer.sum()).rename("score")

    # keep the 64 values of the fine pixel with the max score in each cell
    stacked = score.addBands(fine)
    coarse = (stacked.reduceResolution(reducer=ee.Reducer.max(len(BANDS) + 1), bestEffort=False,
                                       maxPixels=MAX_PIXELS)
              .reproject(coarse_proj)
              .rename(["score"] + BANDS)       # positional rename; fails loudly if band count differs
              .select(BANDS))
    return fine, coarse


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
