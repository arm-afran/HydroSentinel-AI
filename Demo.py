import argparse
from pathlib import Path
import numpy as np
import rasterio
import xarray as xr


def process_ndvi(red_file: Path, nir_file: Path, output_file: Path) -> None:
    with rasterio.open(red_file) as s2_red_src, rasterio.open(nir_file) as s2_nir_src:
        profile = s2_red_src.profile.copy()

        red_band = s2_red_src.read(1).astype(np.float32)
        nir_band = s2_nir_src.read(1).astype(np.float32)

        s2_red_da = xr.DataArray(red_band, dims=["y", "x"])
        s2_nir_da = xr.DataArray(nir_band, dims=["y", "x"])

        denominator = s2_nir_da + s2_red_da
        ndvi_raster = xr.where(denominator == 0, np.nan, (s2_nir_da - s2_red_da) / denominator)

        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(ndvi_raster.to_numpy().astype(np.float32), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", type=Path, required=True)
    parser.add_argument("--nir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    process_ndvi(args.red, args.nir, args.output)