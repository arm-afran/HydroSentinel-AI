import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def assess_climate_risk(
    vector_path: Path,
    raster_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    gdf = gpd.read_file(vector_path)

    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        nodata_val = src.nodata if src.nodata is not None else -9999.0
        mean_depths = []
        max_depths = []

        for geom in gdf.geometry:
            try:
                masked_data, _ = mask(src, [geom], crop=True, nodata=nodata_val)
                valid_vals = masked_data[masked_data != nodata_val]
                if valid_vals.size > 0:
                    mean_depths.append(float(np.mean(valid_vals)))
                    max_depths.append(float(np.max(valid_vals)))
                else:
                    mean_depths.append(0.0)
                    max_depths.append(0.0)
            except (ValueError, AttributeError):
                mean_depths.append(0.0)
                max_depths.append(0.0)

    gdf["mean_depth"] = mean_depths
    gdf["max_depth"] = max_depths

    exposure = gdf.get("population", 1.0).fillna(1.0)
    sensitivity = gdf.get("infra_val", 1.0).fillna(1.0)

    vulnerability = (gdf["mean_depth"] * 0.3 + gdf["max_depth"] * 0.7) * (exposure * sensitivity)
    max_vuln = vulnerability.max()
    gdf["risk_metric"] = vulnerability / max_vuln if max_vuln > 0 else 0.0

    critical_alerts = gdf[gdf["risk_metric"] >= threshold].copy()

    if critical_alerts.crs and critical_alerts.crs.to_epsg() != 4326:
        critical_alerts = critical_alerts.to_crs(epsg=4326)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    critical_alerts.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    assess_climate_risk(args.vector, args.raster, args.output, args.threshold)