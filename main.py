import argparse
from pathlib import Path
import numpy as np
import rasterio
import torch

from Demo import process_ndvi
from model import SpatiotemporalUNetConvLSTM
from risk_engine import assess_climate_risk


def run_pipeline(
    red_path: Path,
    nir_path: Path,
    vector_path: Path,
    output_dir: Path,
    threshold: float,
    time_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ndvi_path = output_dir / "ndvi.tif"
    flood_pred_path = output_dir / "flood_prediction.tif"
    risk_output_path = output_dir / "critical_alerts.geojson"

    process_ndvi(red_path, nir_path, ndvi_path)

    with rasterio.open(ndvi_path) as src:
        profile = src.profile.copy()
        ndvi_data = src.read(1)

    input_tensor = torch.from_numpy(ndvi_data).float().unsqueeze(0).unsqueeze(0).unsqueeze(2)
    input_tensor = input_tensor.repeat(1, 1, time_steps, 1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatiotemporalUNetConvLSTM(in_channels=1, out_channels=1).to(device)
    model.eval()

    with torch.no_grad():
        pred_tensor = model(input_tensor.to(device))
        pred_array = pred_tensor.squeeze().cpu().numpy()
        if pred_array.ndim == 3:
            pred_array = pred_array.mean(axis=0)

    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(flood_pred_path, "w", **profile) as dst:
        dst.write(pred_array.astype(np.float32), 1)

    assess_climate_risk(
        vector_path=vector_path,
        raster_path=flood_pred_path,
        output_path=risk_output_path,
        threshold=threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", type=Path, required=True)
    parser.add_argument("--nir", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--time-steps", type=int, default=5)

    args = parser.parse_args()
    run_pipeline(
        red_path=args.red,
        nir_path=args.nir,
        vector_path=args.vector,
        output_dir=args.output_dir,
        threshold=args.threshold,
        time_steps=args.time_steps,
    )