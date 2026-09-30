# HydroSentinel-AI

**A Multi-Modal Satellite Data Fusion Framework for Climate Risk Forecasting**

## Abstract
HydroSentinel-AI is a production-grade, deep learning pipeline designed for the predictive forecasting of flash floods and agricultural droughts. By leveraging multi-modal satellite data fusion—specifically Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 Optical imagery—this architecture provides high-fidelity spatio-temporal predictions mapped directly to the UN-FAO climate risk framework. The system implements a robust preprocessing pipeline and a 3D-UNet Convolutional Long Short-Term Memory (ConvLSTM) network to extract non-linear spatial dependencies and temporal climate dynamics, enabling early warning and actionable geospatial intelligence.

## System Architecture

The pipeline is organized into three decoupled, highly scalable modules: Data Ingestion & Preprocessing, Feature Engineering, and Spatio-Temporal Modeling.

### 1. Preprocessing Pipeline
*   **Sentinel-1 SAR Processing:** Employs polarimetric calibration and **Lee filtering** for speckle noise reduction while preserving high-frequency edge details necessary for waterbody delineation.
*   **Sentinel-2 Optical Processing:** Utilizes the **S2cloudless** machine learning algorithm for robust cloud and cloud-shadow masking, ensuring optical data integrity prior to index computation.

### 2. Feature Extraction
*   **Vegetation and Water Indices:** Automated computation of the Normalized Difference Vegetation Index (**NDVI**) and Normalized Difference Water Index (**NDWI**) to quantify biomass health and surface water saturation.
*   **Temporal Stacking:** Co-registration and temporal alignment of SAR backscatter (VV/VH) and optical indices into a unified multi-dimensional tensor.

### 3. Deep Learning Framework
*   **3D-UNet ConvLSTM:** The core predictive engine integrates a 3D-UNet architecture for high-resolution spatial feature extraction with ConvLSTM bottleneck layers to model sequence-to-sequence temporal dynamics. This enables precise pixel-wise forecasting of hydrological anomalies over configurable temporal horizons.

## Mathematical Framework

The quantitative foundations of the feature extraction and risk mapping modules are defined by the following relationships:

### Normalized Difference Vegetation Index (NDVI)
Quantifies vegetation density and health, critical for agricultural drought assessment:
$$NDVI=\frac{NIR-Red}{NIR+Red}$$

### Normalized Difference Water Index (NDWI)
Delineates open water features and enhances soil moisture mapping for flash flood prediction:
$$NDWI=\frac{Green-NIR}{Green+NIR}$$

### UN-FAO Climate Risk Framework
The pipeline outputs are systematically mapped to the UN-FAO risk definition, integrating the probability of a hydrological anomaly with infrastructural exposure and ecological susceptibility:
$$Risk=Hazard \times Exposure \times Vulnerability$$

## Production Deployment Guide

### Prerequisites
*   Ubuntu 20.04/22.04 LTS or RHEL 8
*   NVIDIA GPU (Ampere/Hopper architecture recommended) with CUDA 11.8+
*   Python 3.10+

### Environment Setup
Execute the following shell commands to provision the environment and install dependencies:

```bash
# Clone the repository
git clone [https://github.com/your-org/HydroSentinel-AI.git](https://github.com/your-org/HydroSentinel-AI.git)
cd HydroSentinel-AI

# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip and install core PyTorch components
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install pipeline dependencies
pip install -r requirements.txt
