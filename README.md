# 🌳 SylvaSense

### Satellite-Powered Forest Intelligence

**SylvaSense** is a prototype forest-intelligence dashboard designed for automated tree enumeration, canopy analysis, aboveground biomass estimation, carbon assessment, and temporal forest-degradation monitoring using multi-spectral and SAR satellite imagery.

**Problem Statement:** ORION-PS-03 — Automated Tree Enumeration & Aboveground Biomass Estimation from Multi-Spectral & SAR Satellite Imagery

**Team:** Shadow Monarch

---

## 🚀 Live Demo

### 🌐 SylvaSense Dashboard

[Open SylvaSense Live Demo](https://sylvasense-karunya.streamlit.app/?utm_source=chatgpt.com)

The deployed Streamlit dashboard demonstrates the complete forest-monitoring workflow through an interactive interface.

---

## 🎯 Problem

Traditional forest inventory and monitoring can be expensive, time-consuming, and difficult to scale across large forest regions.

SylvaSense aims to transform satellite observations into actionable forest intelligence by providing:

* 🌲 Tree enumeration
* 🌿 Canopy intelligence
* 🌱 Aboveground biomass estimation
* 🌍 Carbon-stock estimation
* 📉 Forest degradation detection
* 📊 Temporal forest monitoring
* 🗺️ GIS-compatible GeoJSON outputs

---

## 💡 Proposed Solution

SylvaSense combines optical multispectral information with Synthetic Aperture Radar (SAR) information to create a multi-modal forest-analysis workflow.

```text
Optical + SAR Data
        ↓
Preprocessing
        ↓
Optical + SAR Fusion
        ↓
Canopy Segmentation
        ↓
Tree Enumeration
        ↓
Biomass Estimation
        ↓
Carbon Estimation
        ↓
Temporal Change Detection
        ↓
Forest Intelligence Dashboard
```

---

## 🛰️ Technical Approach

### 1. Data Acquisition

The proposed production system is designed around:

* Sentinel-2 multispectral imagery
* Sentinel-1 SAR imagery
* Landsat data
* Reference / LiDAR data where available

### 2. Preprocessing

The workflow includes:

* Cloud-aware optical preprocessing
* SAR preprocessing and filtering
* Raster alignment

### 3. Multi-Modal Fusion

Optical and SAR information are combined to provide complementary information about vegetation and forest structure.

### 4. Canopy Segmentation

The proposed production architecture uses computer-vision segmentation to identify forest canopy regions.

### 5. Tree Enumeration

Detected canopy instances can be represented as individual tree/crown features with spatial and structural attributes.

### 6. Biomass & Carbon

Aboveground biomass is estimated using multispectral, SAR and canopy-related features.

The proposed carbon calculation is:

```text
Carbon Stock = AGB × Carbon Fraction
```

and:

```text
CO₂e = Carbon Stock × 44/12
```

### 7. Temporal Change Detection

Multi-temporal observations are compared to identify:

* Canopy loss
* Tree-count changes
* Degradation patterns
* Potential forest-risk hotspots

---

## 🖥️ Dashboard Features

The SylvaSense dashboard provides interactive views for:

### 🗺️ Map Layers

* RGB
* NDVI
* SAR
* Canopy
* Biomass
* Degradation

### 🌲 Tree Intelligence

* Tree detection
* Tree crowns
* Tree centroids
* Crown area
* Canopy density

### 🌱 Biomass & Carbon

* Aboveground biomass
* Carbon stock
* CO₂ equivalent
* Biomass distribution

### ❤️ Forest Health

The prototype dashboard combines indicators such as:

* NDVI
* Canopy density
* SAR stability
* Canopy loss

to produce a forest-health assessment.

### 📅 Temporal Intelligence

The dashboard provides:

* Year-wise tree counts
* Canopy-loss information
* Tree-count changes
* Degradation risk
* Degradation heatmaps

---

## 🧪 Current Prototype

The current deployed version is a **prototype/demo implementation** built with Python and Streamlit.

It demonstrates:

* The end-to-end workflow
* Interactive map layers
* Forest metrics
* Tree/canopy outputs
* Biomass and carbon outputs
* Temporal-change visualization
* GeoJSON output structure

**Important:** The current demonstration uses locally generated synthetic optical/SAR-like raster data and seeded demo detections. It is **not currently performing live Sentinel-1/Sentinel-2 inference**.

---

## 🏗️ Production Architecture

The planned production architecture is:

```text
Sentinel-1 & Sentinel-2
          ↓
   Data Acquisition
          ↓
     Preprocessing
          ↓
   Optical + SAR Fusion
          ↓
   Canopy Segmentation
          ↓
    Tree Enumeration
          ↓
   Biomass Regression
          ↓
  Temporal Change Detection
          ↓
      GeoJSON Output
          ↓
  SylvaSense Dashboard
```

### Planned AI / CV Models

* Mask2Former — canopy segmentation
* YOLOv8-OBB — tree/crown detection
* PyTorch — model training and inference
* Regression models — biomass estimation

---

## 🛠️ Technology Stack

### Prototype

* Python
* Streamlit
* NumPy
* Pandas
* Pillow
* Altair
* JSON / GeoJSON

### Planned Production Stack

* PyTorch
* YOLOv8-OBB
* Mask2Former
* Sentinel Hub
* Google Earth Engine
* Sentinel-1
* Sentinel-2
* Rasterio
* GDAL
* FastAPI
* Mapbox / Deck.gl

---

## 📍 Supported Demo Regions

The prototype provides demonstration presets for:

* Western Ghats — Silent Valley, Kerala
* Sundarbans — West Bengal
* Nilgiri Biosphere — Tamil Nadu
* Kaziranga Buffer — Assam
* Satpura — Madhya Pradesh

---

## ⚙️ Running Locally

### 1. Clone / download the project

```bash
git clone <your-repository-url>
cd SylvaSense
```

### 2. Install dependencies

```bash
pip install streamlit numpy pandas pillow altair
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Impact

SylvaSense aims to support:

**Forest Inventory**

Automated tree and canopy enumeration at scale.

**Climate Intelligence**

Biomass and carbon-stock estimation for forest monitoring.

**Early Warning**

Temporal canopy-change and degradation monitoring.

**Scalable Monitoring**

Repeatable satellite-based forest assessment from regional to larger scales.

**GIS Integration**

Structured GeoJSON outputs for downstream geospatial workflows.

---

## 🔮 Future Scope

The next development stages include:

1. Live Sentinel-1/Sentinel-2 data ingestion
2. Trained canopy-segmentation models
3. Trained tree/crown detection models
4. Real biomass-regression models
5. Large-area cloud inference
6. Improved temporal change detection
7. Automated forest-degradation alerts
8. Integration with GIS and forest-management systems

---

## 👥 Team

### Shadow Monarch

**Project:** SylvaSense
**Problem Statement:** ORION-PS-03
**Domain:** Remote Sensing • Computer Vision • AI • Forest Intelligence

---

## 📜 References

The project is based on established remote-sensing missions, geospatial technologies and research in:

* Sentinel-1 SAR
* Sentinel-2 multispectral imagery
* Landsat
* Forest canopy detection
* Tree-crown segmentation
* Aboveground biomass estimation
* SAR/optical fusion
* Forest change detection
* Computer vision and deep learning
