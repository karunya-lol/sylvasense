"""SylvaSense — prototype dashboard for ORION-PS-03 (demo / synthetic data)."""

from __future__ import annotations

import json
import time
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

st.set_page_config(
    page_title="SylvaSense | Forest Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGIONS: dict[str, dict[str, Any]] = {
    "Western Ghats — Silent Valley, Kerala": {
        "bbox": [76.35, 11.02, 76.52, 11.18],
        "seed": 17,
        "trees": 1247,
        "crown_area_ha": 68.4,
        "canopy_pct": 74.8,
        "mean_crown_m2": 54.9,
        "agb": 184.6,
        "carbon": 92.3,
        "co2e": 338.5,
        "loss_pct": 4.7,
        "tree_delta": -82,
        "risk": "MEDIUM",
        "health": 87,
        "health_status": "HEALTHY",
        "ndvi_ind": 0.74,
        "sar_stability": 0.81,
        "trees_2024": 1329,
        "trees_2025": 1288,
        "trees_2026": 1247,
        "note": "Moist evergreen canopy with high NDVI contrast along ridges.",
    },
    "Sundarbans — West Bengal": {
        "bbox": [88.85, 21.72, 89.12, 21.95],
        "seed": 42,
        "trees": 1086,
        "crown_area_ha": 51.2,
        "canopy_pct": 61.2,
        "mean_crown_m2": 47.1,
        "agb": 148.0,
        "carbon": 74.0,
        "co2e": 271.6,
        "loss_pct": 6.8,
        "tree_delta": -94,
        "risk": "HIGH",
        "health": 72,
        "health_status": "WATCH",
        "ndvi_ind": 0.61,
        "sar_stability": 0.68,
        "trees_2024": 1210,
        "trees_2025": 1140,
        "trees_2026": 1086,
        "note": "Mangrove mosaic; tidal channels appear as low-backscatter corridors.",
    },
    "Nilgiri Biosphere — Tamil Nadu": {
        "bbox": [76.40, 11.22, 76.68, 11.48],
        "seed": 9,
        "trees": 1318,
        "crown_area_ha": 64.1,
        "canopy_pct": 71.9,
        "mean_crown_m2": 48.6,
        "agb": 188.4,
        "carbon": 94.2,
        "co2e": 345.7,
        "loss_pct": 4.4,
        "tree_delta": -71,
        "risk": "MEDIUM",
        "health": 84,
        "health_status": "HEALTHY",
        "ndvi_ind": 0.71,
        "sar_stability": 0.79,
        "trees_2024": 1412,
        "trees_2025": 1360,
        "trees_2026": 1318,
        "note": "Shola–grassland matrix with patchy canopy edges.",
    },
    "Kaziranga Buffer — Assam": {
        "bbox": [93.25, 26.48, 93.48, 26.68],
        "seed": 63,
        "trees": 942,
        "crown_area_ha": 42.8,
        "canopy_pct": 54.6,
        "mean_crown_m2": 45.4,
        "agb": 126.7,
        "carbon": 63.4,
        "co2e": 232.7,
        "loss_pct": 8.2,
        "tree_delta": -118,
        "risk": "HIGH",
        "health": 64,
        "health_status": "STRESSED",
        "ndvi_ind": 0.54,
        "sar_stability": 0.62,
        "trees_2024": 1124,
        "trees_2025": 1028,
        "trees_2026": 942,
        "note": "Floodplain woodland with seasonal disturbance signatures.",
    },
    "Satpura — Madhya Pradesh": {
        "bbox": [78.20, 22.35, 78.48, 22.58],
        "seed": 28,
        "trees": 1174,
        "crown_area_ha": 58.9,
        "canopy_pct": 66.8,
        "mean_crown_m2": 50.2,
        "agb": 164.1,
        "carbon": 82.1,
        "co2e": 301.3,
        "loss_pct": 5.5,
        "tree_delta": -88,
        "risk": "MEDIUM",
        "health": 78,
        "health_status": "WATCH",
        "ndvi_ind": 0.66,
        "sar_stability": 0.74,
        "trees_2024": 1290,
        "trees_2025": 1232,
        "trees_2026": 1174,
        "note": "Dry deciduous stand; SAR highlights structure in leaf-off patches.",
    },
}

MAP_H, MAP_W = 560, 900
ANALYSIS_STEPS = [
    "Satellite imagery acquisition",
    "Optical preprocessing",
    "SAR preprocessing",
    "Optical + SAR fusion",
    "Canopy segmentation",
    "Tree enumeration",
    "Biomass estimation",
    "Temporal change detection",
]
MAP_PRODUCTS = ["RGB", "NDVI", "SAR", "CANOPY", "BIOMASS", "DEGRADATION"]
PRODUCT_KEY = {
    "RGB": "rgb",
    "NDVI": "ndvi",
    "SAR": "sar",
    "CANOPY": "canopy",
    "BIOMASS": "biomass",
    "DEGRADATION": "degradation",
}


def _fractal_noise(h: int, w: int, seed: int, octaves: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for i in range(octaves):
        gh = max(2, h // (2 ** (octaves - i)))
        gw = max(2, w // (2 ** (octaves - i)))
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = np.array(
            Image.fromarray((grid * 255).astype(np.uint8), mode="L").resize(
                (w, h), Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        ) / 255.0
        acc += layer * amp
        total += amp
        amp *= 0.52
    return acc / total


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _colormap(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    t = np.clip(values, 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.float32)
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        span = max(t1 - t0, 1e-6)
        mask = (t >= t0) & (t <= t1)
        u = ((t - t0) / span)[..., None]
        c0a = np.array(c0, dtype=np.float32)
        c1a = np.array(c1, dtype=np.float32)
        rgb[mask] = (1.0 - u[mask]) * c0a + u[mask] * c1a
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb, mode="RGB")


@st.cache_data(show_spinner=False)
def generate_layers(seed: int) -> dict[str, np.ndarray]:
    h, w = MAP_H, MAP_W
    elev = _fractal_noise(h, w, seed)
    texture = _fractal_noise(h, w, seed + 11, octaves=5)
    moisture = _fractal_noise(h, w, seed + 23, octaves=4)
    disturbance = _fractal_noise(h, w, seed + 41, octaves=4)

    canopy = _norm(0.62 * elev + 0.28 * moisture - 0.18 * disturbance)
    ndvi = _norm(0.55 * canopy + 0.30 * moisture + 0.15 * (1.0 - disturbance))
    ndvi = np.clip(0.18 + 0.72 * ndvi, 0, 1)

    yy, xx = np.mgrid[0:h, 0:w]
    river = np.exp(-(((xx / w - 0.42) * 7 + np.sin(yy / 38.0) * 0.35) ** 2) / 0.012)
    ndvi = np.clip(ndvi - 0.45 * river, 0, 1)
    canopy = np.clip(canopy - 0.55 * river, 0, 1)

    canopy_mask = (canopy > 0.46).astype(np.float32)
    biomass = np.clip(canopy * (0.55 + 0.45 * ndvi), 0, 1)
    degradation = np.clip((disturbance - 0.52) * 2.2 * (1.0 - river), 0, 1)
    degradation = degradation * (canopy > 0.18).astype(np.float32)

    speckle = np.random.default_rng(seed + 7).normal(0, 0.08, (h, w)).astype(np.float32)
    sar = _norm(0.7 * elev + 0.2 * texture + speckle)

    r = np.clip(38 + 72 * (1 - ndvi) + 28 * disturbance + 16 * texture, 0, 255)
    g = np.clip(26 + 155 * ndvi + 18 * canopy - 28 * river, 0, 255)
    b = np.clip(16 + 38 * (1 - ndvi) + 95 * river + 10 * moisture, 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

    ndvi_rgb = _colormap(
        ndvi,
        [
            (0.0, (40, 20, 8)),
            (0.25, (166, 97, 26)),
            (0.45, (223, 194, 125)),
            (0.62, (128, 205, 193)),
            (0.80, (53, 151, 143)),
            (1.0, (8, 64, 40)),
        ],
    )
    sar_rgb = _colormap(
        sar,
        [(0.0, (8, 10, 14)), (0.45, (70, 78, 88)), (0.75, (160, 168, 176)), (1.0, (230, 236, 240))],
    )
    canopy_rgb = _colormap(
        canopy_mask * canopy,
        [(0.0, (12, 16, 14)), (0.35, (20, 70, 36)), (0.7, (46, 160, 72)), (1.0, (190, 240, 140))],
    )
    biomass_rgb = _colormap(
        biomass,
        [
            (0.0, (18, 12, 8)),
            (0.3, (84, 48, 5)),
            (0.55, (191, 129, 45)),
            (0.78, (223, 194, 90)),
            (1.0, (255, 255, 204)),
        ],
    )
    deg_rgb = _colormap(
        degradation,
        [
            (0.0, (10, 18, 16)),
            (0.22, (40, 90, 40)),
            (0.45, (230, 190, 50)),
            (0.68, (230, 110, 30)),
            (1.0, (190, 20, 30)),
        ],
    )
    return {
        "rgb": rgb,
        "ndvi": ndvi_rgb,
        "sar": sar_rgb,
        "canopy": canopy_rgb,
        "biomass": biomass_rgb,
        "degradation": deg_rgb,
        "ndvi_raw": ndvi,
        "biomass_raw": biomass,
        "degradation_raw": degradation,
        "canopy_mask": canopy_mask,
    }


def blend_product(layers: dict[str, np.ndarray], product: str, show_degradation: bool) -> np.ndarray:
    base = layers[PRODUCT_KEY[product]].astype(np.float32)
    if show_degradation and product != "DEGRADATION":
        deg = layers["degradation_raw"][..., None]
        overlay = layers["degradation"].astype(np.float32)
        a = np.clip(deg * 0.72, 0, 0.72)
        base = (1.0 - a) * base + a * overlay
    return np.clip(base, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def sample_trees(seed: int, n: int = 110) -> list[dict[str, Any]]:
    layers = generate_layers(seed)
    mask = layers["canopy_mask"]
    deg = layers["degradation_raw"]
    bio = layers["biomass_raw"]
    rng = np.random.default_rng(seed + 99)
    ys, xs = np.where(mask > 0.55)
    if len(xs) == 0:
        return []
    pick = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    trees = []
    for i, k in enumerate(pick, start=1):
        x, y = int(xs[k]), int(ys[k])
        rw = int(7 + 11 * mask[y, x] + rng.uniform(-1.5, 1.5))
        rh = int(6 + 9 * mask[y, x] + rng.uniform(-1.2, 1.2))
        area = round(float(np.pi * rw * rh * 0.42 + rng.uniform(8, 18)), 1)
        trees.append(
            {
                "tree_id": f"T-{i:04d}",
                "x": x,
                "y": y,
                "rw": max(5, rw),
                "rh": max(4, rh),
                "crown_area": area,
                "canopy_density": round(float(0.45 + 0.5 * mask[y, x]), 3),
                "biomass_estimate": round(float(80 + 180 * bio[y, x]), 1),
                "degraded": bool(deg[y, x] > 0.38),
                "deg_level": "high" if deg[y, x] > 0.62 else ("moderate" if deg[y, x] > 0.38 else "low"),
            }
        )
    return trees


def forest_boundary_pts(w: int, h: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(4)
    pts = []
    for t in np.linspace(0, 2 * np.pi, 28, endpoint=False):
        jitter = 0.04 * rng.normal()
        rx = 0.42 + jitter
        ry = 0.40 + 0.03 * np.sin(3 * t)
        x = int(w * (0.50 + rx * np.cos(t) * 0.92))
        y = int(h * (0.50 + ry * np.sin(t) * 0.88))
        pts.append((max(24, min(w - 24, x)), max(36, min(h - 48, y))))
    return pts


def compose_map(
    rgb: np.ndarray,
    bbox: list[float],
    title: str,
    trees: list[dict[str, Any]],
    *,
    show_crowns: bool,
    show_centroids: bool,
    show_boundary: bool,
    analyzed: bool,
) -> Image.Image:
    img = _to_image(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    west, south, east, north = bbox

    for i in range(1, 5):
        x = int(w * i / 5)
        y = int(h * i / 5)
        draw.line([(x, 30), (x, h - 46)], fill=(255, 255, 255, 28), width=1)
        draw.line([(18, y), (w - 18, y)], fill=(255, 255, 255, 28), width=1)
        lon = west + (east - west) * i / 5
        lat = north - (north - south) * i / 5
        draw.text((x + 6, 34), f"{lon:.2f}°", fill=(220, 230, 226, 180))
        draw.text((22, y + 4), f"{lat:.2f}°", fill=(220, 230, 226, 180))

    if show_boundary:
        pts = forest_boundary_pts(w, h)
        draw.line(pts + [pts[0]], fill=(90, 210, 255, 210), width=3)

    if analyzed:
        for t in trees:
            x, y, rw, rh = t["x"], t["y"], t["rw"], t["rh"]
            if y < 34 or y > h - 50 or x < 20 or x > w - 20:
                continue
            if show_crowns:
                color = (232, 92, 64, 200) if t["degraded"] else (120, 230, 150, 210)
                fill = (232, 92, 64, 28) if t["degraded"] else (90, 200, 120, 36)
                draw.ellipse([x - rw, y - rh, x + rw, y + rh], outline=color, fill=fill, width=2)
            if show_centroids:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 230, 120, 230))

    draw.rounded_rectangle([1, 1, w - 2, h - 2], radius=10, outline=(62, 207, 142, 140), width=2)
    draw.rectangle([0, 0, w, 28], fill=(6, 12, 16, 220))
    draw.text((14, 7), f"{title.upper()}  ·  SYNTHETIC AOI", fill=(180, 255, 210, 240))
    draw.rectangle([0, h - 40, w, h], fill=(6, 12, 16, 230))
    draw.text((14, h - 28), "DEMO TILE  ·  EPSG:4326  ·  prototype detections", fill=(200, 214, 208, 230))
    draw.text((w - 168, h - 28), "10 m GSD (illustrative)", fill=(150, 170, 164, 210))

    lx, ly = 16, h - 128
    draw.rounded_rectangle([lx, ly, lx + 188, h - 50], radius=8, fill=(6, 12, 16, 210), outline=(62, 207, 142, 80))
    draw.text((lx + 12, ly + 8), "LEGEND  ·  DEMO", fill=(62, 207, 142, 230))
    draw.ellipse([lx + 14, ly + 32, lx + 20, ly + 38], fill=(255, 230, 120, 230))
    draw.text((lx + 28, ly + 28), "Tree centroid", fill=(220, 230, 224, 220))
    draw.rectangle([lx + 12, ly + 50, lx + 22, ly + 60], outline=(120, 230, 150, 230), width=2)
    draw.text((lx + 28, ly + 48), "Canopy crown", fill=(220, 230, 224, 220))
    draw.polygon([(lx + 12, ly + 80), (lx + 28, ly + 72), (lx + 24, ly + 86)], outline=(90, 210, 255, 230))
    draw.text((lx + 34, ly + 70), "Forest boundary", fill=(220, 230, 224, 220))

    return Image.alpha_composite(img, overlay).convert("RGB").filter(ImageFilter.SMOOTH)


def pixel_to_lonlat(x: int, y: int, bbox: list[float]) -> tuple[float, float]:
    west, south, east, north = bbox
    lon = west + (east - west) * (x / MAP_W)
    lat = north - (north - south) * (y / MAP_H)
    return lon, lat


def build_geojson(region_name: str, meta: dict[str, Any], trees: list[dict[str, Any]], bbox: list[float]) -> str:
    features = []
    for t in trees[:36]:
        lon, lat = pixel_to_lonlat(t["x"], t["y"], bbox)
        dx = 0.0018 + t["rw"] * 0.00012
        dy = 0.0015 + t["rh"] * 0.00012
        ring = [
            [lon - dx, lat - dy],
            [lon + dx, lat - dy * 0.55],
            [lon + dx * 0.85, lat + dy],
            [lon - dx * 0.7, lat + dy * 0.75],
            [lon - dx, lat - dy],
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "tree_id": t["tree_id"],
                    "crown_area": t["crown_area"],
                    "canopy_density": t["canopy_density"],
                    "biomass_estimate": t["biomass_estimate"],
                    "region": region_name,
                    "source": "SylvaSense prototype (synthetic)",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return json.dumps(
        {
            "type": "FeatureCollection",
            "name": "sylvasense_canopy_demo",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": features,
        },
        indent=2,
    )


def sample_chart_frame(ndvi: np.ndarray, biomass: np.ndarray, bbox: list[float]) -> pd.DataFrame:
    rows = []
    h, w = ndvi.shape
    west, south, east, north = bbox
    for i in range(0, h, 20):
        for j in range(0, w, 24):
            rows.append(
                {
                    "lon": west + (east - west) * (j / w),
                    "lat": north - (north - south) * (i / h),
                    "NDVI": float(ndvi[i, j]),
                    "AGB_index": float(biomass[i, j]),
                }
            )
    return pd.DataFrame(rows)


CSS = """
<style>
.stApp {
    background:
        radial-gradient(900px 480px at 0% -10%, rgba(46, 160, 120, 0.16), transparent 50%),
        radial-gradient(800px 400px at 100% 0%, rgba(40, 90, 160, 0.12), transparent 46%),
        #070c10;
}
[data-testid="stHeader"] { background: rgba(7, 12, 16, 0.88); }
[data-testid="stToolbar"] { visibility: hidden; }

.hero { display:flex; justify-content:space-between; gap:1.5rem; align-items:flex-end;
    padding:0.15rem 0 1.05rem 0; border-bottom:1px solid rgba(62,207,142,0.18); margin-bottom:1rem; }
.kicker { letter-spacing:0.32em; font-size:0.7rem; color:#7ee0b0; text-transform:uppercase; font-weight:600; }
.title { font-family: Palatino, Georgia, serif; font-size:2.65rem; letter-spacing:0.14em;
    color:#f3f7f4; margin:0.12rem 0 0.18rem 0; line-height:1; }
.subtitle { color:#9bb0a8; font-style:italic; font-size:1.05rem; margin:0; }
.badge { display:inline-block; margin-top:0.5rem; padding:0.22rem 0.75rem; border-radius:999px;
    border:1px solid rgba(62,207,142,0.45); color:#b7f0d2; font-size:0.72rem; letter-spacing:0.1em; }
.hero-note { max-width:290px; color:#9aaea6; font-size:0.8rem; line-height:1.45;
    background:rgba(12,20,26,0.8); border:1px solid rgba(62,207,142,0.16); border-radius:14px; padding:0.85rem 1rem; }
.section { font-size:0.74rem; letter-spacing:0.16em; text-transform:uppercase; color:#7ee0b0; margin:0.2rem 0 0.65rem 0; }
.card { background:rgba(12,20,26,0.78); border:1px solid rgba(90,140,120,0.18); border-radius:16px; padding:1rem; }
.muted { color:#8fa39b; font-size:0.82rem; }
.kpi-row { display:grid; grid-template-columns:repeat(5,1fr); gap:0.7rem; margin:0.2rem 0 1rem 0; }
.kpi { background:linear-gradient(180deg, rgba(16,32,36,0.95), rgba(10,16,20,0.95));
    border:1px solid rgba(62,207,142,0.16); border-radius:16px; padding:0.9rem 0.95rem; }
.kpi .l { font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:#8fa39b; }
.kpi .v { font-size:1.55rem; color:#f3f7f4; margin-top:0.28rem; font-weight:650; }
.kpi .s { font-size:0.72rem; color:#7ee0b0; margin-top:0.18rem; }
.pipe { display:flex; flex-direction:column; gap:0.35rem; }
.prow { display:flex; justify-content:space-between; align-items:center;
    background:rgba(10,16,20,0.7); border:1px solid rgba(90,140,120,0.14); border-radius:10px; padding:0.42rem 0.7rem; }
.pst { font-size:0.68rem; letter-spacing:0.1em; padding:0.12rem 0.5rem; border-radius:999px; }
.queued { color:#8fa39b; border:1px solid #3a4a46; }
.processing { color:#1a1408; background:#d4b45a; }
.complete { color:#062014; background:#3ecf8e; }
.health-num { font-size:3.2rem; color:#f3f7f4; line-height:1; font-weight:700; }
.health-st { letter-spacing:0.18em; font-size:0.9rem; color:#3ecf8e; margin-top:0.3rem; }
.ind { display:flex; justify-content:space-between; font-size:0.85rem; padding:0.35rem 0; border-bottom:1px solid rgba(255,255,255,0.05); }
.arch { font-family: ui-monospace, Menlo, monospace; font-size:0.86rem; line-height:1.55; color:#c5d6ce; }
div[data-testid="stMetric"] {
    background:linear-gradient(180deg, rgba(16,32,36,0.95), rgba(10,16,20,0.95));
    border:1px solid rgba(62,207,142,0.16); padding:0.85rem 0.95rem; border-radius:16px;
}
[data-testid="stImage"] img { border-radius:14px; }
footer { visibility:hidden; }
</style>
"""


def init_state() -> None:
    if "analyzed_regions" not in st.session_state:
        st.session_state.analyzed_regions = []
    if "pipeline_step" not in st.session_state:
        st.session_state.pipeline_step = -1
    if "analysis_complete_flag" not in st.session_state:
        st.session_state.analysis_complete_flag = False
    if st.session_state.get("pending_deg_overlay"):
        st.session_state.ov_deg = True
        st.session_state.pending_deg_overlay = False


def altair_style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(labelColor="#c5d4c8", titleColor="#7ee0b0", gridColor="#1a2a28")
        .configure_title(color="#e8f0ea", fontSize=14, fontWeight=500, anchor="start")
        .configure_legend(labelColor="#c5d4c8", titleColor="#7ee0b0")
    )


def kpi_html(done: bool, meta: dict[str, Any]) -> str:
    if done:
        vals = [
            ("TREES DETECTED", f"{meta['trees']:,}", "prototype instances"),
            ("CANOPY DENSITY", f"{meta['canopy_pct']:.1f}%", "fractional cover"),
            ("AGB", f"{meta['agb']:.1f} t/ha", "demo estimate"),
            ("CARBON STOCK", f"{meta['carbon']:.1f} tC/ha", "AGB × 0.5"),
            ("CANOPY LOSS", f"{meta['loss_pct']:.1f}%", "2024–2026 demo"),
        ]
    else:
        vals = [
            ("TREES DETECTED", "—", "awaiting analysis"),
            ("CANOPY DENSITY", "—", ""),
            ("AGB", "—", "t/ha"),
            ("CARBON STOCK", "—", "tC/ha"),
            ("CANOPY LOSS", "—", "%"),
        ]
    cells = "".join(
        f'<div class="kpi"><div class="l">{a}</div><div class="v">{b}</div><div class="s">{c}</div></div>'
        for a, b, c in vals
    )
    return f'<div class="kpi-row">{cells}</div>'


def pipeline_html(current: int, done: bool) -> str:
    rows = []
    for i, name in enumerate(ANALYSIS_STEPS):
        if done or current > i:
            stt, cls = "COMPLETE", "complete"
        elif current == i:
            stt, cls = "PROCESSING", "processing"
        else:
            stt, cls = "QUEUED", "queued"
        rows.append(
            f'<div class="prow"><span>{i + 1:02d}  {name}</span><span class="pst {cls}">{stt}</span></div>'
        )
    extra = ""
    if done:
        extra = '<div class="prow"><span><b>ANALYSIS COMPLETE</b></span><span class="pst complete">READY</span></div>'
    return f'<div class="pipe">{"".join(rows)}{extra}</div>'


def run_pipeline() -> None:
    board = st.empty()
    bar = st.progress(0, text="Initialising prototype pipeline…")
    for i, step in enumerate(ANALYSIS_STEPS):
        st.session_state.pipeline_step = i
        board.markdown(pipeline_html(i, False), unsafe_allow_html=True)
        bar.progress((i + 1) / len(ANALYSIS_STEPS), text=f"{step} — PROCESSING")
        time.sleep(0.38)
    board.markdown(pipeline_html(len(ANALYSIS_STEPS), True), unsafe_allow_html=True)
    bar.progress(1.0, text="ANALYSIS COMPLETE")
    st.success("ANALYSIS COMPLETE — prototype products ready (demo mode).")
    time.sleep(0.3)


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="kicker">ORION-PS-03 · Earth Observation</div>
            <div class="title">SYLVASENSE</div>
            <p class="subtitle">Satellite-Powered Forest Intelligence</p>
            <span class="badge">PROTOTYPE • MULTI-SPECTRAL + SAR</span>
          </div>
          <div class="hero-note">
            Synthetic optical + SAR-like rasters for demonstration.
            No live Sentinel-1 / Sentinel-2 inference is running in this build.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="section">Target Forest</div>', unsafe_allow_html=True)
        region = st.selectbox("Forest region", list(REGIONS.keys()), key="region")
        meta = REGIONS[region]
        west, south, east, north = meta["bbox"]
        c_lat = round((south + north) / 2, 4)
        c_lon = round((west + east) / 2, 4)
        if st.session_state.get("synced_region") != region:
            st.session_state.aoi_lat = c_lat
            st.session_state.aoi_lon = c_lon
            st.session_state.aoi_name = region.split("—")[0].strip()
            st.session_state.synced_region = region

        lat = st.number_input("Latitude", format="%.4f", key="aoi_lat")
        lon = st.number_input("Longitude", format="%.4f", key="aoi_lon")
        aoi = st.text_input("Area of Interest", key="aoi_name")
        st.caption("Defaults to the selected region centroid if left unchanged.")

        st.markdown('<div class="section">Map product</div>', unsafe_allow_html=True)
        product = st.radio("Layer", MAP_PRODUCTS, index=0, key="map_product", horizontal=False)
        st.caption("Switch the central raster. Overlays remain independent.")
        show_crowns = st.toggle("Tree crowns", value=True, key="ov_crowns")
        show_centroids = st.toggle("Tree centroids", value=True, key="ov_cent")
        show_boundary = st.toggle("Forest boundary", value=True, key="ov_bound")
        show_deg = st.toggle("Degradation overlay", value=False, key="ov_deg")

        st.divider()
        analyze = st.button("Analyze Target Forest", type="primary", use_container_width=True)
        run = st.button("Run Analysis", use_container_width=True)
        reset = st.button("Reset Demo", use_container_width=True)
        if reset:
            st.session_state.analyzed_regions = []
            st.session_state.pipeline_step = -1
            st.session_state.analysis_complete_flag = False
            st.rerun()
        st.caption(meta["note"])
        st.caption("Prototype / Demo Mode · local synthetic tiles")

    bbox = [
        float(lon) - (east - west) / 2,
        float(lat) - (north - south) / 2,
        float(lon) + (east - west) / 2,
        float(lat) + (north - south) / 2,
    ]
    analysis_key = f"{region}|{round(float(lat), 4)}|{round(float(lon), 4)}"

    if analyze or run:
        run_pipeline()
        if analysis_key not in st.session_state.analyzed_regions:
            st.session_state.analyzed_regions.append(analysis_key)
        st.session_state.pipeline_step = len(ANALYSIS_STEPS)
        st.session_state.analysis_complete_flag = True
        st.session_state.pending_deg_overlay = True
        st.rerun()

    done = analysis_key in st.session_state.analyzed_regions
    layers = generate_layers(int(meta["seed"]))
    trees = sample_trees(int(meta["seed"]))
    raster = blend_product(layers, product, show_deg and done)
    title = aoi or region.split("—")[0].strip()
    display = compose_map(
        raster,
        bbox,
        title,
        trees if done else [],
        show_crowns=show_crowns and done,
        show_centroids=show_centroids and done,
        show_boundary=show_boundary,
        analyzed=done,
    )

    st.markdown(kpi_html(done, meta), unsafe_allow_html=True)

    left, right = st.columns([1.55, 0.45], gap="large")
    with left:
        st.markdown('<div class="section">Forest map / canopy view</div>', unsafe_allow_html=True)
        st.image(display, use_container_width=True)
        st.caption(
            f"Active product: **{product}** · prototype / demo detections · not live satellite inference."
        )
    with right:
        st.markdown('<div class="section">Analysis pipeline</div>', unsafe_allow_html=True)
        step = st.session_state.pipeline_step
        st.markdown(pipeline_html(step if not done else len(ANALYSIS_STEPS), done), unsafe_allow_html=True)
        if done:
            st.success("ANALYSIS COMPLETE")
        else:
            st.info("Click **Analyze Target Forest** to run the prototype pipeline.")

        st.markdown('<div class="section">Export</div>', unsafe_allow_html=True)
        st.download_button(
            label="Export Canopy GeoJSON",
            data=build_geojson(region, meta, trees, bbox),
            file_name="sylvasense_canopy_demo.geojson",
            mime="application/geo+json",
            use_container_width=True,
            disabled=not done,
            help="Demo canopy polygons with tree_id, crown_area, canopy_density, biomass_estimate.",
        )
        st.download_button(
            label="Download GeoJSON",
            data=build_geojson(region, meta, trees, bbox),
            file_name="sylvasense_canopy.geojson",
            mime="application/geo+json",
            use_container_width=True,
            disabled=not done,
        )
        st.caption("Valid GeoJSON FeatureCollection · synthetic canopy polygons.")

    if done:
        t1, t2 = st.columns([1.15, 0.85], gap="large")
        with t1:
            st.markdown('<div class="section">Tree enumeration</div>', unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            e1.metric("Trees Detected", f"{meta['trees']:,}", "prototype / demo detections")
            e2.metric("Detected Crown Area", f"{meta['crown_area_ha']:.1f} ha")
            e3, e4 = st.columns(2)
            e3.metric("Canopy Density", f"{meta['canopy_pct']:.1f}%")
            e4.metric("Mean Crown Area", f"{meta['mean_crown_m2']:.1f} m²")
            st.caption("● Tree centroid  ▢ Canopy crown  ▱ Forest boundary")
            st.caption("Individual outlines on the map are a visual subset of the demo enumeration.")

            st.markdown('<div class="section">Biomass & carbon intelligence</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.metric("Estimated Aboveground Biomass", f"{meta['agb']:.1f} t/ha")
            b2.metric("Carbon Stock", f"{meta['carbon']:.1f} tC/ha")
            b3.metric("CO₂ Equivalent", f"{meta['co2e']:.1f} tCO₂e/ha")
            st.caption("Prototype estimate — calibrated model integration planned")
            bio_vals = layers["biomass_raw"].ravel()[::40] * meta["agb"]
            bio_df = pd.DataFrame({"AGB (t/ha, scaled demo)": bio_vals})
            hist = (
                alt.Chart(bio_df)
                .mark_bar(color="#3ecf8e")
                .encode(
                    x=alt.X("AGB (t/ha, scaled demo):Q", bin=alt.Bin(maxbins=24), title="AGB (t/ha)"),
                    y=alt.Y("count()", title="Pixels (subsampled)"),
                )
                .properties(height=220, title="Biomass distribution — synthetic tile")
            )
            st.altair_chart(altair_style(hist), use_container_width=True)

        with t2:
            st.markdown('<div class="section">Forest health score</div>', unsafe_allow_html=True)
            color = "#3ecf8e" if meta["health"] >= 80 else ("#d4b45a" if meta["health"] >= 70 else "#e07a5f")
            st.markdown(
                f"""
                <div class="card">
                  <div class="health-num">{meta['health']} <span style="font-size:1.2rem;color:#8fa39b">/ 100</span></div>
                  <div class="health-st" style="color:{color}">{meta['health_status']}</div>
                  <div class="muted" style="margin:0.6rem 0 0.4rem 0">Prototype indicators — not live EO scores</div>
                  <div class="ind"><span>NDVI</span><span>{meta['ndvi_ind']:.2f}</span></div>
                  <div class="ind"><span>Canopy Density</span><span>{meta['canopy_pct']:.1f}%</span></div>
                  <div class="ind"><span>SAR Stability</span><span>{meta['sar_stability']:.2f}</span></div>
                  <div class="ind"><span>Canopy Loss</span><span>{meta['loss_pct']:.1f}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section">Temporal canopy intelligence</div>', unsafe_allow_html=True)
        st.caption("Temporal comparison identifies areas of canopy decline and potential forest disturbance.")
        y1, y2, y3, m1, m2, m3 = st.columns(6)
        y1.metric("2024 trees", f"{meta['trees_2024']:,}")
        y2.metric("2025 trees", f"{meta['trees_2025']:,}")
        y3.metric("2026 trees", f"{meta['trees_2026']:,}")
        m1.metric("Canopy Loss", f"{meta['loss_pct']:.1f}%")
        m2.metric("Tree Count Change", f"{meta['tree_delta']}")
        m3.metric("Degradation Risk", meta["risk"])

        d1, d2 = st.columns([1.2, 0.8])
        with d1:
            st.image(
                compose_map(
                    layers["degradation"],
                    bbox,
                    "Degradation heatmap",
                    trees,
                    show_crowns=False,
                    show_centroids=False,
                    show_boundary=True,
                    analyzed=True,
                ),
                caption="Low (green) · Moderate (amber) · High (red) — synthetic disturbance field",
                use_container_width=True,
            )
        with d2:
            tl = pd.DataFrame(
                {
                    "Year": ["2024", "2025", "2026"],
                    "Trees": [meta["trees_2024"], meta["trees_2025"], meta["trees_2026"]],
                }
            )
            line = (
                alt.Chart(tl)
                .mark_line(point=True, color="#e07a5f", strokeWidth=3)
                .encode(x="Year:N", y=alt.Y("Trees:Q", title="Tree count (demo)"))
                .properties(height=260, title="2024 → 2025 → 2026  (prototype timeline)")
            )
            st.altair_chart(altair_style(line), use_container_width=True)
            st.markdown(
                """
                <div class="card">
                <div class="muted">Intensity classes</div>
                <div class="ind"><span>Low degradation</span><span>stable canopy</span></div>
                <div class="ind"><span>Moderate degradation</span><span>edge / thinning</span></div>
                <div class="ind"><span>High degradation</span><span>disturbance candidate</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section">Canopy & NDVI products</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            st.image(
                compose_map(
                    layers["canopy"],
                    bbox,
                    "Canopy regions",
                    trees,
                    show_crowns=True,
                    show_centroids=True,
                    show_boundary=True,
                    analyzed=True,
                ),
                caption="Canopy regions + instance outlines (demo)",
                use_container_width=True,
            )
        with p2:
            df = sample_chart_frame(layers["ndvi_raw"], layers["biomass_raw"], bbox)
            heat = (
                alt.Chart(df)
                .mark_rect()
                .encode(
                    x=alt.X("lon:Q", title="Longitude"),
                    y=alt.Y("lat:Q", title="Latitude"),
                    color=alt.Color("NDVI:Q", scale=alt.Scale(scheme="yellowgreenblue")),
                    tooltip=["lon:Q", "lat:Q", "NDVI:Q", "AGB_index:Q"],
                )
                .properties(height=340, title="Sampled NDVI field — synthetic")
                .interactive()
            )
            st.altair_chart(altair_style(heat), use_container_width=True)

    with st.expander("System Architecture"):
        st.markdown(
            """
            <div class="arch">
            Sentinel-2 Optical<br/>
            +<br/>
            Sentinel-1 SAR<br/>
            ↓<br/>
            Preprocessing<br/>
            ↓<br/>
            Multi-Modal Fusion<br/>
            ↓<br/>
            Canopy Instance Segmentation<br/>
            ↓<br/>
            Tree Enumeration<br/>
            ↓<br/>
            AGB Regression<br/>
            ↓<br/>
            Carbon Estimation<br/>
            ↓<br/>
            Temporal Change Detection<br/>
            ↓<br/>
            SylvaSense Dashboard
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("**Prototype / Demo Pipeline** — this build uses locally generated rasters and seeded detections, not live Sentinel processing.")

    with st.expander("About this prototype"):
        st.markdown(
            """
            **SylvaSense** is a hackathon prototype for automated forest intelligence.

            Map tiles, NDVI, SAR lookalikes, canopy detections, AGB, carbon, health, and
            change metrics are generated locally from seeded noise and region presets.
            They illustrate the operator workflow and are **not** an operational forest inventory.
            """
        )


if __name__ == "__main__":
    main()
