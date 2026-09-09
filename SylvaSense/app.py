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
        "trees": 18420,
        "canopy_pct": 78.4,
        "agb": 212.6,
        "carbon": 106.3,
        "loss_pct": 3.1,
        "note": "Moist evergreen canopy with high NDVI contrast along ridges.",
    },
    "Sundarbans — West Bengal": {
        "bbox": [88.85, 21.72, 89.12, 21.95],
        "seed": 42,
        "trees": 12110,
        "canopy_pct": 61.2,
        "agb": 148.0,
        "carbon": 74.0,
        "loss_pct": 6.8,
        "note": "Mangrove mosaic; tidal channels appear as low-backscatter corridors.",
    },
    "Nilgiri Biosphere — Tamil Nadu": {
        "bbox": [76.40, 11.22, 76.68, 11.48],
        "seed": 9,
        "trees": 15680,
        "canopy_pct": 71.9,
        "agb": 188.4,
        "carbon": 94.2,
        "loss_pct": 4.4,
        "note": "Shola–grassland matrix with patchy canopy edges.",
    },
    "Kaziranga Buffer — Assam": {
        "bbox": [93.25, 26.48, 93.48, 26.68],
        "seed": 63,
        "trees": 9800,
        "canopy_pct": 54.6,
        "agb": 126.7,
        "carbon": 63.4,
        "loss_pct": 8.2,
        "note": "Floodplain woodland with seasonal disturbance signatures.",
    },
    "Satpura — Madhya Pradesh": {
        "bbox": [78.20, 22.35, 78.48, 22.58],
        "seed": 28,
        "trees": 14250,
        "canopy_pct": 66.8,
        "agb": 164.1,
        "carbon": 82.1,
        "loss_pct": 5.5,
        "note": "Dry deciduous stand; SAR highlights structure in leaf-off patches.",
    },
}

MAP_H, MAP_W = 540, 860
ANALYSIS_STEPS = [
    "Loading satellite imagery",
    "Processing optical bands",
    "Processing SAR data",
    "Detecting canopy",
    "Estimating biomass",
    "Detecting temporal change",
]
LAYER_META = [
    ("rgb", "RGB", True, "Simulated optical composite"),
    ("ndvi", "NDVI", False, "Vegetation index lookalike"),
    ("sar", "SAR", False, "C-band backscatter lookalike"),
    ("canopy", "Canopy", True, "Crown cover mask"),
    ("biomass", "Biomass", False, "AGB intensity"),
    ("degradation", "Degradation", False, "Disturbance candidates"),
]


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
    degradation = np.clip((disturbance - 0.58) * 2.4 * (1.0 - river), 0, 1)
    degradation = degradation * (canopy > 0.22).astype(np.float32)

    speckle = np.random.default_rng(seed + 7).normal(0, 0.08, (h, w)).astype(np.float32)
    sar = _norm(0.7 * elev + 0.2 * texture + speckle)

    r = np.clip(40 + 70 * (1 - ndvi) + 30 * disturbance + 18 * texture, 0, 255)
    g = np.clip(28 + 150 * ndvi + 20 * canopy - 25 * river, 0, 255)
    b = np.clip(18 + 40 * (1 - ndvi) + 90 * river + 12 * moisture, 0, 255)
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
            (0.0, (12, 18, 14)),
            (0.25, (80, 40, 10)),
            (0.5, (200, 80, 20)),
            (0.78, (220, 30, 40)),
            (1.0, (255, 220, 180)),
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


def blend_layers(layers: dict[str, np.ndarray], active: list[str]) -> np.ndarray:
    order = ["rgb", "ndvi", "sar", "canopy", "biomass", "degradation"]
    selected = [k for k in order if k in active]
    if not selected:
        return layers["rgb"]
    base_key = selected[0]
    acc = layers[base_key].astype(np.float32)
    alphas = {
        "ndvi": 0.55,
        "sar": 0.50,
        "canopy": 0.42,
        "biomass": 0.48,
        "degradation": 0.40,
        "rgb": 0.50,
    }
    for key in selected[1:]:
        overlay = layers[key].astype(np.float32)
        if key in ("canopy", "degradation"):
            strength = layers["canopy_mask" if key == "canopy" else "degradation_raw"]
            strength = np.clip(strength, 0, 1)[..., None]
            a = alphas[key] * (0.25 + 0.75 * strength)
            acc = (1.0 - a) * acc + a * overlay
        else:
            a = alphas.get(key, 0.45)
            acc = (1.0 - a) * acc + a * overlay
    return np.clip(acc, 0, 255).astype(np.uint8)


def annotate_map(rgb: np.ndarray, bbox: list[float], title: str, legend: bool = True) -> Image.Image:
    img = _to_image(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    west, south, east, north = bbox

    for i in range(1, 5):
        x = int(w * i / 5)
        y = int(h * i / 5)
        draw.line([(x, 28), (x, h - 44)], fill=(255, 255, 255, 32), width=1)
        draw.line([(18, y), (w - 18, y)], fill=(255, 255, 255, 32), width=1)
        lon = west + (east - west) * i / 5
        lat = north - (north - south) * i / 5
        draw.text((x + 6, 32), f"{lon:.2f}°", fill=(236, 242, 232, 190))
        draw.text((22, y + 4), f"{lat:.2f}°", fill=(236, 242, 232, 190))

    draw.rounded_rectangle([1, 1, w - 2, h - 2], radius=10, outline=(212, 180, 90, 150), width=2)
    draw.rectangle([0, 0, w, 26], fill=(8, 20, 14, 210))
    draw.text((14, 6), title.upper(), fill=(232, 214, 140, 240))
    draw.rectangle([0, h - 38, w, h], fill=(8, 20, 14, 220))
    draw.text((14, h - 26), "AOI  ·  DEMO TILE  ·  EPSG:4326", fill=(214, 226, 216, 230))
    draw.text((w - 158, h - 26), "10 m GSD (illustrative)", fill=(176, 196, 180, 210))

    if legend:
        lx, ly = w - 168, h - 132
        draw.rounded_rectangle([lx, ly, w - 16, h - 50], radius=8, fill=(8, 18, 12, 200), outline=(201, 162, 39, 90))
        draw.text((lx + 12, ly + 8), "LEGEND", fill=(212, 180, 90, 230))
        swatches = [
            ((70, 160, 80), "Canopy"),
            ((220, 70, 50), "Degradation"),
            ((40, 90, 150), "Water / low"),
        ]
        for i, (color, label) in enumerate(swatches):
            yy = ly + 30 + i * 16
            draw.rectangle([lx + 12, yy, lx + 26, yy + 10], fill=color + (220,))
            draw.text((lx + 34, yy - 2), label, fill=(220, 230, 220, 220))

    return Image.alpha_composite(img, overlay).convert("RGB").filter(ImageFilter.SMOOTH)


def draw_detections(base: Image.Image, layers: dict[str, np.ndarray], seed: int) -> Image.Image:
    rng = np.random.default_rng(seed + 99)
    mask = layers["canopy_mask"]
    deg = layers["degradation_raw"]
    ys, xs = np.where(mask > 0.55)
    img = base.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if len(xs) == 0:
        return base
    n = min(90, len(xs))
    pick = rng.choice(len(xs), size=n, replace=False)
    for k in pick:
        x, y = int(xs[k]), int(ys[k])
        if y > img.size[1] - 50 or y < 30 or x < 24 or x > img.size[0] - 24:
            continue
        rw = int(6 + 10 * mask[y, x])
        rh = int(5 + 8 * mask[y, x])
        disturbed = deg[y, x] > 0.35
        color = (232, 92, 64, 200) if disturbed else (110, 230, 140, 210)
        draw.ellipse([x - rw, y - rh, x + rw, y + rh], outline=color, width=2)
    return Image.alpha_composite(img, layer).convert("RGB")


def sample_chart_frame(ndvi: np.ndarray, biomass: np.ndarray, bbox: list[float]) -> pd.DataFrame:
    step_y, step_x = 18, 22
    rows = []
    h, w = ndvi.shape
    west, south, east, north = bbox
    for i in range(0, h, step_y):
        for j in range(0, w, step_x):
            rows.append(
                {
                    "lon": west + (east - west) * (j / w),
                    "lat": north - (north - south) * (i / h),
                    "NDVI": float(ndvi[i, j]),
                    "AGB_index": float(biomass[i, j]),
                }
            )
    return pd.DataFrame(rows)


def build_geojson(region_name: str, meta: dict[str, Any]) -> str:
    west, south, east, north = meta["bbox"]
    rng = np.random.default_rng(meta["seed"] + 3)
    features = []
    for i in range(12):
        cx = float(rng.uniform(west + 0.02, east - 0.02))
        cy = float(rng.uniform(south + 0.02, north - 0.02))
        dx = float(rng.uniform(0.004, 0.012))
        dy = float(rng.uniform(0.003, 0.010))
        ring = [
            [cx - dx, cy - dy],
            [cx + dx, cy - dy * 0.6],
            [cx + dx * 0.8, cy + dy],
            [cx - dx * 0.7, cy + dy * 0.8],
            [cx - dx, cy - dy],
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"canopy-{i + 1:02d}",
                    "region": region_name,
                    "class": "tree_crown_cluster",
                    "ndvi_mean": round(float(rng.uniform(0.52, 0.86)), 3),
                    "agb_t_ha": round(float(rng.uniform(90, 240)), 1),
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


CSS = """
<style>
.stApp {
    background:
        radial-gradient(900px 420px at 0% 0%, rgba(61, 140, 86, 0.16), transparent 55%),
        radial-gradient(700px 380px at 100% 0%, rgba(212, 180, 90, 0.07), transparent 50%),
        #08140f;
}
[data-testid="stHeader"] { background: rgba(8, 20, 15, 0.85); }
[data-testid="stToolbar"] { visibility: hidden; }

.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1.5rem;
    padding: 0.2rem 0 1rem 0;
    border-bottom: 1px solid rgba(212, 180, 90, 0.22);
    margin-bottom: 1.1rem;
}
.kicker {
    letter-spacing: 0.34em;
    font-size: 0.72rem;
    color: #d4b45a;
    font-weight: 600;
    text-transform: uppercase;
}
.title {
    font-family: Palatino, "Palatino Linotype", Georgia, serif;
    font-size: 2.7rem;
    letter-spacing: 0.12em;
    color: #f4f7f2;
    margin: 0.15rem 0 0.15rem 0;
    line-height: 1;
}
.subtitle {
    color: #a9c0b0;
    font-style: italic;
    font-size: 1.05rem;
    margin: 0;
}
.badge {
    display: inline-block;
    margin-top: 0.55rem;
    padding: 0.2rem 0.72rem;
    border: 1px solid rgba(212, 180, 90, 0.5);
    border-radius: 999px;
    color: #ead79a;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-note {
    max-width: 280px;
    color: #9fb3a6;
    font-size: 0.82rem;
    line-height: 1.45;
    background: rgba(18, 38, 27, 0.7);
    border: 1px solid rgba(125, 180, 140, 0.18);
    border-radius: 14px;
    padding: 0.85rem 1rem;
}
.section-label {
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #d4b45a;
    margin: 0.3rem 0 0.7rem 0;
}
.scene-card {
    background: rgba(14, 32, 22, 0.78);
    border: 1px solid rgba(125, 180, 140, 0.18);
    border-radius: 16px;
    padding: 1rem 1rem 0.85rem 1rem;
}
.scene-k { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8fa898; }
.scene-v { margin: 0.2rem 0 0.85rem 0; color: #e8f0ea; font-size: 0.95rem; }
.step-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin: 0.2rem 0 1rem 0; }
.step {
    font-size: 0.72rem;
    padding: 0.28rem 0.55rem;
    border-radius: 999px;
    border: 1px solid rgba(125, 180, 140, 0.22);
    color: #9fb3a6;
    background: rgba(12, 28, 20, 0.6);
}
.step.done { color: #d8f0d8; border-color: rgba(80, 170, 110, 0.55); background: rgba(40, 90, 60, 0.35); }
.step.active { color: #1a1408; border-color: #d4b45a; background: #d4b45a; }

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(22, 48, 34, 0.95), rgba(12, 28, 20, 0.95));
    border: 1px solid rgba(125, 180, 140, 0.2);
    padding: 0.85rem 0.95rem;
    border-radius: 16px;
}
div[data-testid="stMetric"] label { color: #9fb3a6 !important; }
[data-testid="stImage"] img { border-radius: 14px; }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    background: rgba(18, 38, 27, 0.6);
    border-radius: 10px;
    padding: 0.4rem 0.9rem;
}
footer { visibility: hidden; }
</style>
"""


def init_state() -> None:
    if "analyzed_regions" not in st.session_state:
        st.session_state.analyzed_regions = []
    if "pipeline_step" not in st.session_state:
        st.session_state.pipeline_step = -1
    if "pending_result_layers" not in st.session_state:
        st.session_state.pending_result_layers = False
    if st.session_state.pending_result_layers:
        st.session_state.layer_canopy = True
        st.session_state.layer_degradation = True
        st.session_state.pending_result_layers = False


def altair_style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(labelColor="#c5d4c8", titleColor="#d4b45a", gridColor="#1d3328")
        .configure_title(color="#e8f0ea", fontSize=14, fontWeight=500, anchor="start")
        .configure_legend(labelColor="#c5d4c8", titleColor="#d4b45a")
    )


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="kicker">ORION-PS-03 · Forest Intelligence</div>
            <div class="title">SYLVASENSE</div>
            <p class="subtitle">“Satellite-Powered Forest Intelligence”</p>
            <span class="badge">Prototype / Demo Mode</span>
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
        st.markdown('<div class="section-label">Mission control</div>', unsafe_allow_html=True)
        region = st.selectbox("Forest Region", list(REGIONS.keys()), key="region")
        meta = REGIONS[region]

        st.markdown('<div class="section-label">Layers</div>', unsafe_allow_html=True)
        active_layers: list[str] = []
        cols = st.columns(2)
        for i, (key, label, default, help_text) in enumerate(LAYER_META):
            with cols[i % 2]:
                on = st.toggle(label, value=default, key=f"layer_{key}", help=help_text)
            if on:
                active_layers.append(key)

        st.divider()
        run = st.button("Run Analysis", type="primary", use_container_width=True)
        reset = st.button("Reset Demo", use_container_width=True)
        if reset:
            st.session_state.analyzed_regions = []
            st.session_state.pipeline_step = -1
            st.rerun()

        st.caption(meta["note"])
        st.caption("Prototype / Demo Mode · local synthetic tiles")

    if run:
        progress = st.progress(0, text="Starting pipeline…")
        status_box = st.empty()
        for i, step in enumerate(ANALYSIS_STEPS):
            st.session_state.pipeline_step = i
            progress.progress((i + 1) / (len(ANALYSIS_STEPS) + 1), text=f"{step}…")
            status_box.info(f"**Pipeline:** {step}")
            time.sleep(0.42)
        progress.progress(1.0, text="Analysis complete")
        status_box.success("Analysis complete — demo products ready")
        time.sleep(0.35)
        st.session_state.pipeline_step = len(ANALYSIS_STEPS)
        if region not in st.session_state.analyzed_regions:
            st.session_state.analyzed_regions.append(region)
        st.session_state.pending_result_layers = True
        st.rerun()

    done = region in st.session_state.analyzed_regions
    layers = generate_layers(int(meta["seed"]))
    composite = blend_layers(layers, active_layers)
    map_title = region.split("—")[0].strip()
    map_img = annotate_map(composite, meta["bbox"], map_title)
    display = draw_detections(map_img, layers, int(meta["seed"])) if done else map_img

    steps_html = ['<div class="step-row">']
    for i, step in enumerate(ANALYSIS_STEPS + ["Analysis complete"]):
        cls = "step"
        if done or st.session_state.pipeline_step >= i:
            cls += " done"
        if not done and st.session_state.pipeline_step == i:
            cls = "step active"
        steps_html.append(f'<span class="{cls}">{i + 1}. {step}</span>')
    steps_html.append("</div>")
    st.markdown("".join(steps_html), unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    if done:
        c1.metric("Trees Detected", f"{meta['trees']:,}", "demo crown objects")
        c2.metric("Canopy Density", f"{meta['canopy_pct']:.1f}%", "fractional cover")
        c3.metric("Estimated AGB", f"{meta['agb']:.1f} t/ha", "allometric proxy")
        c4.metric("Carbon Stock", f"{meta['carbon']:.1f} tC/ha", "AGB × 0.5")
        c5.metric("Canopy Loss", f"{meta['loss_pct']:.1f}%", "temporal change (demo)")
    else:
        c1.metric("Trees Detected", "—", "run analysis")
        c2.metric("Canopy Density", "—")
        c3.metric("Estimated AGB", "—", "t/ha")
        c4.metric("Carbon Stock", "—", "tC/ha")
        c5.metric("Canopy Loss", "—", "%")

    st.write("")
    map_col, info_col = st.columns([1.7, 0.5], gap="large")
    with map_col:
        st.markdown('<div class="section-label">Forest intelligence map</div>', unsafe_allow_html=True)
        st.image(display, use_container_width=True)
        if done:
            st.caption(
                "Green ellipses = canopy detections on high-cover pixels. "
                "Red ellipses = disturbance candidates. Synthetic demo overlay."
            )
        else:
            st.caption("Select layers, then click **Run Analysis** to generate canopy and change products.")

    with info_col:
        west, south, east, north = meta["bbox"]
        layer_txt = ", ".join(k.upper() for k in active_layers) if active_layers else "RGB (fallback)"
        status_txt = "Products ready" if done else "Awaiting analysis"
        st.markdown(
            f"""
            <div class="scene-card">
              <div class="scene-k">Region</div>
              <div class="scene-v">{region}</div>
              <div class="scene-k">Bounding box</div>
              <div class="scene-v">{south:.3f}–{north:.3f}°N<br/>{west:.3f}–{east:.3f}°E</div>
              <div class="scene-k">Active layers</div>
              <div class="scene-v">{layer_txt}</div>
              <div class="scene-k">Status</div>
              <div class="scene-v">{status_txt}<br/><span style="color:#d4b45a">Prototype / Demo Mode</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.download_button(
            label="Download GeoJSON",
            data=build_geojson(region, meta),
            file_name="sylvasense_canopy_demo.geojson",
            mime="application/geo+json",
            use_container_width=True,
            disabled=not done,
            help="Sample canopy polygons for this demo AOI (not live model output).",
        )
        if not done:
            st.caption("Enabled after analysis completes.")
        else:
            st.caption("Valid FeatureCollection of synthetic crown clusters.")

    if done:
        tab_map, tab_charts = st.tabs(["Canopy & degradation", "NDVI analytics"])
        with tab_map:
            a, b = st.columns(2)
            with a:
                st.image(
                    annotate_map(layers["canopy"], meta["bbox"], "Canopy detection", legend=False),
                    caption="Canopy detection raster (demo)",
                    use_container_width=True,
                )
            with b:
                st.image(
                    annotate_map(layers["degradation"], meta["bbox"], "Degradation", legend=False),
                    caption="Degradation / disturbance raster (demo)",
                    use_container_width=True,
                )
        with tab_charts:
            df = sample_chart_frame(layers["ndvi_raw"], layers["biomass_raw"], meta["bbox"])
            heat = (
                alt.Chart(df)
                .mark_rect()
                .encode(
                    x=alt.X("lon:Q", title="Longitude"),
                    y=alt.Y("lat:Q", title="Latitude"),
                    color=alt.Color("NDVI:Q", scale=alt.Scale(scheme="yellowgreenblue"), title="NDVI"),
                    tooltip=[
                        alt.Tooltip("lon:Q", format=".3f"),
                        alt.Tooltip("lat:Q", format=".3f"),
                        alt.Tooltip("NDVI:Q", format=".2f"),
                        alt.Tooltip("AGB_index:Q", format=".2f", title="AGB index"),
                    ],
                )
                .properties(height=320, title="Sampled NDVI field — synthetic, hover for values")
                .interactive()
            )
            hist = (
                alt.Chart(pd.DataFrame({"NDVI": layers["ndvi_raw"].ravel()[::50]}))
                .mark_bar(color="#3d9b6e")
                .encode(
                    x=alt.X("NDVI:Q", bin=alt.Bin(maxbins=28), title="NDVI"),
                    y=alt.Y("count()", title="Pixels (subsampled)"),
                )
                .properties(height=260, title="NDVI distribution — demo tile")
            )
            st.altair_chart(altair_style(heat), use_container_width=True)
            st.altair_chart(altair_style(hist), use_container_width=True)

    with st.expander("About this prototype"):
        st.markdown(
            """
            **SylvaSense** is a hackathon prototype for automated forest intelligence.

            This build is **demo mode only**. Map tiles, NDVI, SAR lookalikes, canopy
            detections, AGB, carbon, and change metrics are generated locally from seeded
            noise and region presets. They illustrate the operator workflow — region
            select → layer fusion → analysis → GeoJSON export — and are **not** an
            operational forest inventory.

            A production system would ingest Sentinel-2 and Sentinel-1 scenes, run canopy
            segmentation, apply regional allometry, and detect temporal loss. Those models
            are **not executed here**.
            """
        )


if __name__ == "__main__":
    main()
