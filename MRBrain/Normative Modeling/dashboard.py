import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from typing import List, Dict, Tuple

# =============================
# Page config & Styling
# =============================
st.set_page_config(
    page_title="Normative Modeling Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle, colorful theming tweaks
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .stMetric {background: linear-gradient(135deg, #f0f4ff 0%, #eefaf3 100%); border-radius: 16px; padding: 0.5rem 0.75rem;}
    .metric-label {color: #4f46e5 !important;}
    .metric-value {color: #0ea5e9 !important;}
    div[data-testid="stSidebar"] {background: linear-gradient(180deg, #f8fafc 0%, #f0f9ff 100%);} 
    .pill {display:inline-block; padding:6px 10px; border-radius:9999px; background:#ecfeff; color:#0369a1; font-size:12px; margin-right:6px;}
    .danger {background:#fef2f2; color:#b91c1c;}
    .success {background:#ecfdf5; color:#047857;}
    .neutral {background:#eef2ff; color:#4338ca;}
    .caption {font-size:12px; color:#64748b}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Utilities
# =============================
def parse_filename(fname: str) -> Tuple[str, str]:
    """Extract (sex, region) from a filename like 'female_left_hippocampus.xlsx'."""
    base = os.path.basename(fname)
    name, ext = os.path.splitext(base)
    if ext.lower() == ".gz":
        name, ext2 = os.path.splitext(name)
    # Expect prefixes male_ or female_
    if name.startswith("male_"):
        return ("male", name[len("male_"):])
    if name.startswith("female_"):
        return ("female", name[len("female_"):])
    # Fallback: unknown, return the whole as region
    return ("unknown", name)

@st.cache_data(show_spinner=False)
def scan_folder(base_folder: str) -> Dict[str, List[str]]:
    """Return available regions per sex found in base_folder."""
    regions: Dict[str, set] = {"male": set(), "female": set()}
    if not os.path.isdir(base_folder):
        return {"male": [], "female": []}

    for fname in os.listdir(base_folder):
        if not fname.lower().endswith(".xlsx"):
            continue
        sex, region = parse_filename(fname)
        if sex in regions:
            regions[sex].add(region)
    # sort natural
    def sort_key(x):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", x)]

    return {k: sorted(list(v), key=sort_key) for k, v in regions.items()}

@st.cache_data(show_spinner=False)
def load_percentiles(base_folder: str, sex: str, region: str) -> pd.DataFrame:
    """Load a percentile table for the given sex & region. Returns a DataFrame with 'Age' and percentile columns."""
    candidate = os.path.join(base_folder, f"{sex}_{region}.xlsx")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"File not found: {candidate}")
    df = pd.read_excel(candidate)

    # Normalize columns: ensure 'Age' exists and percentile columns are numeric labels
    # Expected: 'Age' + '1th', '2th', ..., '99th'
    # We'll rename to integer percent labels (1,2,...,99), leaving 'Age' as is
    rename_map = {}
    for c in df.columns:
        if isinstance(c, str) and c.lower().endswith("th") and c[:-2].isdigit():
            rename_map[c] = int(c[:-2])
    df = df.rename(columns=rename_map)
    # Ensure Age is first column for convenience
    cols = [c for c in ["Age"] + sorted([x for x in df.columns if isinstance(x, int)]) if c in df.columns]
    df = df[cols]
    return df

# =============================
# Sidebar Controls
# =============================
st.sidebar.title("🧭 Controls")

# Base folder (defaults to user's path from prompt)
DEFAULT_BASE = r"Percentiles"
base_folder = st.sidebar.text_input("Percentiles folder", value=DEFAULT_BASE, help="Folder with files like 'female_left_hippocampus.xlsx'.")

available = scan_folder(base_folder)

if not any(available.values()):
    st.sidebar.markdown("<span class='pill danger'>No files detected</span>", unsafe_allow_html=True)
    st.stop()

sex = st.sidebar.radio("Sex", options=["female", "male"], horizontal=True, index=0)

# Search + select region
query = st.sidebar.text_input("Search brain region", placeholder="Type to filter… e.g., hippocampus, thalamus")
regions = available.get(sex, [])
if query:
    q = query.strip().lower()
    regions_filtered = [r for r in regions if q in r.lower()]
    # Promote startswith matches to top
    starts = [r for r in regions_filtered if r.lower().startswith(q)]
    contains = [r for r in regions_filtered if not r.lower().startswith(q)]
    regions_filtered = starts + contains
else:
    regions_filtered = regions

if not regions_filtered:
    st.sidebar.info("No regions match your search. Try a different keyword.")
    st.stop()

region = st.sidebar.selectbox("Brain region", options=regions_filtered, index=0)

# Percentile selection
DEFAULT_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# We'll read the DF to know available percentile columns
try:
    df_region = load_percentiles(base_folder, sex, region)
    percentile_cols = [c for c in df_region.columns if isinstance(c, int)]
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

selected_pcts = st.sidebar.multiselect(
    "Percentile curves",
    options=sorted(percentile_cols),
    default=[p for p in DEFAULT_PCTS if p in percentile_cols],
    help="Add or remove percentile curves in real time.",
)

# Add smoothing toggle
smooth = st.sidebar.checkbox("Smooth curves", value=True, help="Apply light smoothing to percentile curves for visual clarity.")

# =============================
# User Points Manager
# =============================
st.sidebar.markdown("---")
st.sidebar.subheader("➕ Add your data points")
if "user_points" not in st.session_state:
    st.session_state.user_points = pd.DataFrame(columns=["Age", "Volume", "Label"])  # Label optional

with st.sidebar.form("add_point_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        age_in = st.number_input("Age", min_value=int(df_region["Age"].min()), max_value=int(df_region["Age"].max()), value=int(df_region["Age"].median()))
    with c2:
        vol_in = st.number_input("Volume", min_value=0.0, step=1.0, format="%0.3f")
    label_in = st.text_input("Label (optional)", placeholder="e.g., Subject A")
    submitted = st.form_submit_button("Add point")
    if submitted:
        new_row = {"Age": age_in, "Volume": vol_in, "Label": label_in}
        st.session_state.user_points = pd.concat([st.session_state.user_points, pd.DataFrame([new_row])], ignore_index=True)

# Delete selected points
if not st.session_state.user_points.empty:
    st.sidebar.markdown("<span class='pill neutral'>Manage points</span>", unsafe_allow_html=True)
    idxs = st.sidebar.multiselect("Select rows to delete", options=list(st.session_state.user_points.index))
    if st.sidebar.button("Delete selected", use_container_width=True):
        st.session_state.user_points = st.session_state.user_points.drop(idxs).reset_index(drop=True)

# =============================
# Main Area
# =============================
st.title("🧠 Normative Modeling Dashboard")
st.caption("Interactive percentile curves by age with real-time overlays of your data points.")

# Headline metrics
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Sex", sex.capitalize())
with colB:
    st.metric("Region", region.replace("_", " ").title())
with colC:
    st.metric("Curves shown", len(selected_pcts))

# Prepare chart data
age = df_region["Age"].astype(float).values

# Optional smoothing using a simple moving average to avoid heavy dependencies

def smooth_series(y: np.ndarray, window: int = 5) -> np.ndarray:
    if not smooth or window <= 1:
        return y
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")

# Build Plotly figure
fig = go.Figure()

# Add percentile curves
if selected_pcts:
    for p in sorted(selected_pcts):
        if p not in df_region.columns:
            continue
        y = df_region[p].astype(float).values
        y_sm = smooth_series(y)
        fig.add_trace(go.Scatter(
            x=age,
            y=y_sm,
            mode="lines",
            name=f"{p}th",
            hovertemplate="Age %{x}<br>Percentile {p}th<br>Vol %{y:.3f}<extra></extra>",
        ))

# Fill between 5-95 or 25-75 if available (nice ribbon)
for lo, hi, label in [(5, 95, "5–95 band"), (25, 75, "25–75 band")]:
    if lo in df_region.columns and hi in df_region.columns:
        y_lo = smooth_series(df_region[lo].astype(float).values)
        y_hi = smooth_series(df_region[hi].astype(float).values)
        fig.add_trace(go.Scatter(
            x=np.concatenate([age, age[::-1]]),
            y=np.concatenate([y_hi, y_lo[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0.06)',
            line=dict(color='rgba(0,0,0,0)'),
            name=label,
            hoverinfo='skip',
            showlegend=True,
        ))

# Add user points
if not st.session_state.user_points.empty:
    pts = st.session_state.user_points
    fig.add_trace(
        go.Scatter(
            x=pts["Age"],
            y=pts["Volume"],
            mode="markers+text" if ("Label" in pts and pts["Label"].notna().any()) else "markers",
            text=pts["Label"] if "Label" in pts else None,
            textposition="top center",
            marker=dict(size=10, line=dict(width=1)),
            name="Your points",
            hovertemplate=(
                "Age %{x}<br>Vol %{y:.3f}" + ("<br>%{text}" if ("Label" in pts) else "") + "<extra></extra>"
            ),
        )
    )

fig.update_layout(
    height=650,
    margin=dict(l=20, r=20, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis_title="Age",
    yaxis_title="Volume",
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# =============================
# Data Views
# =============================
with st.expander("🔎 Inspect raw percentile table"):
    st.dataframe(df_region, use_container_width=True)

with st.expander("📝 Your points"):
    if st.session_state.user_points.empty:
        st.info("No points added yet. Use the sidebar to add (Age, Volume) points.")
    else:
        st.dataframe(st.session_state.user_points, use_container_width=True)

# =============================
# Export options
# =============================
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Download your points as CSV"):
        csv = st.session_state.user_points.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save CSV",
            data=csv,
            file_name=f"user_points_{sex}_{region}.csv",
            mime='text/csv',
        )
with col2:
    # Export the plotted subset of percentiles for reproducibility
    if st.button("Download plotted curves as CSV"):
        cols = ["Age"] + [p for p in sorted(selected_pcts) if p in df_region.columns]
        sub = df_region[cols]
        csv = sub.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save CSV",
            data=csv,
            file_name=f"percentiles_{sex}_{region}.csv",
            mime='text/csv',
        )

# Footer
st.markdown(
    """
    <div class='caption'>Tip: Use the sidebar search to quickly jump to any brain region. Type the first few letters (e.g., <b>hyp</b>) to see matches like <b>hypothalamus</b> at the top.</div>
    """,
    unsafe_allow_html=True,
)