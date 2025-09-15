# app.py — Kids Food Insecurity Ops Dashboard (Lite, no matplotlib)
import os, sys, io, json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ---------- Page ----------
st.set_page_config(page_title="Kids Food Insecurity Ops Dashboard (MN)", layout="wide")
st.title("Kids Food Insecurity Ops — Minnesota")
st.caption(f"Python: {sys.version.split()[0]} • Streamlit: {st.__version__}")

with st.expander("Debug (env & files)"):
    try:
        st.write("CWD:", os.getcwd())
        st.write("Top-level files:", sorted(os.listdir(".")))
        if os.path.isdir("data"):
            st.write("data/ →", sorted(os.listdir("data")))
    except Exception as e:
        st.error(f"Debug listing failed: {e}")

# ---------- Sidebar: scope ----------
st.sidebar.header("Population focus")
segment = st.sidebar.selectbox("Segment", ["Children (0–17)"], index=0)  # fixed to children

st.sidebar.header("Weights (sum to 100) — Children")
w1 = st.sidebar.slider("Household Risk (child)", 0, 100, 32)
w2 = st.sidebar.slider("Access & Environment",   0, 100, 22)
w3 = st.sidebar.slider("Experiential (child)",   0, 100, 24)
w4 = st.sidebar.slider("Resilience (child)",     0, 100, 12)
w5 = st.sidebar.slider("Policy Buffer (child)",  0, 100, 10)

st.sidebar.header("RAG Thresholds")
thr_green = st.sidebar.slider("Green max", 0, 100, 40)
thr_amber = st.sidebar.slider("Amber max", 0, 100, 60)
thr_red   = st.sidebar.slider("Red max",   0, 100, 80)

st.sidebar.header("Policy Levers (Children) — ∆ risk reduction %")
sim_snap   = st.sidebar.slider("SNAP (HHs w/ kids)", 0, 50, 8)
sim_school = st.sidebar.slider("School Meals (NSLP/SSO/SFSP)", 0, 50, 10)
sim_summer = st.sidebar.slider("Summer EBT / Sites", 0, 50, 6)
policy_total_reduction = (sim_snap + sim_school + sim_summer) / 100.0

st.sidebar.header("Overlays & Data")
use_mmg = st.sidebar.checkbox("Overlay: Map the Meal Gap — Children", True)
use_lila = st.sidebar.checkbox("Overlay: USDA LI/LA (Food Access) proxy", True)
mmg_upload = st.sidebar.file_uploader("Upload mmg_children.csv (optional)", type=["csv"])
lila_upload = st.sidebar.file_uploader("Upload li_la.csv (optional)", type=["csv"])

# ---------- 1) Load Base (Child) Data ----------
st.subheader("1) Load Base Data (Children)")

@st.cache_data(show_spinner=False)
def load_base_df() -> pd.DataFrame:
    base = Path(__file__).resolve().parent / "data" / "sample_child_dashboard.csv"
    return pd.read_csv(base)

if mmg_upload is not None:
    mmg_df = pd.read_csv(mmg_upload)
else:
    mmg_df = None

if lila_upload is not None:
    lila_df = pd.read_csv(lila_upload)
else:
    lila_df = None

df = load_base_df()

# ---------- 2) Join Overlays (optional) ----------
st.subheader("2) Overlays")
notes = []

# MMG children overlay
if use_mmg:
    if mmg_df is not None:
        mmg = mmg_df.copy()
    else:
        # use columns already in sample file (mmg_child_rate, mmg_child_count, mmg_meal_cost)
        mmg = df[["fips","mmg_child_rate","mmg_child_count","mmg_meal_cost"]].copy()
    mmg_cols = [c for c in ["mmg_child_rate","mmg_child_count","mmg_meal_cost"] if c in mmg.columns]
    if "fips" in mmg.columns and mmg_cols:
        df = df.drop(columns=[c for c in mmg_cols if c in df.columns], errors="ignore") \
               .merge(mmg[["fips"]+mmg_cols], on="fips", how="left")
        notes.append("MMG Children overlay applied.")
    else:
        st.warning("MMG overlay skipped — columns missing (need fips + mmg_child_rate/count).")

# LI/LA proxy overlay
if use_lila:
    if lila_df is not None:
        lila = lila_df.copy()
    else:
        lila = df[["fips","li_la_share"]].copy()
    if "fips" in lila.columns and "li_la_share" in lila.columns:
        df = df.drop(columns=["li_la_share"], errors="ignore").merge(lila[["fips","li_la_share"]], on="fips", how="left")
        notes.append("LI/LA (access) overlay applied.")
    else:
        st.warning("LI/LA overlay skipped — need fips + li_la_share.")

if notes:
    st.info(" • " + " • ".join(notes))

# ---------- 3) Compute Child CFIRI ----------
st.subheader("3) Child Composite (CFIRI-Child) & RAG")

required_cols = [
    "county","fips","lat","lon",
    "child_population",
    "child_household_risk","child_access_score","child_experience_score",
    "child_resilience_score","child_policy_buffer"
]
miss = [c for c in required_cols if c not in df.columns]
if miss:
    st.error(f"Missing columns: {miss}")
    st.stop()

# clean numerics
for c in ["child_population","child_household_risk","child_access_score","child_experience_score","child_resilience_score","child_policy_buffer"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["child_household_risk","child_access_score","child_experience_score","child_resilience_score","child_policy_buffer"])
for c in ["child_household_risk","child_access_score","child_experience_score","child_resilience_score","child_policy_buffer"]:
    df[c] = df[c].clip(0,100)

total_w = max(1, w1 + w2 + w3 + w4 + w5)
weights = dict(
    child_household_risk = w1/total_w,
    child_access_score   = w2/total_w,
    child_experience_score = w3/total_w,
    child_resilience_score = w4/total_w,
    child_policy_buffer  = w5/total_w
)

def compute_child_cfiri(r):
    raw = (
        weights["child_household_risk"] * r["child_household_risk"] +
        weights["child_access_score"] * r["child_access_score"] +
        weights["child_experience_score"] * r["child_experience_score"] +
        weights["child_resilience_score"] * r["child_resilience_score"] +
        weights["child_policy_buffer"] * r["child_policy_buffer"]
    )
    return max(0.0, min(100.0, raw * (1.0 - policy_total_reduction)))

df["CFIRI_CHILD"] = df.apply(compute_child_cfiri, axis=1)

def rag_label(v):
    if v <= thr_green: return "Green"
    if v <= thr_amber: return "Amber"
    if v <= thr_red:   return "Red"
    return "Critical"

df["RAG_CHILD"] = df["CFIRI_CHILD"].apply(rag_label)

# ---------- 4) Results & Export ----------
st.subheader("4) Results")
cols = ["county","fips","child_population","child_household_risk","child_access_score",
        "child_experience_score","child_resilience_score","child_policy_buffer",
        "CFIRI_CHILD","RAG_CHILD"]
# include overlays if present
if "mmg_child_rate" in df.columns: cols += ["mmg_child_rate"]
if "mmg_child_count" in df.columns: cols += ["mmg_child_count"]
if "mmg_meal_cost" in df.columns: cols += ["mmg_meal_cost"]
if "li_la_share" in df.columns: cols += ["li_la_share"]

st.dataframe(df[cols].sort_values("CFIRI_CHILD", ascending=False), use_container_width=True)

st.subheader("RAG Summary (children)")
counts = df["RAG_CHILD"].value_counts().reindex(["Green","Amber","Red","Critical"]).fillna(0).astype(int)
st.write({k:int(v) for k, v in counts.items()})
st.markdown(f"**Legend** — Green ≤ **{thr_green}**, Amber ≤ **{thr_amber}**, Red ≤ **{thr_red}**, Critical > **{thr_red}**")

export = df[cols].copy()
export.insert(0, "segment", "children")
export.insert(1, "weights", f"{w1}/{w2}/{w3}/{w4}/{w5}")
export.insert(2, "policy_reduction_pct", int(policy_total_reduction * 100))
st.download_button("Download results CSV (children)", export.to_csv(index=False).encode(), "mn_child_cfiri_results.csv")

# ---------- 5) Map ----------
st.subheader("5) Map — CFIRI (Children)")
m = df.copy()
m["size"] = 1000 * (m["CFIRI_CHILD"].clip(0,100)/100.0 + 0.2)
color_map = {"Green":[0,153,0],"Amber":[255,191,0],"Red":[220,53,69],"Critical":[128,0,0]}
m["color"] = m["RAG_CHILD"].map(color_map)

tooltip = {"html": "<b>{county}</b><br/>CFIRI (Child): {CFIRI_CHILD}<br/>RAG: {RAG_CHILD}",
           "style": {"backgroundColor": "steelblue", "color": "white"}}

st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer("ScatterplotLayer", data=m,
                      get_position=["lon","lat"],
                      get_radius="size",
                      get_fill_color="color",
                      pickable=True)],
    initial_view_state=pdk.ViewState(latitude=46.0, longitude=-94.0, zoom=5),
    tooltip=tooltip
))

# ---------- 6) Quick impact lens (illustrative) ----------
st.subheader("6) Illustrative Impact Lens (Children)")
st.caption("Simple what-if: applies overall policy reduction to composite. For rigorous policy analysis use SNAP/WIC/school-meal microsimulation inputs.")

if "mmg_child_count" in df.columns and "mmg_child_rate" in df.columns:
    baseline = df[["county","fips","child_population","mmg_child_rate","mmg_child_count"]].copy()
    baseline["expected_child_fi_count"] = baseline["child_population"] * (baseline["mmg_child_rate"]/100.0)
    baseline["expected_child_fi_count_after"] = baseline["expected_child_fi_count"] * (1.0 - policy_total_reduction)
    delta = int(baseline["expected_child_fi_count"].sum() - baseline["expected_child_fi_count_after"].sum())
    st.write(f"Estimated statewide reduction in children experiencing food insecurity (illustrative): **{delta:,}**")
else:
    st.caption("Add MMG Children overlay for an estimated statewide change in children experiencing FI.")
