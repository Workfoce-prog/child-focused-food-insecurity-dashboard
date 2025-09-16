# app.py — Kids Food Insecurity Ops Dashboard (MN, children focus)
# Safe loader with in-memory fallback (no more FileNotFoundError)

import os, sys, io, json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ---------- Page ----------
st.set_page_config(page_title="Kids Food Insecurity Ops — Minnesota", layout="wide")
st.title("Kids Food Insecurity Ops — Minnesota (Children 0–17)")
st.caption(f"Python: {sys.version.split()[0]} • Streamlit: {st.__version__}")

# ---------- Bootstrap paths & clear stale cache once ----------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)
try:
    # clears old cached versions of the loader that expected a file
    st.cache_data.clear()
except Exception:
    pass

with st.expander("Debug (env & files)"):
    try:
        st.write("CWD:", os.getcwd())
        st.write("App dir:", str(BASE))
        st.write("Top-level files:", sorted(os.listdir(".")))
        st.write("data/ exists:", DATA.exists())
        if DATA.exists():
            st.write("data/ →", sorted(os.listdir(DATA)))
    except Exception as e:
        st.error(f"Debug listing failed: {e}")

# ---------- Sidebar ----------
st.sidebar.header("Population")
st.sidebar.write("Segment: **Children (0–17)**")

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

st.sidebar.header("Policy Levers — Δ risk reduction %")
sim_snap   = st.sidebar.slider("SNAP (HHs w/ kids)", 0, 50, 8)
sim_school = st.sidebar.slider("School Meals (NSLP/SSO/SFSP)", 0, 50, 10)
sim_summer = st.sidebar.slider("Summer EBT / Sites", 0, 50, 6)
policy_total_reduction = (sim_snap + sim_school + sim_summer) / 100.0

st.sidebar.header("Overlays & Uploads")
use_mmg  = st.sidebar.checkbox("Overlay: Map the Meal Gap — Children", True)
use_lila = st.sidebar.checkbox("Overlay: USDA LI/LA (Food Access) proxy", True)
mmg_upload  = st.sidebar.file_uploader("Upload mmg_children.csv (optional)", type=["csv"])
lila_upload = st.sidebar.file_uploader("Upload li_la.csv (optional)", type=["csv"])

# ---------- SAFE LOADER with embedded sample fallback ----------
EMBEDDED_SAMPLE_CSV = """county,fips,lat,lon,population,child_population,child_household_risk,child_access_score,child_experience_score,child_resilience_score,child_policy_buffer,mmg_child_rate,mmg_child_count,mmg_meal_cost,li_la_share
Hennepin,27053,45.003,-93.265,1284565,325000,67,57,62,52,47,11.8,38350,3.60,0.21
Ramsey,27123,44.953,-93.090,552352,120000,72,54,64,49,47,12.6,15120,3.50,0.24
Dakota,27037,44.731,-93.089,444567,115000,52,60,47,57,52,8.9,10235,3.45,0.14
Beltrami,27007,47.525,-94.885,47371,9000,62,42,57,46,36,17.5,1575,3.40,0.39
Mahnomen,27087,47.318,-95.969,5590,1300,64,40,60,44,35,18.6,242,3.50,0.45
Carver,27019,44.820,-93.784,110621,32000,37,67,32,62,57,5.8,1856,3.55,0.10
Scott,27139,44.660,-93.537,154237,42000,40,65,34,60,56,6.2,2604,3.55,0.12
St Louis,27137,47.520,-92.362,200419,42000,50,48,46,53,41,12.1,5082,3.45,0.33
"""

@st.cache_data(show_spinner=False)
def load_base_df() -> pd.DataFrame:
    target = DATA / "sample_child_dashboard.csv"
    # 1) Try reading from disk first
    if target.exists():
        return pd.read_csv(target)

    # 2) Fall back to embedded CSV (always available)
    df = pd.read_csv(io.StringIO(EMBEDDED_SAMPLE_CSV))

    # 3) Best-effort: persist to disk for future runs (ignore failures)
    try:
        df.to_csv(target, index=False)
    except Exception:
        pass

    st.info(f"Loaded embedded sample and seeded: data/{target.name}")
    return df

# ---------- 1) Load Data ----------
st.subheader("1) Load Base Data (Children)")
if mmg_upload is not None:
    mmg_df = pd.read_csv(mmg_upload)
else:
    mmg_df = None

if lila_upload is not None:
    lila_df = pd.read_csv(lila_upload)
else:
    lila_df = None

df = load_base_df()

# ---------- 2) Overlays ----------
st.subheader("2) Overlays")
notes = []

if use_mmg:
    if mmg_df is not None:
        mmg = mmg_df.copy()
    else:
        mmg = df[["fips","mmg_child_rate","mmg_child_count","mmg_meal_cost"]].copy()
    mmg_cols = [c for c in ["mmg_child_rate","mmg_child_count","mmg_meal_cost"] if c in mmg.columns]
    if "fips" in mmg.columns and mmg_cols:
        df = df.drop(columns=[c for c in mmg_cols if c in df.columns], errors="ignore") \
               .merge(mmg[["fips"]+mmg_cols], on="fips", how="left")
        notes.append("MMG Children overlay applied.")
    else:
        st.warning("MMG overlay skipped — need fips + mmg_child_rate/count.")

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

# ---------- 3) Child Composite & RAG ----------
st.subheader("3) Child Composite (CFIRI-Child) & RAG")

required_cols = [
    "county","fips","lat","lon","child_population",
    "child_household_risk","child_access_score","child_experience_score",
    "child_resilience_score","child_policy_buffer"
]
miss = [c for c in required_cols if c not in df.columns]
if miss:
    st.error(f"Missing columns: {miss}")
    st.stop()

for c in ["child_population","child_household_risk","child_access_score","child_experience_score","child_resilience_score","child_policy_buffer"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["child_household_risk","child_access_score","child_experience_score","child_resilience_score","child_policy_buffer"])
for c in ["child_household_risk","child_access_score","child_experience_score","child_resilience_score","child_policy_buffer"]:
    df[c] = df[c].clip(0,100)

total_w = max(1, w1 + w2 + w3 + w4 + w5)
weights = dict(
    child_household_risk   = w1/total_w,
    child_access_score     = w2/total_w,
    child_experience_score = w3/total_w,
    child_resilience_score = w4/total_w,
    child_policy_buffer    = w5/total_w
)

def compute_child_cfiri(r):
    raw = (
        weights["child_household_risk"]   * r["child_household_risk"] +
        weights["child_access_score"]     * r["child_access_score"] +
        weights["child_experience_score"] * r["child_experience_score"] +
        weights["child_resilience_score"] * r["child_resilience_score"] +
        weights["child_policy_buffer"]    * r["child_policy_buffer"]
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
m["size"]  = 1000 * (m["CFIRI_CHILD"].clip(0,100)/100.0 + 0.2)
color_map  = {"Green":[0,153,0], "Amber":[255,191,0], "Red":[220,53,69], "Critical":[128,0,0]}
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

# ---------- 6) Illustrative Impact Lens ----------
st.subheader("6) Illustrative Impact Lens (Children)")
st.caption("Applies the global policy reduction to composite for a quick what-if. For policy-grade estimates, plug in SNAP/WIC/school-meal microsimulation outputs.")

if {"mmg_child_count","mmg_child_rate"}.issubset(df.columns):
    baseline = df[["county","fips","child_population","mmg_child_rate","mmg_child_count"]].copy()
    baseline["expected_child_fi_count"] = baseline["child_population"] * (baseline["mmg_child_rate"]/100.0)
    baseline["expected_child_fi_count_after"] = baseline["expected_child_fi_count"] * (1.0 - policy_total_reduction)
    delta = int(baseline["expected_child_fi_count"].sum() - baseline["expected_child_fi_count_after"].sum())
    st.write(f"Estimated statewide reduction in children experiencing food insecurity (illustrative): **{delta:,}**")
else:
    st.caption("Add MMG Children overlay to view an estimated statewide change in child FI counts.")
# ---------- 7) Action Recommendations ----------
st.subheader("7) Action Recommendations (auto-generated)")

# Controls
scope = st.radio("Target scope", ["All counties", "Only Red/Critical"], horizontal=True, index=1)
top_n = st.slider("Top N actions per county", 1, 10, 5)

# Optional category filter (you can uncheck to hide a bucket)
categories_all = [
    "School Nutrition", "SNAP / WIC", "Access & Logistics",
    "CBO / Food Shelves", "Screening & Referrals", "Cost & Procurement"
]
categories_on = st.multiselect("Include categories", categories_all, default=categories_all)

# Helper to safely fetch optional fields
def _get(r, col, default=np.nan):
    try:
        return r[col]
    except KeyError:
        return default

def recommend_actions_for_row(r):
    """
    Returns list of (category, action, priority_score, reason) for a single county row.
    Priority is a simple severity/scale blend:
      base + RAG bump + CFIRI scaling + (optional) MMG rate bump + access bump
    """
    recs = []

    def add(cat, action, reason, base=1.0):
        if cat not in categories_on:
            return
        score = base
        # severity bump
        rag = r["RAG_CHILD"]
        if rag == "Critical": score += 2.5
        elif rag == "Red":    score += 1.5
        # overall risk scaling
        score += float(r["CFIRI_CHILD"]) / 100.0  # +0..+1
        # scale-of-need bump (if MMG rate present)
        mmg_rate = _get(r, "mmg_child_rate", np.nan)
        if pd.notna(mmg_rate): score += float(mmg_rate) / 50.0  # up to +2
        # access penalty awareness
        if float(r["child_access_score"]) > 60: score += 0.5
        recs.append((cat, action, round(score, 2), reason))

    # --- RULES ---

    # 1) School Nutrition — use when policy buffer shortfall is high or county is severe
    if (float(r["child_policy_buffer"]) >= 55) or (r["RAG_CHILD"] in ("Red","Critical")):
        add("School Nutrition",
            "Accelerate CEP/Universal Meals + Breakfast After the Bell",
            "High policy buffer shortfall and/or severe RAG; school meals coverage can reduce child FI quickly.", base=2.2)

    # 2) SNAP / WIC — when household risk or MMG rate is high
    if (float(r["child_household_risk"]) >= 60) or (pd.notna(_get(r,"mmg_child_rate")) and float(_get(r,"mmg_child_rate")) >= 12):
        add("SNAP / WIC",
            "Targeted SNAP outreach + WIC enrollment & recert simplifications",
            "Elevated household risk and/or child FI rate; increase take-up and retention.", base=2.0)

    # 3) Access & Logistics — when access score or LI/LA share is high
    li_la = _get(r, "li_la_share", np.nan)   # 0–1 proportion if present
    if (float(r["child_access_score"]) >= 60) or (pd.notna(li_la) and float(li_la) >= 0.25):
        add("Access & Logistics",
            "Add summer/after-school meal sites + mobile markets; adjust transit/last-mile",
            "Access deficits (distance/rurality/LI-LA) suggest siting/logistics fixes.", base=1.8)

    # 4) CBO / Food Shelves — when resilience shortfall is high
    if float(r["child_resilience_score"]) >= 55:
        add("CBO / Food Shelves",
            "Capacity grants (cold storage, staffing) + cross-county logistics & volunteer drivers",
            "Resilience shortfall indicates network capacity gaps.", base=1.6)

    # 5) Screening & Referrals — when experiential signal is high
    if float(r["child_experience_score"]) >= 60:
        add("Screening & Referrals",
            "School-based screening (Hunger Vital Sign) + backpack/weekend meals + 211/benefits navigators",
            "High experiential indicators call for direct, school-centered interventions.", base=1.5)

    # 6) Cost & Procurement — when local meal cost is high
    mmg_cost = _get(r, "mmg_meal_cost", np.nan)
    if pd.notna(mmg_cost) and float(mmg_cost) >= 3.60:
        add("Cost & Procurement",
            "Boost per-meal reimbursement; cooperative bulk purchasing & food-hub partnerships",
            "Elevated local meal costs; pooling & co-ops can stretch dollars.", base=1.2)

    # 7) Scale-up prompt — big kid population + severe RAG
    if (int(r["child_population"]) >= 30000) and (r["RAG_CHILD"] in ("Red","Critical")):
        add("CBO / Food Shelves",
            "Stand up a county coordination cell to scale hubs & cross-program referrals",
            "Large child population with severe risk warrants centrally coordinated scale-up.", base=1.4)

    return recs

# Choose target set
target_df = df.copy()
if scope == "Only Red/Critical":
    target_df = target_df[target_df["RAG_CHILD"].isin(["Red","Critical"])].copy()

# Build plan table (one row per county-action)
plan_rows = []
for _, row in target_df.iterrows():
    recs = sorted(recommend_actions_for_row(row), key=lambda x: -x[2])[:top_n]
    for cat, action, score, reason in recs:
        plan_rows.append({
            "county": row["county"],
            "RAG": row["RAG_CHILD"],
            "CFIRI_CHILD": round(float(row["CFIRI_CHILD"]), 1),
            "child_population": int(row["child_population"]),
            "category": cat,
            "action": action,
            "priority": score,
            "reason": reason,
        })

if len(plan_rows) == 0:
    st.info("No recommendations under the current filters.")
else:
    plan = pd.DataFrame(plan_rows).sort_values(["priority","CFIRI_CHILD","child_population"], ascending=[False, False, False])
    st.markdown("**Prioritized county-action plan**")
    st.dataframe(plan, use_container_width=True, hide_index=True)

    # Summary by action across state
    st.markdown("**Roll-up by action (across targeted counties)**")
    summary = (plan
               .groupby(["category","action"], as_index=False)
               .agg(counties=("county","nunique"),
                    avg_priority=("priority","mean"),
                    total_children=("child_population","sum"))
               .sort_values(["avg_priority","total_children"], ascending=[False, False]))
    summary["avg_priority"] = summary["avg_priority"].round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Downloads
    st.download_button("Download county-action plan (CSV)", plan.to_csv(index=False).encode(), "child_action_plan_by_county.csv")
    st.download_button("Download action roll-up (CSV)", summary.to_csv(index=False).encode(), "child_action_rollup.csv")

    # Small guidance box
    st.caption(
        "Priority is a heuristic that blends severity (RAG), overall risk (CFIRI), scale of need (MMG child rate if available), "
        "and access deficits. Tune weights upstream and narrow scope to focus operations."
    )

