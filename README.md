# Kids Food Insecurity Ops Dashboard (MN)

A Streamlit app centered on **children (0–17)**. It builds a child CFIRI (composite risk) and can overlay:
- **Map the Meal Gap — Children** (county child FI rate/count, meal cost)
- **USDA Food Access (LI/LA)** proxy

## Files
- `app.py` — the app (no matplotlib)
- `requirements.txt` — minimal deps
- `runtime.txt` — pin Python 3.12 (Streamlit Cloud)
- `data/sample_child_dashboard.csv` — bundled sample with child columns

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)
- Main file path: `app.py`
- Ensure `runtime.txt` contains `3.12`
- Clear cache & restart

## Data schema (sample)
`county,fips,lat,lon,population,child_population,
child_household_risk,child_access_score,child_experience_score,child_resilience_score,child_policy_buffer,
mmg_child_rate,mmg_child_count,mmg_meal_cost,li_la_share`

You can upload optional overlays:
- `mmg_children.csv` columns: `fips,mmg_child_rate,mmg_child_count,mmg_meal_cost`
- `li_la.csv` columns: `fips,li_la_share`

> Impact lens is illustrative. For policy-grade estimates, plug in outputs from SNAP/WIC/school-meal microsimulations.
