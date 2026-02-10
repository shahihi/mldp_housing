import os
import joblib
import pandas as pd
import streamlit as st

# Page config 
st.set_page_config(page_title="Housing Resale Price Estimator", page_icon="🏠", layout="centered")

# Colors used
TEXT = "#422618"
DARK = "#693713DC"
MID = "#a47340"
LIGHT = "#cfb69f"

# styles
st.markdown(
    f"""
<style>
.stApp {{ background: {LIGHT}; }}

html, body, [class*="css"], .stMarkdown, .stText, .stCaption, .stSubheader, .stTitle {{
  color: {TEXT} !important;
}}

h1, h2, h3 {{ color: {DARK} !important; }}

.card {{
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(105,55,19,0.20);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 8px 26px rgba(105,55,19,0.12);
  margin-bottom: 14px;
}}

.stButton > button {{
  background: {DARK};
  color: {LIGHT};
  border: 0;
  border-radius: 14px;
  padding: 12px 14px;
  font-weight: 650;
  width: 100%;
}}
.stButton > button:hover {{
  background: {MID};
  color: {LIGHT};
}}

section[data-testid="stSidebar"] {{
  background: rgba(255,255,255,0.35);
  border-right: 1px solid rgba(105,55,19,0.18);
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

div[data-testid="stMetric"] {{
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(105,55,19,0.18);
  border-radius: 18px;
  padding: 12px 14px;
}}
div[data-testid="stMetric"] label {{
  color: {DARK} !important;
  font-weight: 650;
}}
</style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_lr_fe.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

#  Header 
st.markdown(
    f"""
<div class="card">
  <h2 style="margin:0;">Housing Resale Price Estimator</h2>
  <p style="margin:6px 0 0 0; color:{TEXT};">
    Enter property details below to estimate its resale price.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# Helpers
def parse_int(label: str, raw: str, *, min_v: int, max_v: int) -> tuple[int | None, str | None]:
    if raw is None or raw.strip() == "":
        return None, f"{label} cannot be empty."
    try:
        v = int(float(raw))  
    except ValueError:
        return None, f"{label} must be a whole number."
    if v < min_v or v > max_v:
        return None, f"{label} must be between {min_v} and {max_v}."
    return v, None

def parse_float(label: str, raw: str, *, min_v: float, max_v: float | None = None) -> tuple[float | None, str | None]:
    if raw is None or raw.strip() == "":
        return None, f"{label} cannot be empty."
    try:
        v = float(raw)
    except ValueError:
        return None, f"{label} must be a number."
    if v < min_v:
        return None, f"{label} must be at least {min_v}."
    if max_v is not None and v > max_v:
        return None, f"{label} must be at most {max_v}."
    return v, None

#  Inputs 
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Size & Layout")

    bedroom_options = list(range(1, 21))  # 1..20
    bedroom_count = st.selectbox(
        "Bedroom Count",
        options=bedroom_options,
        index=None,                 # no selection on first load
        placeholder="Select bedrooms..."
    )

    area_raw = st.text_input(
        "Area of the floor (sqm)",
        value="",
        placeholder="e.g. 85.5"
    )

    floor_raw = st.text_input(
        "Floor Level",
        value="",
        placeholder="1 to 50"
    )

with col2:
    st.markdown("### Location & Building")

    center_raw = st.text_input(
        "Distance to City Center (m)",
        value="",
        placeholder="e.g. 1200"
    )

    metro_raw = st.text_input(
        "Distance to Metro (m)",
        value="",
        placeholder="e.g. 250"
    )

    age_raw = st.text_input(
        "Property Age (years)",
        value="",
        placeholder="0 to 1000"
    )

predict = st.button("Predict Price", use_container_width=True)

if predict:
    errors = []

    # Bedroom dropdown validation
    if bedroom_count is None:
        errors.append("Bedroom Count cannot be empty.")

    area, err = parse_float("Net square meters", area_raw, min_v=1.0)
    if err: errors.append(err)

    floor, err = parse_int("Floor Level", floor_raw, min_v=1, max_v=50)
    if err: errors.append(err)

    center_distance, err = parse_float("Distance to City Center (m)", center_raw, min_v=0.0)
    if err: errors.append(err)

    metro_distance, err = parse_float("Distance to Metro (m)", metro_raw, min_v=0.0)
    if err: errors.append(err)

    age, err = parse_int("Property Age (years)", age_raw, min_v=0, max_v=1000)
    if err: errors.append(err)

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    new_property = pd.DataFrame([{
        "bedroom_count": int(bedroom_count),
        "net_sqm": float(area),
        "center_distance": float(center_distance),
        "metro_distance": float(metro_distance),
        "floor": int(floor),
        "age": int(age),
    }])

    new_property["sqm_per_bedroom"] = new_property["net_sqm"] / new_property["bedroom_count"]

    # Align columns
    new_property = new_property.reindex(columns=features)

    pred = model.predict(new_property)[0]
    st.metric("Predicted House Price", f"{pred:,.2f}")
    st.caption("Note: This is an estimate based on historical patterns; actual market prices may differ.")
