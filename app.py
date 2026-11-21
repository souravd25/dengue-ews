"""Dengue early-warning dashboard.  Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import load_raw, build_features, add_targets, FEATURES, driver_group

ROOT = Path(__file__).parent
st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")

TIERS = ["Low", "Medium", "High", "Critical"]
COLORS = {"Low": "#2e9e5b", "Medium": "#f2c94c", "High": "#f2994a", "Critical": "#d64545"}
ADVICE = {
    "Low": "Routine surveillance. No extra action needed.",
    "Medium": "Watch closely. Check drains and stagnant water in the ward.",
    "High": "Start source-reduction drives and alert local clinics.",
    "Critical": "Prioritise this ward: fogging, larval control, hospital readiness.",
}
DEFAULT_ORIGIN = pd.Timestamp("2025-07-14")

st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .6rem .9rem; border-radius: 10px;}
    .tier-card {padding: .6rem .9rem; border-radius: 10px; background: rgba(128,128,128,.08);}
    </style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    with open(ROOT / "models" / "meta.json") as f:
        meta = json.load(f)
    return joblib.load(ROOT / "models" / "models.joblib"), meta


@st.cache_data
def load_data():
    return add_targets(build_features(load_raw()))


@st.cache_data
def load_metrics():
    return pd.read_csv(ROOT / "models" / "metrics.csv")


@st.cache_data(show_spinner=False)
def heatmap_frame(city, origin, h, _mod):
    """Predicted cases for every ward of a city over the last 26 origins up to `origin`."""
    wk = sorted(valid[valid["date"] <= origin]["date"].unique())[-26:]
    hm = valid[(valid["city"] == city) & (valid["date"].isin(wk))].copy()
    hm["pred"] = _mod["point"].predict(hm[FEATURES])
    return hm.pivot(index="ward_id", columns="date", values="pred")


try:
    models, meta = load_models()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run `python -m src.train` from the project root first.")
    st.stop()

valid = df.dropna(subset=FEATURES)
th = meta["tier_thresholds"]
cuts = [th["medium"] + 1e-9, th["high"] + 1e-9, th["critical"] + 1e-9]
tier_of = lambda x: TIERS[int(np.digitize(x, cuts))]
val_end = pd.Timestamp(meta["split"]["val_end"])

# ---------------------------------------------------------------- header
st.title("🦟 Urban Dengue Early-Warning System")
st.caption("Weather + multi-week rainfall lags → ward-level risk 2–4 weeks ahead "
           "(Dhaka, Chattogram, Gazipur, Narayanganj)")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Inputs")
    cities = sorted(df["city"].unique())
    city = st.selectbox("City", cities, index=cities.index("Chattogram") if "Chattogram" in cities else 0)
    wards = sorted(df[df["city"] == city]["ward_id"].unique())
    ward = st.selectbox("Ward", wards)
    h = st.radio("Forecast window (weeks ahead)", meta["horizons"],
                 index=min(1, len(meta["horizons"]) - 1), horizontal=True)

    dates = sorted(valid["date"].dt.date.unique())
    earlier = [d for d in dates if pd.Timestamp(d) <= DEFAULT_ORIGIN]
    default = earlier[-1] if earlier else dates[-1]   # never crashes if the date is missing
    origin = st.select_slider("Forecast origin (last observed week)", options=dates, value=default)

    st.divider()
    st.markdown("**Risk tiers (predicted weekly cases)**")
    st.markdown(
        f"<span style='color:{COLORS['Low']}'>● Low</span> ≤ {th['medium']:.0f}<br>"
        f"<span style='color:{COLORS['Medium']}'>● Medium</span> {th['medium']:.0f} – {th['high']:.0f}<br>"
        f"<span style='color:{COLORS['High']}'>● High</span> {th['high']:.0f} – {th['critical']:.0f}<br>"
        f"<span style='color:{COLORS['Critical']}'>● Critical</span> &gt; {th['critical']:.0f}",
        unsafe_allow_html=True)
    st.caption(f"Models trained on data ≤ {meta['split']['val_end']}; later weeks are true out-of-sample.")

# ---------------------------------------------------------------- predictions
origin = pd.Timestamp(origin)
snap = valid[valid["date"] == origin].copy()          # all wards at the origin week
if snap.empty:
    st.warning("No data for this origin week. Pick another date.")
    st.stop()

mod = models[h]
X = snap[FEATURES]
snap["pred"] = mod["point"].predict(X)                # ONE vectorised call for every ward
snap["lo"] = np.minimum(mod["lo"].predict(X), snap["pred"])
snap["hi"] = np.maximum(mod["hi"].predict(X), snap["pred"])
snap["tier"] = snap["pred"].map(tier_of)
target_date = origin + pd.Timedelta(weeks=h)

ward_rows = snap[snap["ward_id"] == ward]
if ward_rows.empty:
    st.warning(f"Ward {ward} has no data for {origin.date()}. Pick another ward or date.")
    st.stop()
row = ward_rows.iloc[0]

if origin > val_end:
    st.success(f"Out-of-sample forecast: origin {origin.date()} is after the training period "
               f"(target week {target_date.date()}).")
else:
    st.info(f"Origin {origin.date()} is inside the training period, so accuracy here is optimistic. "
            "Choose an origin after " + str(val_end.date()) + " for an honest test.")

tab1, tab2, tab3 = st.tabs(["Ward forecast", "City ward ranking", "Model validation"])

# ---------------------------------------------------------------- tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='tier-card'><b>Risk tier</b><br>"
                f"<span style='font-size:2rem;color:{COLORS[row['tier']]}'>● {row['tier']}</span></div>",
                unsafe_allow_html=True)
    c2.metric(f"Predicted cases (week of {target_date.date()})", f"{row['pred']:.1f}")
    c3.metric("80% interval", f"{row['lo']:.1f} – {row['hi']:.1f}")
    actual = df[(df["ward_id"] == ward) & (df["date"] == target_date)]["dengue_cases"]
    has_actual = len(actual) > 0 and pd.notna(actual.iloc[0])
    c4.metric("Actual cases (if known)", f"{int(actual.iloc[0])}" if has_actual else "—")
    st.markdown(f"**Suggested action:** {ADVICE[row['tier']]}")

    left, right = st.columns(2)
    # Per-prediction drivers: LightGBM exact tree contributions (SHAP-style), grouped
    contrib = mod["point"].predict(X[snap["ward_id"] == ward], pred_contrib=True)[0][:-1]
    cdf = pd.DataFrame({"feature": FEATURES, "c": contrib})
    cdf["driver"] = cdf["feature"].map(driver_group)
    g = cdf.groupby("driver")["c"].sum().sort_values()
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                           marker_color=["#d64545" if v > 0 else "#2e9e5b" for v in g.values]))
    fig.update_layout(title="Main drivers (log-scale push: red ↑ raises risk, green ↓ lowers)",
                      height=340, margin=dict(l=0, r=0, t=40, b=0))
    left.plotly_chart(fig, width="stretch")

    hist = df[(df["ward_id"] == ward) & (df["date"] <= origin) & (df["date"] > origin - pd.Timedelta(weeks=12))]
    fig2 = go.Figure(go.Bar(x=hist["date"], y=hist["rainfall_mm"], name="Rainfall (mm)", marker_color="#4f8ef7"))
    # add_vline(annotation_text=...) crashes on datetime axes in several plotly versions,
    # so draw the line and the label separately.
    lag_x = (origin - pd.Timedelta(weeks=3)).to_pydatetime()
    fig2.add_shape(type="line", x0=lag_x, x1=lag_x, y0=0, y1=1, xref="x", yref="paper",
                   line=dict(dash="dot", color="#888"))
    fig2.add_annotation(x=lag_x, y=1, xref="x", yref="paper", text="~3-wk breeding lag",
                        showarrow=False, yanchor="bottom", xanchor="right")
    fig2.update_layout(title="Recent weekly rainfall (lagged inputs)", height=340,
                       margin=dict(l=0, r=0, t=50, b=0))
    right.plotly_chart(fig2, width="stretch")

    st.caption("Rainfall lags fed to the model (mm): " +
               ", ".join(f"{k}w={row[f'rain_lag_{k}w']:.0f}" for k in range(1, 9)))

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader(f"{city}: wards ranked by predicted cases, {h} weeks after {origin.date()}")
    cs = snap[snap["city"] == city].sort_values("pred", ascending=False)

    counts = cs["tier"].value_counts()
    kc = st.columns(4)
    for col, t in zip(kc, TIERS[::-1]):
        col.metric(f"{t} wards", int(counts.get(t, 0)))

    fig3 = px.bar(cs, x="ward_id", y="pred", color="tier", color_discrete_map=COLORS,
                  category_orders={"tier": TIERS, "ward_id": cs["ward_id"].tolist()},  # keep ranking order
                  error_y=cs["hi"] - cs["pred"], error_y_minus=cs["pred"] - cs["lo"],
                  labels={"pred": "Predicted cases", "ward_id": "Ward", "tier": "Tier"})
    fig3.update_layout(xaxis_type="category", height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig3, width="stretch")

    table = cs[["ward_id", "tier", "pred", "lo", "hi", "water_logging_risk_index", "sanitation_score",
                "population_density_sqkm"]].round(2)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("⬇ Download ranking (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"ranking_{city}_{origin.date()}_{h}w.csv", mime="text/csv")

    pv = heatmap_frame(city, origin, h, mod)
    fig4 = px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="Pred. cases"))
    fig4.update_layout(title="Ward × week predicted-risk heatmap (last 26 origins)", height=380)
    st.plotly_chart(fig4, width="stretch")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Chronological validation (train ≤2022 · validation 2023 · test 2024–25)")
    try:
        st.dataframe(load_metrics().round(3), width="stretch", hide_index=True)
    except FileNotFoundError:
        st.warning("models/metrics.csv not found. Run `python -m src.train`.")
    st.caption("Baselines: global train mean and per-ward seasonal (week-of-year) mean. "
               "Lower MAE/RMSE is better.")

    tcol = f"target_{h}w"
    tw = df[(df["ward_id"] == ward) & (df["date"] > val_end)].dropna(subset=FEATURES + [tcol])
    if tw.empty:
        st.info("No out-of-sample weeks available for this ward.")
    else:
        p = mod["point"].predict(tw[FEATURES])
        lo, hi = mod["lo"].predict(tw[FEATURES]), mod["hi"].predict(tw[FEATURES])
        tw = tw.assign(pred=p, lo=np.minimum(np.minimum(lo, hi), p), hi=np.maximum(np.maximum(lo, hi), p))
        tw["tdate"] = tw["date"] + pd.Timedelta(weeks=h)

        m1, m2 = st.columns(2)
        m1.metric(f"{ward} MAE (out-of-sample)", f"{(tw['pred'] - tw[tcol]).abs().mean():.2f}")
        m2.metric("80% interval coverage", f"{((tw[tcol] >= tw['lo']) & (tw[tcol] <= tw['hi'])).mean():.0%}")

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw["hi"], line=dict(width=0), showlegend=False))
        fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw["lo"], fill="tonexty", line=dict(width=0),
                                  fillcolor="rgba(79,142,247,0.2)", name="80% interval"))
        fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw[tcol], name="Actual", line=dict(color="#888")))
        fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw["pred"], name="Predicted", line=dict(color="#d64545")))
        fig5.update_layout(title=f"{ward}: actual vs predicted, out-of-sample ({h}-week horizon)", height=380)
        st.plotly_chart(fig5, width="stretch")
