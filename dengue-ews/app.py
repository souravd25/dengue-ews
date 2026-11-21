"""Dengue early-warning dashboard.  Run: streamlit run app.py"""
import json, joblib, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from src.features import load_raw, build_features, add_targets, FEATURES, driver_group

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")
TIERS = ["Low", "Medium", "High", "Critical"]
COLORS = {"Low": "#2e9e5b", "Medium": "#f2c94c", "High": "#f2994a", "Critical": "#d64545"}


@st.cache_resource
def load_models():
    return joblib.load("models/models.joblib"), json.load(open("models/meta.json"))


@st.cache_data
def load_data():
    return add_targets(build_features(load_raw()))


models, meta = load_models()
df = load_data()
th = meta["tier_thresholds"]
cuts = [th["medium"] + 1e-9, th["high"] + 1e-9, th["critical"] + 1e-9]
tier_of = lambda x: TIERS[int(np.digitize(x, cuts))]

st.title("🦟 Urban Dengue Early-Warning System")
st.caption("Weather + multi-week rainfall lags → ward-level risk 2–4 weeks ahead (Dhaka, Chattogram, Gazipur, Narayanganj)")

with st.sidebar:
    st.header("Inputs")
    city = st.selectbox("City", sorted(df["city"].unique()), index=sorted(df["city"].unique()).index("Chattogram"))
    wards = sorted(df[df["city"] == city]["ward_id"].unique())
    ward = st.selectbox("Ward", wards)
    h = st.radio("Forecast window (weeks ahead)", meta["horizons"], index=1, horizontal=True)
    valid = df.dropna(subset=FEATURES)
    dates = sorted(valid["date"].dt.date.unique())
    origin = st.select_slider("Forecast origin (last observed week)", options=dates, value=pd.Timestamp("2025-07-14").date())
    st.caption(f"Models trained on data ≤ {meta['split']['val_end']}; weeks after that are true out-of-sample.")

origin = pd.Timestamp(origin)
snap = valid[valid["date"] == origin].copy()  # all wards at the origin week
mod = models[h]
X = snap[FEATURES]
snap["pred"] = mod["point"].predict(X)          # ONE vectorised call for every ward
snap["lo"] = np.minimum(mod["lo"].predict(X), snap["pred"])
snap["hi"] = np.maximum(mod["hi"].predict(X), snap["pred"])
snap["tier"] = snap["pred"].map(tier_of)
target_date = origin + pd.Timedelta(weeks=h)
row = snap[snap["ward_id"] == ward].iloc[0]

tab1, tab2, tab3 = st.tabs(["Ward forecast", "City ward ranking", "Model validation"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Risk tier**<br><span style='font-size:2.2rem;color:{COLORS[row.tier]}'>● {row.tier}</span>",
                unsafe_allow_html=True)
    c2.metric(f"Predicted cases (week of {target_date.date()})", f"{row.pred:.1f}")
    c3.metric("80% interval", f"{row.lo:.1f} – {row.hi:.1f}")
    actual = df[(df["ward_id"] == ward) & (df["date"] == target_date)]["dengue_cases"]
    c4.metric("Actual cases (if known)", f"{int(actual.iloc[0])}" if len(actual) else "—")

    left, right = st.columns(2)
    # Per-prediction drivers: LightGBM exact tree contributions (SHAP-style), grouped
    contrib = mod["point"].predict(X[snap["ward_id"] == ward], pred_contrib=True)[0][:-1]
    cdf = pd.DataFrame({"feature": FEATURES, "c": contrib})
    cdf["driver"] = cdf["feature"].map(driver_group)
    g = cdf.groupby("driver")["c"].sum().sort_values()
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                           marker_color=["#d64545" if v > 0 else "#2e9e5b" for v in g.values]))
    fig.update_layout(title="Main drivers (log-scale push: red ↑ raises risk, green ↓ lowers)", height=340,
                      margin=dict(l=0, r=0, t=40, b=0))
    left.plotly_chart(fig, width="stretch")

    hist = df[(df["ward_id"] == ward) & (df["date"] <= origin) & (df["date"] > origin - pd.Timedelta(weeks=12))]
    fig2 = go.Figure(go.Bar(x=hist["date"], y=hist["rainfall_mm"], name="Rainfall (mm)", marker_color="#4f8ef7"))
    fig2.add_vline(x=origin - pd.Timedelta(weeks=3), line_dash="dot",
                   annotation_text="~3-wk breeding lag", annotation_position="top left")
    fig2.update_layout(title="Recent weekly rainfall (lagged inputs)", height=340, margin=dict(l=0, r=0, t=40, b=0))
    right.plotly_chart(fig2, width="stretch")

    lag_cols = [f"rain_lag_{k}w" for k in range(1, 9)]
    st.caption("Rainfall lags fed to the model (mm): " +
               ", ".join(f"{k}w={row[f'rain_lag_{k}w']:.0f}" for k in range(1, 9)))

with tab2:
    st.subheader(f"{city}: wards ranked by predicted cases, {h} weeks after {origin.date()}")
    cs = snap[snap["city"] == city].sort_values("pred", ascending=False)
    fig3 = px.bar(cs, x="ward_id", y="pred", color="tier", color_discrete_map=COLORS,
                  category_orders={"tier": TIERS}, error_y=cs["hi"] - cs["pred"],
                  error_y_minus=cs["pred"] - cs["lo"], labels={"pred": "Predicted cases", "ward_id": "Ward"})
    st.plotly_chart(fig3, width="stretch")
    st.dataframe(cs[["ward_id", "tier", "pred", "lo", "hi", "water_logging_risk_index", "sanitation_score",
                     "population_density_sqkm"]].round(2), width="stretch", hide_index=True)
    # ward x week risk heatmap (last 26 origins) for the city
    wk = sorted(valid[valid["date"] <= origin]["date"].unique())[-26:]
    hm = valid[(valid["city"] == city) & (valid["date"].isin(wk))].copy()
    hm["pred"] = mod["point"].predict(hm[FEATURES])
    pv = hm.pivot(index="ward_id", columns="date", values="pred")
    fig4 = px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="Pred. cases"))
    fig4.update_layout(title="Ward × week predicted-risk heatmap (last 26 origins)", height=380)
    st.plotly_chart(fig4, width="stretch")

with tab3:
    st.subheader("Chronological validation (train ≤2022 · validation 2023 · test 2024–25)")
    mt = pd.read_csv("models/metrics.csv")
    st.dataframe(mt.round(3), width="stretch", hide_index=True)
    st.caption("Baselines: global train mean and per-ward seasonal (week-of-year) mean. Lower MAE/RMSE is better.")
    tw = df[(df["ward_id"] == ward) & (df["date"] > meta["split"]["val_end"])].dropna(subset=FEATURES + [f"target_{h}w"])
    tw = tw.assign(pred=mod["point"].predict(tw[FEATURES]),
                   lo=mod["lo"].predict(tw[FEATURES]), hi=mod["hi"].predict(tw[FEATURES]))
    tw["tdate"] = tw["date"] + pd.Timedelta(weeks=h)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw["hi"], line=dict(width=0), showlegend=False))
    fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw["lo"], fill="tonexty", line=dict(width=0),
                              fillcolor="rgba(79,142,247,0.2)", name="80% interval"))
    fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw[f"target_{h}w"], name="Actual", line=dict(color="#333")))
    fig5.add_trace(go.Scatter(x=tw["tdate"], y=tw["pred"], name="Predicted", line=dict(color="#d64545")))
    fig5.update_layout(title=f"{ward}: actual vs predicted, out-of-sample ({h}-week horizon)", height=380)
    st.plotly_chart(fig5, width="stretch")
