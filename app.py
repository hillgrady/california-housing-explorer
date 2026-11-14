"""California Housing Explorer: an interactive dashboard and price predictor.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import CATEGORIES, LABELS, VALUE_CAP, filter_data, load_data, summary
from src.model import build_row, train

st.set_page_config(page_title="California Housing Explorer", page_icon="🏠", layout="wide")

COLORS = dict(zip(CATEGORIES, px.colors.qualitative.Set2))
MAP_POINTS = 6000  # cap on map markers so the page stays fast on any filter
CITIES = {
    "Custom location": None,
    "San Francisco": (37.77, -122.42, "NEAR BAY"),
    "Oakland": (37.80, -122.27, "NEAR BAY"),
    "San Jose": (37.34, -121.89, "<1H OCEAN"),
    "Los Angeles": (34.05, -118.24, "<1H OCEAN"),
    "Santa Monica": (34.02, -118.49, "NEAR OCEAN"),
    "San Diego": (32.72, -117.16, "NEAR OCEAN"),
    "Sacramento": (38.58, -121.49, "INLAND"),
    "Fresno": (36.74, -119.79, "INLAND"),
    "Bakersfield": (35.37, -119.02, "INLAND"),
    "Santa Barbara": (34.42, -119.70, "NEAR OCEAN"),
}


# --------------------------------------------------------------------------- cache
@st.cache_data(show_spinner="Loading data ...")
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner="Training the price model (first visit only, about 10 seconds) ...")
def get_model():
    return train(get_data())


def money(x: float) -> str:
    return f"${x:,.0f}"


def scatter_map(df: pd.DataFrame, color: str, **kw):
    """Plotly map that works on both newer (scatter_map) and older (scatter_mapbox) Plotly."""
    fn = getattr(px, "scatter_map", None)
    args = dict(lat="latitude", lon="longitude", color=color, zoom=4.6,
                center={"lat": 36.8, "lon": -119.4}, height=560, **kw)
    if fn is not None:
        fig = fn(df, map_style="open-street-map", **args)
    else:
        fig = px.scatter_mapbox(df, mapbox_style="open-street-map", **args)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


# --------------------------------------------------------------------------- pages
def home_page():
    df, model = get_data(), get_model()
    st.title("🏠 California Housing Explorer")
    st.markdown(
        "Explore house values across California and get a model estimate for any district. "
        "The data are the **1990 U.S. Census**: each of the **20,640** rows is a block group "
        "(a *district* of usually 600 to 3,000 people). All prices are in **1990 dollars**.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", f"{len(df):,}")
    c2.metric("Median house value", money(df["median_house_value"].median()))
    c3.metric("Model accuracy (R²)", f"{model.metrics['r2']:.2f}")
    c4.metric("Typical model error", money(model.metrics["mae"]))
    st.subheader("What you can do")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 🔎 Explore")
        st.markdown("Filter districts by ocean proximity, income, house age and value. See them on a "
                    "map, compare groups, plot any two variables and download the filtered data.")
        st.page_link(PAGES["explore"], label="Open the dashboard", icon="🔎")
    with b:
        st.markdown("#### 💲 Predict")
        st.markdown("Pick a city or a custom location, describe the district and get an estimated "
                    "median house value with an 80% range, an explanation and similar real districts.")
        st.page_link(PAGES["predict"], label="Open the predictor", icon="💲")
    st.info("Things to know: house values in the data are capped at $500,001, 207 districts have no "
            "bedroom count, and the ISLAND category has only 5 districts.", icon="ℹ️")


def explore_page():
    df = get_data()
    st.title("🔎 Explore the data")
    with st.sidebar:
        st.header("Filters")
        cats = st.multiselect("Ocean proximity", CATEGORIES, default=CATEGORIES)
        inc_lo, inc_hi = float(df.median_income.min()), float(df.median_income.max())
        income = st.slider("Median income ($10,000s)", inc_lo, inc_hi, (inc_lo, inc_hi), step=0.1)
        age = st.slider("Median house age (years)", 1, 52, (1, 52))
        value = st.slider("Median house value ($)", 0, VALUE_CAP, (0, VALUE_CAP), step=10_000,
                          format="$%d")
        include_capped = st.checkbox("Include districts at the $500,001 cap", value=True,
                                     help="Values at the cap are censored: the true value is higher.")
    f = filter_data(df, cats, income, age, value, include_capped)
    s = summary(f)
    if s["districts"] == 0:
        st.warning("No districts match these filters. Widen the filters in the sidebar.")
        return
    c = st.columns(5)
    c[0].metric("Districts", f"{s['districts']:,}", f"{s['districts'] / len(df):.0%} of all",
                delta_color="off")
    c[1].metric("Median value", money(s["median_value"]))
    c[2].metric("Median income", money(s["median_income"] * 10_000))
    c[3].metric("Population", f"{s['population']:,}")
    c[4].metric("At the value cap", f"{s['capped_share']:.1%}")

    t_map, t_dist, t_rel, t_cat, t_data = st.tabs(
        ["🗺️ Map", "📊 Distribution", "📈 Relationships", "🏷️ By category", "📄 Data"])
    with t_map:
        color = st.radio("Color districts by", ["median_house_value", "median_income", "ocean_proximity"],
                         format_func=LABELS.get, horizontal=True)
        m = f.sample(MAP_POINTS, random_state=0) if len(f) > MAP_POINTS else f
        kw = ({"color_discrete_map": COLORS} if color == "ocean_proximity"
              else {"color_continuous_scale": "Viridis"})
        fig = scatter_map(m, color, hover_data={"median_house_value": ":$,.0f", "median_income": ":.2f",
                                                "ocean_proximity": True, "latitude": False,
                                                "longitude": False}, opacity=0.6, **kw)
        st.plotly_chart(fig, use_container_width=True)
        if len(f) > MAP_POINTS:
            st.caption(f"Showing a random {MAP_POINTS:,} of {len(f):,} districts to keep the map fast.")
    numeric = ["median_house_value", "median_income", "housing_median_age", "rooms_per_household",
               "bedrooms_per_room", "people_per_household", "population", "households"]
    with t_dist:
        var = st.selectbox("Variable", numeric, format_func=LABELS.get)
        split = st.checkbox("Split by ocean proximity", value=True)
        d = f[f[var] < f[var].quantile(0.995)] if var.endswith("household") else f
        fig = px.histogram(d, x=var, color="ocean_proximity" if split else None, nbins=60,
                           color_discrete_map=COLORS, labels=LABELS, barmode="overlay", opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(f[var].describe().to_frame(LABELS[var]).T.round(2), use_container_width=True)
    with t_rel:
        a, b = st.columns(2)
        x = a.selectbox("X axis", numeric, index=1, format_func=LABELS.get)
        y = b.selectbox("Y axis", numeric, index=0, format_func=LABELS.get)
        d = f[[x, y, "ocean_proximity"]].dropna()
        d = d.sample(MAP_POINTS, random_state=0) if len(d) > MAP_POINTS else d
        fig = px.scatter(d, x=x, y=y, color="ocean_proximity", opacity=0.4, labels=LABELS,
                         color_discrete_map=COLORS)
        full = f[[x, y]].dropna()
        if len(full) > 2 and x != y:
            slope, icpt = np.polyfit(full[x], full[y], 1)
            xs = np.linspace(full[x].min(), full[x].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="Linear fit",
                                     line=dict(color="black", width=2)))
            r = full[x].corr(full[y])
            st.markdown(f"**Correlation r = {r:.3f}** across {len(full):,} districts. "
                        f"Each +1 in {LABELS[x].lower()} goes with {slope:+,.2f} in {LABELS[y].lower()}.")
        st.plotly_chart(fig, use_container_width=True)
    with t_cat:
        metric = st.selectbox("Compare", numeric, format_func=LABELS.get, key="cat_metric")
        g = (f.groupby("ocean_proximity", observed=True)[metric]
             .agg(["count", "median", "mean"]).reset_index())
        a, b = st.columns([3, 2])
        fig = px.box(f, x="ocean_proximity", y=metric, color="ocean_proximity", labels=LABELS,
                     color_discrete_map=COLORS, category_orders={"ocean_proximity": CATEGORIES})
        fig.update_layout(showlegend=False)
        a.plotly_chart(fig, use_container_width=True)
        b.dataframe(g.rename(columns={"ocean_proximity": "Category", "count": "Districts",
                                      "median": "Median", "mean": "Mean"}).round(2),
                    use_container_width=True, hide_index=True)
        if (g["count"] < 30).any():
            b.caption("Groups with fewer than 30 districts (such as ISLAND) give unreliable averages.")
    with t_data:
        cols = ["longitude", "latitude", "ocean_proximity", "median_house_value", "median_income",
                "housing_median_age", "households", "population", "rooms_per_household",
                "bedrooms_per_room", "people_per_household"]
        st.dataframe(f[cols].round(3), use_container_width=True, height=420, hide_index=True)
        st.download_button("⬇️ Download filtered data (CSV)", f[cols].to_csv(index=False).encode(),
                           "california_housing_filtered.csv", "text/csv")


def predict_page():
    model = get_model()
    st.title("💲 Predict a district's median house value")
    st.markdown("Describe a California district and get the model's estimate in **1990 dollars**. "
                "Start from a city or enter coordinates.")
    city = st.selectbox("Start from a city", list(CITIES))
    preset = CITIES[city]
    with st.form("predict"):
        st.subheader("Location")
        a, b, c = st.columns(3)
        lat = a.number_input("Latitude", 32.0, 42.5, preset[0] if preset else 36.78, step=0.01, format="%.2f")
        lon = b.number_input("Longitude", -125.0, -113.0, preset[1] if preset else -119.42, step=0.01,
                             format="%.2f")
        ocean = c.selectbox("Ocean proximity", CATEGORIES,
                            index=CATEGORIES.index(preset[2]) if preset else 1)
        st.subheader("District")
        a, b, c = st.columns(3)
        income = a.slider("Median household income ($10,000s)", 0.5, 15.0, 3.9, 0.1,
                          help="3.9 means $39,000 a year (1990 dollars).")
        age = b.slider("Median house age (years)", 1, 52, 29)
        households = c.number_input("Households", 10, 6000, 410, step=10)
        a, b, c = st.columns(3)
        rooms = a.number_input("Rooms per household", 1.0, 15.0, 5.2, step=0.1)
        bedrooms = b.number_input("Bedrooms per room", 0.05, 0.6, 0.21, step=0.01)
        people = c.number_input("People per household", 1.0, 10.0, 2.8, step=0.1)
        go_ = st.form_submit_button("Estimate value", type="primary", use_container_width=True)
    if not go_:
        st.caption("Set the inputs and press **Estimate value**.")
        return
    row = build_row(lat, lon, ocean, income, age, households, rooms, bedrooms, people)
    errors, warnings = model.validate(row)
    for e in errors:
        st.error(e, icon="🚫")
    if errors:
        return
    for w in warnings:
        st.warning(w, icon="⚠️")
    p = model.predict(row)
    a, b, c = st.columns(3)
    a.metric("Estimated median house value", money(p["prediction"]))
    b.metric("80% range, low", money(p["low"]))
    c.metric("80% range, high", money(p["high"]))
    if p["high"] >= VALUE_CAP - 1:
        st.info("The upper range reaches the $500,001 cap of the data, so the true value could be higher.",
                icon="ℹ️")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What changes the estimate")
        sens = model.sensitivity(row)
        fig = px.bar(sens, x="Difference", y="Change", orientation="h", text_auto=",.0f",
                     color=sens["Difference"] > 0, color_discrete_map={True: "#2a9d8f", False: "#e76f51"})
        fig.update_layout(showlegend=False, xaxis_title="Change in estimate ($)", yaxis_title="",
                          height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each bar changes one input and keeps the others fixed.")
    with right:
        st.subheader("Where it is")
        pt = pd.DataFrame([{"latitude": lat, "longitude": lon, "label": "Your district"}])
        fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
        style = {"map_style": "open-street-map"} if fn is getattr(px, "scatter_map", None) else {"mapbox_style": "open-street-map"}
        fig = fn(pt, lat="latitude", lon="longitude", hover_name="label", zoom=9,
                 center={"lat": lat, "lon": lon}, height=360, **style)
        fig.update_traces(marker=dict(size=16, color="#e63946"))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Nearest real districts in the 1990 data")
    sim = model.similar_districts(row)
    st.dataframe(sim.rename(columns=LABELS).style.format({
        LABELS["median_house_value"]: "${:,.0f}", LABELS["median_income"]: "{:.2f}",
        LABELS["latitude"]: "{:.2f}", LABELS["longitude"]: "{:.2f}"}),
        use_container_width=True, hide_index=True)
    with st.expander("Model inputs sent to the model"):
        st.json(row)


def model_page():
    model = get_model()
    m = model.metrics
    st.title("📐 About the model")
    st.markdown(
        f"""
The estimate comes from a **histogram gradient boosting regressor** (scikit learn) trained on
{m['n_train']:,} districts and tested on {m['n_test']:,} districts it never saw. Two more models
predict the 10th and 90th percentiles, which give the **80% range**.
""")
    c = st.columns(4)
    c[0].metric("R² (test)", f"{m['r2']:.3f}")
    c[1].metric("Mean absolute error", money(m["mae"]))
    c[2].metric("Root mean squared error", money(m["rmse"]))
    c[3].metric("Share of test districts inside the 80% range", f"{m['coverage_80']:.0%}")
    st.markdown(
        """
**Inputs.** Location (latitude, longitude, ocean proximity), median income, median house age and the
district's size (households, rooms, bedrooms, people). The predictor asks for per household values and
converts them to the district totals that the model was trained on.

**Safety checks.** The app refuses locations outside California or far from any 1990 district, and
warns when an input is outside the range seen in training.

**Limits.** Prices are in 1990 dollars. The training target is capped at \\$500,001, so the model cannot
predict above it. The 80% range covers fewer test districts than 80% (see above), so treat it as a
rough guide.
""")


PAGES = {
    "home": st.Page(home_page, title="Home", icon="🏠", default=True),
    "explore": st.Page(explore_page, title="Explore", icon="🔎", url_path="explore"),
    "predict": st.Page(predict_page, title="Predict", icon="💲", url_path="predict"),
    "model": st.Page(model_page, title="About the model", icon="📐", url_path="model"),
}
st.navigation(list(PAGES.values())).run()
