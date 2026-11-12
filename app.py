# Streamlit website for the Diwali Air Quality project
# Run locally:  streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set this to False once data/city_day.csv is the REAL Kaggle file
USING_SAMPLE_DATA = True

diwali_dates = {
    2015: "2015-11-11",
    2016: "2016-10-30",
    2017: "2017-10-19",
    2018: "2018-11-07",
    2019: "2019-10-27",
    2020: "2020-11-14",
}
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

st.set_page_config(page_title="Diwali AQI Analyzer", page_icon="🪔")
st.title("🪔 Diwali Air Quality Impact Analyzer")
st.write("How much does AQI (air pollution) rise in Indian cities around Diwali?")

if USING_SAMPLE_DATA:
    st.warning("This website is using SIMULATED sample data for demonstration, not real measurements.")


# ---------- Load and clean data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/city_day.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["AQI"])
    for col in pollutants:
        df[col] = df[col].fillna(df[col].median())
    return df

df = load_data()


# ---------- Days around Diwali ----------
@st.cache_data
def make_festival_data(days):
    all_windows = []
    for year in diwali_dates:
        diwali = pd.Timestamp(diwali_dates[year])
        start = diwali - pd.Timedelta(days=days)
        end = diwali + pd.Timedelta(days=days)
        part = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
        part["Day_Offset"] = (part["Date"] - diwali).dt.days
        all_windows.append(part)
    festival_df = pd.concat(all_windows)

    festival_df["Period"] = "After"
    festival_df.loc[festival_df["Day_Offset"] <= 1, "Period"] = "Diwali"
    festival_df.loc[festival_df["Day_Offset"] < 0, "Period"] = "Before"
    return festival_df

days = st.sidebar.slider("Days before/after Diwali to look at", 3, 10, 7)
danger = st.sidebar.slider("Danger zone AQI", 150, 400, 300)
festival_df = make_festival_data(days)


# ---------- City comparison ----------
summary = festival_df.pivot_table(index="City", columns="Period", values="AQI", aggfunc="mean")
summary = summary[["Before", "Diwali", "After"]].dropna()
summary["Change_%"] = (summary["Diwali"] - summary["Before"]) / summary["Before"] * 100
summary["Danger_Zone"] = np.where(summary["Diwali"] > danger, "Yes", "No")
summary = summary.round(1).sort_values("Change_%", ascending=False)

tab1, tab2, tab3 = st.tabs(["All cities", "One city", "Predict AQI"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Average rise", f"{summary['Change_%'].mean():.1f} %")
    col2.metric("Biggest rise", summary["Change_%"].idxmax())
    col3.metric("Cities in danger zone", int((summary["Danger_Zone"] == "Yes").sum()))
    st.subheader("% rise in AQI on Diwali vs before")
    st.bar_chart(summary["Change_%"])
    st.dataframe(summary)

with tab2:
    city = st.selectbox("Choose a city", summary.index)
    city_line = festival_df[festival_df["City"] == city].groupby("Day_Offset")["AQI"].mean()
    st.subheader("Average AQI around Diwali (0 = Diwali day)")
    st.line_chart(city_line)

with tab3:
    st.write("A Linear Regression model predicts AQI from pollutant levels.")
    X = df[pollutants]
    y = df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write("Model R2 score on test data:", round(r2_score(y_test, model.predict(X_test)), 3))

    values = []
    for p in pollutants:
        default = float(df[p].median())
        maximum = float(df[p].quantile(0.99))
        values.append(st.slider(p, 0.0, maximum, default))
    prediction = model.predict(pd.DataFrame([values], columns=pollutants))[0]
    st.subheader(f"Predicted AQI: {prediction:.0f}")
