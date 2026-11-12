# Diwali Air Quality Project
# Idea: check how much AQI (air pollution) rises in Indian cities around Diwali.
# Libraries: pandas, numpy, scikit-learn, matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Diwali date of each year in our data
diwali_dates = {
    2015: "2015-11-11",
    2016: "2016-10-30",
    2017: "2017-10-19",
    2018: "2018-11-07",
    2019: "2019-10-27",
    2020: "2020-11-14",
}

# pollutant columns we will use
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# folder to save charts and results
os.makedirs("outputs", exist_ok=True)


# ---------- Step 1: Load the data ----------
df = pd.read_csv("data/city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])   # convert text dates into real dates

print(df.shape)
print(df.head())
print(df.isnull().sum())                  # how many missing values in each column


# ---------- Step 2: Clean the data ----------
df = df.dropna(subset=["AQI"])            # remove rows where AQI is missing

for col in pollutants:                    # fill missing pollutant values with the median
    df[col] = df[col].fillna(df[col].median())


# ---------- Step 3: Take 7 days before and after each Diwali ----------
all_windows = []
for year in diwali_dates:
    diwali = pd.Timestamp(diwali_dates[year])
    start = diwali - pd.Timedelta(days=7)
    end = diwali + pd.Timedelta(days=7)

    part = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    part["Day_Offset"] = (part["Date"] - diwali).dt.days   # -2 = 2 days before Diwali
    all_windows.append(part)

festival_df = pd.concat(all_windows)

# label every row as Before / Diwali / After
festival_df["Period"] = "After"
festival_df.loc[festival_df["Day_Offset"] <= 1, "Period"] = "Diwali"
festival_df.loc[festival_df["Day_Offset"] < 0, "Period"] = "Before"


# ---------- Step 4: Compare cities ----------
# average AQI of each city in each period
summary = festival_df.pivot_table(index="City", columns="Period", values="AQI", aggfunc="mean")
summary = summary[["Before", "Diwali", "After"]].dropna()

# percent change from Before to Diwali
summary["Change_%"] = (summary["Diwali"] - summary["Before"]) / summary["Before"] * 100

# mark cities where AQI on Diwali crossed 300 (very poor)
summary["Danger_Zone"] = np.where(summary["Diwali"] > 300, "Yes", "No")

summary = summary.round(1).sort_values("Change_%", ascending=False)
print(summary.head(10))

changes = summary["Change_%"].values
print("Average rise:", round(np.mean(changes), 1), "%")
print("Std deviation:", round(np.std(changes), 1))
print("Biggest rise:", round(np.max(changes), 1), "% in", summary["Change_%"].idxmax())
print("Cities in danger zone:", (summary["Danger_Zone"] == "Yes").sum(), "out of", len(summary))

summary.to_csv("outputs/city_summary.csv")


# ---------- Step 5: Charts ----------
# Chart 1: top 10 cities with biggest rise
top10 = summary.head(10)
plt.figure(figsize=(9, 5))
plt.bar(top10.index, top10["Change_%"], color="orange")
plt.title("Top 10 cities: AQI rise on Diwali vs week before")
plt.ylabel("Change (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/chart1_top10_cities.png")
plt.show()

# Chart 2: AQI day by day around Diwali (0 = Diwali day)
all_cities_avg = festival_df.groupby("Day_Offset")["AQI"].mean()
delhi_avg = festival_df[festival_df["City"] == "Delhi"].groupby("Day_Offset")["AQI"].mean()

plt.figure(figsize=(9, 5))
plt.plot(all_cities_avg.index, all_cities_avg.values, marker="o", label="All cities")
plt.plot(delhi_avg.index, delhi_avg.values, marker="o", label="Delhi")
plt.axvline(0, color="red", linestyle="--", label="Diwali day")
plt.title("AQI around Diwali")
plt.xlabel("Days from Diwali")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/chart2_aqi_around_diwali.png")
plt.show()


# ---------- Step 6: Machine learning (predict AQI from pollutants) ----------
X = df[pollutants]    # input
y = df["AQI"]         # output we want to predict

# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)            # train the model
predictions = model.predict(X_test)    # predict on data the model has not seen

print("R2 score:", round(r2_score(y_test, predictions), 3))
print("Mean absolute error:", round(mean_absolute_error(y_test, predictions), 1))

for name, coef in zip(pollutants, model.coef_):
    print(name, round(coef, 3))

# Chart 3: actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, s=5, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.title("Actual vs predicted AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.tight_layout()
plt.savefig("outputs/chart3_actual_vs_predicted.png")
plt.show()
