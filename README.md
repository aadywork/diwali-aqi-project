# Diwali Air Quality Impact Analyzer

**Question:** How much does air pollution (AQI) rise in Indian cities around Diwali, and can a simple ML model predict AQI from pollutant levels?

> **Note on data:** `data/city_day.csv` currently contains SIMULATED sample data made by `make_sample_data.py` (same columns as the Kaggle dataset "Air Quality Data in India 2015-2020"). To get real results, replace it with the real Kaggle `city_day.csv` and set `USING_SAMPLE_DATA = False` in `app.py`.

## Idea
Every year, Diwali is followed by news of "danger zone" air quality. This project compares each city's AQI in the days **before** Diwali, on **Diwali** (day 0 and 1) and **after**, across 2015-2020.

## Libraries and what they do here
| Library | Used for |
|---|---|
| Pandas | Loading the CSV, cleaning missing values, grouping by city, pivot tables |
| NumPy | Percent change, mean / std / max, danger-zone flag with `np.where` |
| Scikit-learn | Linear Regression to predict AQI from pollutants; train/test split; R2 and MAE |
| Matplotlib | 3 charts saved in `outputs/` |
| Streamlit | A small website that shows the results |

## Files
```
main.py              analysis + model, prints results, saves charts to outputs/
app.py               Streamlit website (same logic, interactive)
make_sample_data.py  creates the simulated sample data
requirements.txt     libraries to install
data/city_day.csv    the dataset
```

## How to run
```
pip install -r requirements.txt
python main.py            # analysis, charts, model score
streamlit run app.py      # website on your own computer
```

## Steps in main.py
1. Load data
2. Clean data (drop rows with no AQI, fill pollutant gaps with the median)
3. Take 7 days before and after each Diwali
4. Compare cities: before vs Diwali vs after, % rise, danger-zone flag
5. Charts
6. Linear Regression model and its score

## Results
*(Fill in after running on the dataset you present.)*
- City with the biggest rise: ...
- Average rise across cities: ...
- Cities in danger zone: ...
- Model R2: ...   MAE: ...

## Limitations
- Sample data is simulated, so its results only show that the pipeline works, not real findings.
- Data ends in 2020.
- Weather, wind and stubble burning also affect AQI, so this shows correlation, not proof that crackers alone caused the rise.
- Missing values are filled with the median, which is an assumption.
- AQI is calculated from pollutant values, so predicting it from them gives a high score by design.

## Future work
- Use the real Kaggle data, add electricity consumption and weather data.
- Try Random Forest and predict next-day AQI.
