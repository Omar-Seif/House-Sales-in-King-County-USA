# 🏠 King County House Price Prediction

An end-to-end machine learning project that predicts residential property prices in King County, USA, and provides an intelligent price range estimation via an interactive Streamlit application.

---

## 📌 Project Overview

This project implements a full production-style ML pipeline:

- Data Cleaning & Wrangling
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Comparison (Linear → Ridge → Random Forest → XGBoost)
- Hyperparameter Tuning
- Prediction Interval Estimation
- Neighborhood Clustering (KMeans on lat/long)
- Interactive Streamlit Deployment

The final model achieves:

- **R²:** 0.904
- **MAE:** ~$64,788
- **MAPE:** ~12%
- **Accuracy:** ~88%

---

## 📊 Dataset

- **House Sales in King County, USA**
- ~21,000 properties
- Sold between May 2014 – May 2015
- Includes structural, geographic, and quality-based features

---

## 🧠 Feature Engineering Highlights

- **Log transformation** for skewed square-foot features
- Derived features:
  - `house_age`
  - `basement_flag`
  - `renovated_flag`
- Removed multicollinear features:
  - `yr_built`
  - `yr_renovated`
  - `sqft_above`
  - `sqft_lot15`
- **Neighborhood clustering** via KMeans on (lat, long)
- Map-based geographic input

---

## 🤖 Model Development

**Models tested:**

| Model | Test R² | Test MAE |
|-------|---------|----------|
| Linear Regression | 0.736 | ~$112k |
| Ridge Regression | 0.736 | ~$112k |
| Random Forest | 0.859 | ~$74k |
| XGBoost | 0.904 | ~$64k |

**Final Model**

Tuned XGBoost with:
- `n_estimators=400`
- `max_depth=7`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`

---

## 📈 Prediction Interval

Instead of returning a single price point, the application provides:

**Estimated Price ± Model MAE**

This provides a realistic and business-oriented price range.

---

## 🗺️ Interactive Streamlit App

**Features:**

- Clickable map (restricted to King County bounds)
- Auto-derived latitude/longitude
- Neighborhood dropdown (KMeans clustered)
- Checkbox-based binary inputs
- Slider-based rating inputs
- Year-built calendar input
- Raw input display (no log distortion)
- Real-time price range estimation

---

## 📂 Project Structure


project/    
├── data/    
│   ├── data_wrangling.csv    
│   ├── eda_FE.csv    
│   └── kc_house_data.csv    
├── json/    
│   ├── geo_meta.json    
│   └── interval.json    
├── model/    
│   ├── geo_clusterer.joblib    
│   └── model.joblib    
├── src/    
│   ├── data_wrangling.ipynb    
│   ├── eda_FE.ipynb    
│   ├── lat_long_cord.ipynb    
│   ├── ML-pipeline.ipynb    
│   └── house_sales_app.py    
├── README.md    
└── requirements.txt    
  

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Omar-Seif/House-Sales-in-King-County-USA.git](https://github.com/Omar-Seif/House-Sales-in-King-County-USA.git)
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app**
   ```bash
   cd src
   streamlit run src/house_sales_app.py
   ```

---

## 🏆 Key Takeaways

- Location is the strongest predictor (lat + cluster)
- Grade significantly impacts valuation
- Boosting methods outperform bagging
- Log transformation greatly improves regression stability
- Production-ready ML requires consistent feature transformation

---

## 📬 Author

Omar Seif    
LinkedIn: [https://www.linkedin.com/in/omar-farahat-313987277/](https://www.linkedin.com/in/omar-farahat-313987277/)      
GitHub: [Omar Seif](https://github.com/Omar-Seif)      

