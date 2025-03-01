# **Stock Price Analyzer**

## **Overview**

Stock Price Analyzer is a comprehensive machine learning project designed to predict stock prices using both traditional and deep learning models. It integrates sentiment analysis from financial news and incorporates technical indicators to enhance predictive performance. The project is structured into different modules, each handling data collection, feature engineering, model training, evaluation, and visualization.

## **Project Structure**

```
StockPriceAnalyzer/
│── models/
│   ├── ARIMA_Predictions.parquet
│   ├── LSTM_No_Indicators_Predictions.parquet
│   ├── LSTM_with_Indicators_Predictions.parquet
│   ├── RandomForest_No_Indicators.parquet
│   ├── RandomForest_With_Indicators.parquet
│   ├── Other model-related files...
│
│── news_data/
│   ├── Google_News.parquet
│   ├── Yahoo_Finance_News.parquet
│   ├── Google_News_with_Sentiment.parquet
│   ├── Yahoo_Finance_News_with_Sentiment.parquet
│   ├── Sentiment_Report.parquet
│
│── stock_data/
│   ├── AAPL_stock_data.parquet
│   ├── Processed_Stock_Data.parquet
│   ├── Processed_Stock_Data_with_Indicators.parquet
│   ├── Cleaned_Stock_Data.parquet
│   ├── Stock_Data_with_Indicators.parquet
│
│── .env
│── .gitignore
│── data_collection_and_cleaning.ipynb
│── feature_engineering.ipynb
│── models.ipynb
│── recommendation.ipynb
│── sentiment_analyzer.ipynb
│── sentiment_twitter.ipynb
│── sentiment.py
│── trial_arima_model.ipynb
│── visualization.ipynb
```

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/StockPriceAnalyzer.git
cd StockPriceAnalyzer
```

### **2. Set Up a Virtual Environment (Optional)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **3. Install Required Packages**

```bash
pip install -r requirements.txt
```

Ensure you have `TA-Lib` and `yfinance` installed for technical indicators and stock data retrieval.

---

## **Project Modules**

### **1. Data Collection & Cleaning (**``**)**

- Fetches historical stock data using `yfinance`
- Cleans the dataset by handling missing values and anomalies
- Saves the processed data in `.parquet` format for efficiency

### **2. Feature Engineering (**``**)**

- Computes various technical indicators such as:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
- Prepares datasets for model training (with and without indicators)

### **3. Model Training & Predictions (**``**)**

- Trains and evaluates multiple models:
  - **Random Forest** (With & Without Indicators)
  - **LSTM** (With & Without Indicators)
  - **ARIMA** (Initially included but dropped due to poor performance)
- Generates predictions and saves them in `.parquet` format

### **4. Sentiment Analysis (**``** & **``**)**

- Fetches financial news from:
  - **Yahoo Finance**
  - **Google News**
- Uses `NLTK VADER` for sentiment scoring of news headlines
- Generates a **Buy/Sell/Hold** recommendation based on sentiment scores

#### **Sentiment Processing (**``**)**

This script:

- Loads news data
- Applies sentiment analysis using VADER
- Determines an overall sentiment score
- Generates a **trading recommendation** based on sentiment trends
- Identifies the **best and worst** news articles

### **5. Recommendation System (**``**)**

- Uses sentiment scores and stock trends to generate insights
- Aims to provide **actionable trading advice**

### **6. Visualization (**``**)**

- Uses **Plotly** for interactive stock price and sentiment trend visualizations
- Visualizes:
  - Actual vs Predicted stock prices
  - Technical indicators (RSI, MACD, SMA, EMA)
  - Sentiment trends over time
  - Buy/Sell/Hold recommendations

---

## **Usage**

### **1. Running Stock Data Collection**

```python
# Open and run data_collection_and_cleaning.ipynb
```

### **2. Generating Technical Indicators**

```python
# Open and run feature_engineering.ipynb
```

### **3. Training Models**

```python
# Open and run models.ipynb
```

### **4. Running Sentiment Analysis**

```bash
python sentiment.py
```

### **5. Visualizing Results**

```python
# Open and run visualization.ipynb
```

---

## **Results & Observations**

1. **Random Forest performed better without indicators** (lower MAE & RMSE)
2. **LSTM performed better without indicators** (slightly worse when using indicators)
3. **Sentiment Analysis provides additional insights** but is not directly integrated into model training
4. **Buy/Sell/Hold recommendations** are generated from sentiment scores

---

## **Future Enhancements**

- Integrate **real-time stock price fetching**
- Improve sentiment analysis with **Transformer-based NLP models**
- Develop a **Streamlit frontend** for user-friendly interaction
- Fine-tune hyperparameters for **better LSTM performance**

---

## **Contributors**

- **Naushil Khajanchi** - Data Scientist, ML Engineer
- **GitHub**: [GitHub Profile](https://github.com/Naushil7)
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/naushilkhajanchi/)
---

## **License**

This project is open-source and available under the **MIT License**.