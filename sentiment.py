import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not installed
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to apply sentiment analysis
def apply_sentiment_analysis(news_file):
    # Load news data
    news_df = pd.read_parquet(news_file)
    
    # Apply sentiment scoring
    news_df['sentiment_score'] = news_df['title'].apply(lambda text: sia.polarity_scores(text)['compound'])
    
    return news_df

# Function to generate Buy/Sell/Hold recommendation
def get_recommendation(sentiment_score):
    if sentiment_score > 0.3:
        return "Buy"
    elif sentiment_score < -0.3:
        return "Sell"
    else:
        return "Hold"

# Main function to process sentiment data
def generate_sentiment_report():
    # Apply sentiment to both Yahoo and Google News
    google_news_df = pd.read_parquet("./news_data/Google_News_with_Sentiment.parquet")
    yahoo_news_df = pd.read_parquet("./news_data/Yahoo_Finance_News_with_Sentiment.parquet")
    
    # Compute overall sentiment scores
    google_sentiment = google_news_df['sentiment_score'].mean()
    yahoo_sentiment = yahoo_news_df['sentiment_score'].mean()
    overall_sentiment = (google_sentiment + yahoo_sentiment) / 2

    # Generate recommendation
    recommendation = get_recommendation(overall_sentiment)

    # Find best and worst news
    best_google_news = google_news_df.loc[google_news_df['sentiment_score'].idxmax()]
    worst_google_news = google_news_df.loc[google_news_df['sentiment_score'].idxmin()]
    best_yahoo_news = yahoo_news_df.loc[yahoo_news_df['sentiment_score'].idxmax()]
    worst_yahoo_news = yahoo_news_df.loc[yahoo_news_df['sentiment_score'].idxmin()]

    # Print report
    print(f"Google Sentiment Score: {google_sentiment:.2f}")
    print(f"Yahoo Sentiment Score: {yahoo_sentiment:.2f}")
    print(f"Overall Sentiment Score: {overall_sentiment:.2f}")
    print(f"Trading Recommendation: {recommendation}")

    print("\nBest Google News Article:")
    print(f"Title: {best_google_news['title']}")
    print(f"Sentiment Score: {best_google_news['sentiment_score']:.2f}")

    print("\nWorst Google News Article:")
    print(f"Title: {worst_google_news['title']}")
    print(f"Sentiment Score: {worst_google_news['sentiment_score']:.2f}")

    print("\nBest Yahoo News Article:")
    print(f"Title: {best_yahoo_news['title']}")
    print(f"Sentiment Score: {best_yahoo_news['sentiment_score']:.2f}")

    print("\nWorst Yahoo News Article:")
    print(f"Title: {worst_yahoo_news['title']}")
    print(f"Sentiment Score: {worst_yahoo_news['sentiment_score']:.2f}")

    # Save final sentiment summary
    sentiment_summary = pd.DataFrame({
        "Google_Sentiment": [google_sentiment],
        "Yahoo_Sentiment": [yahoo_sentiment],
        "Overall_Sentiment": [overall_sentiment],
        "Recommendation": [recommendation]
    })
    
    sentiment_summary.to_parquet("./news_data/Sentiment_Report.parquet", index=False)

# Run sentiment module
generate_sentiment_report()