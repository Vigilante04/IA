import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle
import feedparser
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from urllib.parse import quote_plus

nltk.download('vader_lexicon')

# ✅ Load T5 Model and Classifier
def load_model_and_tokenizer():
    tokenizer_loaded = T5Tokenizer.from_pretrained(r'C:\Users\abhin\Desktop\miniproject\saved_model')
    model_loaded = T5ForConditionalGeneration.from_pretrained(r'C:\Users\abhin\Desktop\miniproject\saved_model')
    model_loaded.to(torch.device('cpu'))  # Move model to CPU

    with open(r'C:\Users\abhin\Desktop\miniproject\saved_model\classifier.pkl', 'rb') as f:
        classifier_loaded = pickle.load(f)

    print("Model, tokenizer, and classifier loaded.")
    return tokenizer_loaded, model_loaded, classifier_loaded

# ✅ Summarize News
def generate_summary(input_text, tokenizer, model):
    input_ids = tokenizer(f"summarize: {input_text}", max_length=5120, truncation=True, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

# ✅ Sentiment Classification
def classify_sentiment(input_text, classifier):
    labels = ["Positive", "Negative", "Neutral"]
    classification_result = classifier(input_text, labels)
    predicted_sentiment = classification_result['labels'][0]
    confidence_score = int(classification_result['scores'][0] * 100)
    return predicted_sentiment, confidence_score

# ✅ Generate Stock Recommendations
def generate_recommendation(sentiment, confidence_score):
    if sentiment == "Positive":
        if confidence_score > 90:
            recommendation = "Strong Buy"
        elif confidence_score > 75:
            recommendation = "Buy"
        else:
            recommendation = "Consider Buying"
    elif sentiment == "Negative":
        if confidence_score > 90:
            recommendation = "Strong Sell"
        elif confidence_score > 75:
            recommendation = "Sell"
        else:
            recommendation = "Consider Selling"
    else:
        if confidence_score > 80:
            recommendation = "Hold"
        else:
            recommendation = "Neutral"

    return recommendation

# ✅ Fetch Google News & Perform Sentiment Analysis
def get_google_news_rss(query):
    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    news_data = []
    sia = SentimentIntensityAnalyzer()

    for entry in feed.entries[:5]:  # Limit to top 5 news articles
        title = entry.title
        description = entry.summary
        link = entry.link
        sentiment = sia.polarity_scores(description)

        news_data.append({
            "title": title,
            "description": description,
            "url": link,
            "compound": sentiment["compound"],
            "positive": sentiment["pos"],
            "negative": sentiment["neg"],
            "neutral": sentiment["neu"],
        })

    df = pd.DataFrame(news_data)
    return df

# ✅ Classify & Summarize News
def process_news(news_df, tokenizer, model, classifier):
    results = []
    
    for _, row in news_df.iterrows():
        summary = generate_summary(row["description"], tokenizer, model)
        sentiment, confidence_score = classify_sentiment(row["description"], classifier)
        recommendation = generate_recommendation(sentiment, confidence_score)

        results.append({
            "Title": row["title"],
            "Summary": summary,
            "Sentiment": sentiment,
            "Confidence Score": confidence_score,
            "Recommendation": recommendation,
            "URL": row["url"]
        })
    
    return results

# ✅ Main Execution
def main():
    query = input("Enter stock news query (e.g., 'Indian stock market'): ")
    
    # Load Model & Tokenizer
    tokenizer, model, classifier = load_model_and_tokenizer()

    # Fetch News
    news_df = get_google_news_rss(query)

    if not news_df.empty:
        print("\nAnalyzing News and Performing Sentiment Analysis...")
        results = process_news(news_df, tokenizer, model, classifier)

        # Display results
        for i, result in enumerate(results):
            print(f"\n📌 **News {i+1}:** {result['Title']}")
            print(f"🔹 **Summary:** {result['Summary']}")
            print(f"📊 **Sentiment:** {result['Sentiment']} (Confidence: {result['Confidence Score']}%)")
            print(f"💡 **Recommendation:** {result['Recommendation']}")
            print(f"🔗 **Read More:** {result['URL']}\n")
    else:
        print("❗ No news articles found.")

if __name__ == "__main__":
    main()