import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Twitter Sentiment & Emotion Analysis", layout="wide")
st.title("💬 Twitter Sentiment & Emotion Analysis App")
st.markdown("Analyze real tweets for sentiment and emotion using NLP and ML.")

# ------------------ User Inputs ------------------
query = st.text_input("🔍 Enter a topic/hashtag (e.g., Flipkart, elections, #Python):")
tweet_count = st.slider("Number of tweets to analyze:", min_value=50, max_value=500, step=50, value=200)

# ------------------ Load Emotion Model ------------------
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, emotion_model = load_emotion_model()

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = torch.sigmoid(logits)[0].numpy()
    labels = emotion_model.config.id2label
    return labels[np.argmax(probs)]

# ------------------ Fetch Tweets ------------------
@st.cache_data
def fetch_tweets(query, max_tweets):
    tweets_data = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query + " lang:en").get_items()):
        if i >= max_tweets:
            break
        tweets_data.append([tweet.date, tweet.content])
    return pd.DataFrame(tweets_data, columns=["Datetime", "Tweet"])

# ------------------ Sentiment Analysis ------------------
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    return polarity

# ------------------ Main App ------------------
if query:
    with st.spinner("Fetching and analyzing tweets..."):
        df = fetch_tweets(query, tweet_count)
        df["Polarity"] = df["Tweet"].apply(analyze_sentiment)
        df["Sentiment"] = df["Polarity"].apply(
            lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
        )
        df["Date"] = pd.to_datetime(df["Datetime"]).dt.date

        # Detect Emotion
        with st.spinner("Detecting emotions..."):
            df["Emotion"] = df["Tweet"].apply(detect_emotion)

        # Sample Tweets
        st.subheader("📄 Sample Tweets")
        st.dataframe(df[["Date", "Tweet", "Sentiment", "Emotion"]].head(10))

        # 📈 Sentiment Trend Over Time
        st.subheader("📈 Sentiment Trend Over Time")
        trend_data = df.groupby(["Date", "Sentiment"]).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4))
        trend_data.plot(ax=ax, marker='o')
        ax.set_ylabel("Tweet Count")
        ax.set_title("Sentiment Trend")
        st.pyplot(fig)

        # 🏆 Top Tweets by Sentiment
        st.subheader("🏆 Top Tweets by Sentiment")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 😊 Top 3 Positive Tweets")
            top_pos = df[df["Sentiment"] == "Positive"].sort_values(by="Polarity", ascending=False).head(3)
            for i, row in top_pos.iterrows():
                st.write(f"🔹 {row['Tweet']}")

        with col2:
            st.markdown("### 😠 Top 3 Negative Tweets")
            top_neg = df[df["Sentiment"] == "Negative"].sort_values(by="Polarity").head(3)
            for i, row in top_neg.iterrows():
                st.write(f"🔻 {row['Tweet']}")

        # 📊 Sentiment Distribution
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        fig_sent, ax_sent = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax_sent, palette="coolwarm")
        st.pyplot(fig_sent)

        # ☁️ Word Cloud
        st.subheader("☁️ Word Cloud of Tweets")
        all_words = ' '.join(df["Tweet"])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        # 🧠 Emotion Distribution
        st.subheader("🧠 Emotion Distribution in Tweets")
        emotion_counts = df["Emotion"].value_counts()
        fig_emotion, ax_emotion = plt.subplots()
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, ax=ax_emotion, palette="Set2")
        ax_emotion.set_title("Emotion Frequency")
        st.pyplot(fig_emotion)

        # ⬇️ CSV Download
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv, file_name="tweet_sentiment_emotions.csv", mime="text/csv")
