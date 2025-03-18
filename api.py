import requests
import xml.etree.ElementTree as ET
from textblob import TextBlob
from collections import Counter

def fetch_articles_api(company_name, max_articles=15):  # default 15, but you can pass any number
    query = company_name.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}"

    response = requests.get(url)

    if response.status_code != 200:
        return []  # If something went wrong, return an empty list

    root = ET.fromstring(response.content)

    articles = []
    count = 0  # To keep track of the number of articles fetched

    for item in root.iter('item'):
        if count >= max_articles:
            break  # Exit the loop if the limit is reached
        
        title = item.find('title').text
        link = item.find('link').text

        articles.append({
            "title": title,
            "link": link
        })

        count += 1  # Increment the counter

    return articles



def analyze_sentiment_api(summaries):
    sentiments = []
    for summary in summaries:
        blob = TextBlob(summary)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append("Positive")
        elif polarity < 0:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments


def generate_comparative_analysis_api(sentiments):
    sentiment_counts = Counter(sentiments)
    total = sum(sentiment_counts.values())

    return {
        "Sentiment Distribution": dict(sentiment_counts),
        "Summary": f"Out of {total} articles: {sentiment_counts.get('Positive', 0)} Positive, {sentiment_counts.get('Negative', 0)} Negative, {sentiment_counts.get('Neutral', 0)} Neutral."
    }
