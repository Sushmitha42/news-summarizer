import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime

# Import custom API and utility modules
from api import (
    fetch_articles_api,
    analyze_sentiment_api,
    generate_comparative_analysis_api
    
)

from utils import (
    summarize_articles,
    get_topics,
    text_to_speech_hindi,
    generate_coverage_differences,
    generate_topic_overlap
    
)

# Streamlit page configuration
st.set_page_config(page_title="News Summarizer & Hindi TTS", layout="wide")

# App Title and description
st.title("📰 News Summarizer, Sentiment Analysis & Hindi TTS App")
st.write("Fetch news articles about a company, analyze them, and listen to a Hindi audio summary.")

# Current Date and Time display
current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
st.markdown(f"🕒 **Date & Time:** {current_datetime}")

# Input section
company_name = st.text_input("Enter Company Name", "")
article_limit = st.slider("Number of articles to fetch", min_value=5, max_value=50, value=10)

# Fetch News and Analyze button
if st.button("Fetch News and Analyze"):
    if not company_name.strip():
        st.warning("Please enter a company name.")
    else:
        st.info(f"Fetching {article_limit} articles for '{company_name}'...")

        articles = fetch_articles_api(company_name, max_articles=article_limit)

        if not articles:
            st.error("No articles found. Try another company.")
        else:
            st.success(f"Fetched {len(articles)} articles! Analyzing...")

            summaries = summarize_articles(articles)
            sentiments = analyze_sentiment_api(summaries)
            topics = get_topics(summaries)

            comparative_analysis = generate_comparative_analysis_api(sentiments)
            coverage_differences = generate_coverage_differences(summaries, topics)
            topic_overlap = generate_topic_overlap(topics)

            st.success("✅ Analysis Complete!")

            report = {
                "Company": company_name,
                "Articles": []
            }

            for i, article in enumerate(articles):
                report["Articles"].append({
                    "Title": article['title'],
                    "Summary": summaries[i],
                    "Sentiment": sentiments[i],
                    "Topics": topics[i]
                })

            report["Comparative Analysis"] = comparative_analysis
            report["Coverage Differences"] = coverage_differences
            report["Topic Overlap"] = topic_overlap

            st.subheader("📋 Detailed JSON Report")
            st.json(report)

           
            st.subheader("📊 Sentiment Distribution")
            sentiments_df = pd.DataFrame(sentiments, columns=["Sentiment"])
            sentiment_counts = sentiments_df["Sentiment"].value_counts()

            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', color=["green", "red", "gray"], ax=ax)
            plt.title("Sentiment Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Number of Articles")
            st.pyplot(fig)

            st.subheader("☁️ Word Clouds for Topics")
            all_topics = [topic for sublist in topics for topic in sublist]
            if all_topics:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_topics))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.warning("No topics found to generate a word cloud.")

            st.subheader("🔊 Hindi Audio Summary")

            total_articles = comparative_analysis.get('Total Articles', 0)
            positive_articles = comparative_analysis.get('Positive Articles', 0)
            negative_articles = comparative_analysis.get('Negative Articles', 0)
            neutral_articles = comparative_analysis.get('Neutral Articles', 0)

            hindi_summary = (
                f"{company_name} की खबरों का विश्लेषण: "
                f"{total_articles} लेखों में से "
                f"{positive_articles} सकारात्मक, "
                f"{negative_articles} नकारात्मक "
                f"और {neutral_articles} तटस्थ हैं।"
            )

            st.write(hindi_summary)

            audio_file = text_to_speech_hindi(hindi_summary)
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format='audio/mp3')
            else:
                st.error("Failed to generate Hindi speech.")

            st.subheader("⬇️ Download Report")
            json_report = json.dumps(report, indent=4)
            st.download_button("Download JSON Report", json_report, file_name=f"{company_name}_report.json", mime="application/json")



           
