import streamlit as st
import os
from api import fetch_articles_api, analyze_sentiment_api, generate_comparative_analysis_api
from utils import summarize_articles, get_topics, text_to_speech_hindi, generate_coverage_differences, generate_topic_overlap

# Streamlit page setup
st.set_page_config(page_title="News Summarizer & TTS", layout="centered")

# Page title and description
st.title("📰 News Summarizer and Hindi TTS App")
st.write("Fetch news articles about a company, analyze sentiment, and listen to Hindi audio summary.")

# User input for company name
company_name = st.text_input("Enter Company Name", "")
article_limit = st.slider("Number of articles to fetch", min_value=5, max_value=50, value=15)

# Main button to start processing
if st.button("Fetch News and Analyze"):
    if company_name.strip():
        st.info(f"Fetching {article_limit} articles for '{company_name}'...")

        # Fetch articles from API
        articles = fetch_articles_api(company_name, max_articles=article_limit)

        if articles:
            st.success(f"Fetched {len(articles)} articles! Now analyzing...")

            # Step 1: Summarization
            summaries = summarize_articles(articles)

            # Step 2: Sentiment Analysis
            sentiments = analyze_sentiment_api(summaries)

            # Step 3: Topic Extraction
            topics = get_topics(summaries)

            # Step 4: Comparative Sentiment Analysis
            comparative_analysis = generate_comparative_analysis_api(sentiments)

            # Step 5: Coverage Differences between articles
            coverage_differences = generate_coverage_differences(summaries, topics)

            # Step 6: Topic Overlap between articles
            topic_overlap = generate_topic_overlap(topics)

            st.success("Analysis complete!")

            # Building and displaying the report
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

            # Display the JSON report
            st.subheader("📋 Detailed Report")
            st.json(report)

            # Display Coverage Differences
            st.subheader("📝 Coverage Differences")
            if coverage_differences:
                for diff in coverage_differences:
                    st.markdown(f"- **Comparison**: {diff['Comparison']}")
                    st.markdown(f"  **Impact**: {diff['Impact']}")
            else:
                st.warning("Not enough articles to compare coverage differences.")

            # Display Topic Overlap
            st.subheader("🔎 Topic Overlap")
            if topic_overlap:
                st.json(topic_overlap)
            else:
                st.warning("Not enough topics to calculate overlap.")

            # Generate Hindi Text Summary for TTS
            final_summary = f"{company_name} की खबरों का विश्लेषण: {comparative_analysis['Summary']}"
            st.subheader("🔊 Hindi Audio Summary")
            st.write(final_summary)

            # Generate Hindi TTS audio file
            audio_file = text_to_speech_hindi(final_summary)

            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format='audio/mp3')
            else:
                st.error("Failed to generate Hindi speech.")

        else:
            st.error("No articles found. Try a different company name.")

    else:
        st.warning("Please enter a company name.")
