import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
from itertools import combinations

model_path = "models/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

embedding_model = SentenceTransformer("models/all-MiniLM-L6-v2", device='cpu')
kw_model = KeyBERT(embedding_model)

def summarize_articles(articles):
    summaries = []
    for article in articles:
        text = article['title']
        summary = summarizer(text, max_length=30, min_length=5, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries

def get_topics(summaries):
    topics_list = []
    for summary in summaries:
        keywords = kw_model.extract_keywords(summary, top_n=3)
        topics = [kw for kw, _ in keywords]
        topics_list.append(topics)
    return topics_list

def text_to_speech_hindi(text, filename="output.mp3"):
    try:
        tts = gTTS(text=text, lang='hi')
        tts.save(filename)
        return filename if os.path.exists(filename) else None
    except Exception as e:
        print(f"Error in TTS: {e}")
        return None

def generate_coverage_differences(summaries, topics):
    differences = []
    for (i, t1), (j, t2) in combinations(enumerate(topics), 2):
        unique_in_i = set(t1) - set(t2)
        unique_in_j = set(t2) - set(t1)
        if unique_in_i or unique_in_j:
            differences.append({
                "Comparison": f"Article {i+1} vs Article {j+1}",
                "Impact": (
                    f"Article {i+1} unique topics: {', '.join(unique_in_i) or 'None'}; "
                    f"Article {j+1} unique topics: {', '.join(unique_in_j) or 'None'}"
                )
            })
    return differences

def generate_topic_overlap(topics):
    all_topics = [set(t) for t in topics]
    if not all_topics:
        return {}

    common_topics = set.intersection(*all_topics)
    unique_topics = {f"Article {i+1}": list(t - common_topics) for i, t in enumerate(all_topics)}

    return {
        "Common Topics": list(common_topics),
        "Unique Topics": unique_topics
    }

