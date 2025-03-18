import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from gtts import gTTS

# Load summarizer components locally (correct way)
model_path = "models/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

# Load sentence-transformers model locally
embedding_model = SentenceTransformer("models/all-MiniLM-L6-v2", device='cpu')
kw_model = KeyBERT(embedding_model)

def summarize_articles(articles):
    summaries = []
    for article in articles:
        text = article['title']
        summary = summarizer(
            text,
            max_length=30,
            min_length=5,
            do_sample=False
        )[0]['summary_text']
        summaries.append(summary)
    return summaries

def get_topics(summaries):
    topics_list = []
    for summary in summaries:
        keywords = kw_model.extract_keywords(summary, top_n=3)
        topics = [kw for kw, score in keywords]
        topics_list.append(topics)
    return topics_list

def text_to_speech_hindi(text, filename="output.mp3"):
    try:
        # Generate the speech
        tts = gTTS(text=text, lang='hi')
        tts.save(filename)

        # Optional: confirm the file exists
        if os.path.exists(filename):
            print(f"Audio file saved as {filename}")
            return filename
        else:
            print("Failed to save audio file.")
            return None

    except Exception as e:
        print(f"Error in TTS: {e}")
        return None
    
def generate_coverage_differences(summaries, topics):
    differences = []

    # Basic validation
    if len(summaries) < 2:
        return differences  # Need at least two articles to compare

    # Compare first two articles (you can loop over more later)
    comparison = {
        "Comparison": f"Article 1 focuses on: '{summaries[0]}', while Article 2 covers: '{summaries[1]}'.",
        "Impact": (
            f"Article 1 emphasizes topics like {topics[0]}, which may create a positive impression. "
            f"Article 2 discusses {topics[1]}, potentially raising concerns or providing a different viewpoint."
        )
    }

    differences.append(comparison)

    return differences
def generate_topic_overlap(topics):
    if len(topics) < 2:
        return {}

    # Find common and unique topics between the first two articles
    common_topics = list(set(topics[0]).intersection(set(topics[1])))
    unique_topics_1 = list(set(topics[0]).difference(set(topics[1])))
    unique_topics_2 = list(set(topics[1]).difference(set(topics[0])))

    return {
        "Common Topics": common_topics,
        "Unique Topics in Article 1": unique_topics_1,
        "Unique Topics in Article 2": unique_topics_2
    }
