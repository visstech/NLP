# ================================================
# 🧠 News Article Analyzer – Multi-NLP Demo
# Author: Senthil + GPT Assistant
# ================================================

from transformers import pipeline
from keybert import KeyBERT

# -------------------------------
# 🔹 Step 1: Input Text
# -------------------------------
text = """
Apple Inc. announced record profits this quarter, driven by strong iPhone sales in Asia.
Experts believe the company’s revenue will continue to rise, as demand for smartphones remains high.
However, competition from Samsung and other Android manufacturers continues to be intense.
"""

# -------------------------------
# 🔹 Step 2: Sentiment Analysis
# -------------------------------
print("🟢 Sentiment Analysis:")
sentiment_analyzer = pipeline("sentiment-analysis")
sentiment_result = sentiment_analyzer(text)[0]
print(f"Sentiment: {sentiment_result['label']} (Score: {sentiment_result['score']:.2f})")
print("-" * 60)

# -------------------------------
# 🔹 Step 3: Named Entity Recognition (NER)
# -------------------------------
print("🟢 Named Entity Recognition:")
ner = pipeline("ner", grouped_entities=True)
for entity in ner(text):
    print(f"{entity['word']:<25} → {entity['entity_group']} (Score: {entity['score']:.2f})")
print("-" * 60)

# -------------------------------
# 🔹 Step 4: Text Summarization
# -------------------------------
print("🟢 Text Summarization:")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
print("Summary:", summary)
print("-" * 60)

# -------------------------------
# 🔹 Step 5: Keyword Extraction (KeyBERT)
# -------------------------------
print("🟢 Keyword Extraction:")
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(text, top_n=5)
for kw, score in keywords:
    print(f"{kw:<20} → Score: {score:.2f}")
print("-" * 60)

# -------------------------------
# 🔹 Step 6: Topic Classification (Zero-Shot)
# -------------------------------
print("🟢 Topic Classification:")
classifier = pipeline("zero-shot-classification")
candidate_labels = ["business", "technology", "sports", "politics", "health"]
topic_result = classifier(text, candidate_labels=candidate_labels)
print(f"Predicted Topic: {topic_result['labels'][0]} (Confidence: {topic_result['scores'][0]:.2f})")
print("-" * 60)

# -------------------------------
# ✅ Summary of All Insights
# -------------------------------
print("✅ Final Summary of NLP Insights:")
print(f"• Sentiment: {sentiment_result['label']} ({sentiment_result['score']:.2f})")
print(f"• Summary: {summary}")
print(f"• Top Keywords: {[kw for kw, _ in keywords]}")
print(f"• Topic: {topic_result['labels'][0]} ({topic_result['scores'][0]:.2f})")
print("\n🎉 Done! You’ve analyzed a news article with 5 NLP tasks.")
