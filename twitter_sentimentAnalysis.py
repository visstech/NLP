# ============================================
#  Twitter Sentiment Analysis using DistilBERT
#  Author: Senthil + GPT Assistant
# ============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)

# ======== STEP 1: Load the Dataset =========
train_path = r"C:\ML\DataSet\twitter_training.csv"
val_path = r"C:\ML\DataSet\twitter_validation.csv"

# Your dataset has no headers — so add manually
train_df = pd.read_csv(train_path, header=None, names=["ID", "Topic", "Sentiment", "Text"])
val_df = pd.read_csv(val_path, header=None, names=["ID", "Topic", "Sentiment", "Text"])

print("✅ Training Data:", train_df.shape)
print("✅ Validation Data:", val_df.shape)
print("\nSample rows:")
print(train_df.head())

# ======== STEP 2: Clean & Prepare Columns =========
train_df.columns = [col.strip() for col in train_df.columns]
val_df.columns = [col.strip() for col in val_df.columns]

# Drop missing values safely
train_df = train_df.dropna(subset=['Text', 'Sentiment'])
val_df = val_df.dropna(subset=['Text', 'Sentiment'])

# ======== STEP 3: Encode Sentiment Labels =========
encoder = LabelEncoder()
train_df['labels'] = encoder.fit_transform(train_df['Sentiment'])
val_df['labels'] = encoder.transform(val_df['Sentiment'])

print("\nLabel Mapping:", dict(enumerate(encoder.classes_)))

# ======== STEP 4: Convert to Hugging Face Dataset =========
train_dataset = Dataset.from_pandas(train_df[['Text', 'labels']])
val_dataset = Dataset.from_pandas(val_df[['Text', 'labels']])

# ======== STEP 5: Tokenize Text (FIXED padding) =========
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(
        batch['Text'],
        padding="max_length",   # ✅ FIXED: ensures uniform 128-length tensors
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Required columns for Trainer
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# ======== STEP 6: Load DistilBERT Model =========
num_labels = len(encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)

# ======== STEP 7: Define Training Arguments =========
training_args = TrainingArguments(
    output_dir='./twitter_sentiment_model',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,  # set 2 for faster testing
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    save_total_limit=1
)

# ======== STEP 8: Initialize Trainer =========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ======== STEP 9: Train the Model =========
print("\n🚀 Training Started...")
trainer.train()
print("✅ Training Completed!")

# ======== STEP 10: Save Model & Tokenizer =========
model.save_pretrained('./twitter_sentiment_model')
tokenizer.save_pretrained('./twitter_sentiment_model')
print("✅ Model and Tokenizer Saved Successfully!")

# ======== STEP 11: Test on New Sentences =========
print("\n🔍 Testing the model on sample tweets...")

classifier = pipeline("text-classification", model='./twitter_sentiment_model', tokenizer='./twitter_sentiment_model')

test_texts = [
    "I love how smooth the new Microsoft update is!",
    "The Amazon delivery was so late and disappointing.",
    "Facebook ads are getting annoying lately.",
    "This new app looks interesting!"
]

for text in test_texts:
    result = classifier(text)[0]
    # Convert LABEL_0 → integer → actual sentiment name
    label_id = int(result['label'].split('_')[-1])
    label_name = encoder.classes_[label_id]
    print(f"{text} → {label_name} (score: {result['score']:.2f})")

print("\n🎉 Done! Your fine-tuned Twitter sentiment model is ready.")
