# bert_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import pickle

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\ML\DataSet\text.csv",encoding='ISO-8859-1',sep=",",
    quotechar='"',
    engine="python",
    on_bad_lines='skip')

# Encode labels
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

# Save label encoder
pickle.dump(le, open("label_encoder.pkl", "wb"))

# 2️⃣ Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label_enc"].tolist(), test_size=0.2, random_state=42
)

# 3️⃣ Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# 4️⃣ Convert to torch dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# 5️⃣ Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_))

# 6️⃣ Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 7️⃣ Train model
trainer.train()

# 8️⃣ Save model and tokenizer
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
print("✅ BERT model and tokenizer saved!")
