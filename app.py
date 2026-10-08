from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ------------------------------------------------------------
# TEXT CLEANING
# Same preprocessing as your Colab project
# ------------------------------------------------------------

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------
# LOAD DATASET AND TRAIN MODEL
# Same TF-IDF + Multinomial Naive Bayes pipeline
# ------------------------------------------------------------

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["content"] = df["title"] + " " + df["text"]

df["clean_text"] = df["content"].apply(clean_text)

df = df[df["clean_text"].str.len() > 0]
df = df.reset_index(drop=True)

X = df["clean_text"]
y = df["label"]

# Use the same 80/20 split and random_state as your notebook.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = tfidf.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

def predict_news(news):
    cleaned_news = clean_text(news)
    news_tfidf = tfidf.transform([cleaned_news])

    prediction = model.predict(news_tfidf)[0]
    probabilities = model.predict_proba(news_tfidf)[0]

    fake_probability = probabilities[0] * 100
    true_probability = probabilities[1] * 100

    if prediction == 0:
        label = "FAKE NEWS"
        status = "fake"
    else:
        label = "TRUE NEWS"
        status = "true"

    return {
        "label": label,
        "status": status,
        "fake_probability": round(fake_probability, 2),
        "true_probability": round(true_probability, 2)
    }


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data.get("news", "").strip()

    if not news:
        return jsonify({
            "error": "Please enter a news article first."
        }), 400

    return jsonify(predict_news(news))


if __name__ == "__main__":
    app.run(debug=True)
