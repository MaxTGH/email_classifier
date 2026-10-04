"""Step 2: a simple baseline BEFORE any deep learning.

TF-IDF turns each message into word-importance scores; logistic regression
draws lines between the classes. Trains in about a second. The transformer has
to beat this number to be worth the extra complexity.
"""
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent  # project folder

df = pd.read_csv(ROOT / "data" / "emails.csv")
# Same split (seed + stratify) as finetune.py so the scores are comparable.
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Dumbest possible model: always guess the most common label.
dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
print(f"Always-guess-majority accuracy: {dummy.score(test['text'], test['label']):.3f}\n")

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000), #linear but uses softmax (which is hte logistc part)
)
model.fit(train["text"], train["label"])
print("TF-IDF + logistic regression:")
print(classification_report(test["label"], model.predict(test["text"]), digits=3))

# Peek inside: which words push a message toward each label?
vec, clf = model[0], model[1]
words = vec.get_feature_names_out()
for i, label in enumerate(clf.classes_):
    top = clf.coef_[i].argsort()[-8:][::-1]
    print(f"{label:>10}: {', '.join(words[top])}")
