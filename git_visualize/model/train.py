import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from common.utils import clean_text

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
true_df["label"] = 1
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
data["text"] = data["text"].apply(clean_text)

X_train, _, y_train, _ = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
