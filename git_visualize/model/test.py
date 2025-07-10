import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from common.utils import clean_text

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
true_df["label"] = 1
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
data["text"] = data["text"].apply(clean_text)

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

X = vectorizer.transform(data["text"])
y = data["label"]

y_pred = model.predict(X)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
