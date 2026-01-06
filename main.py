import joblib
import os


texts = [
    "I love this product",
    "This is amazing",
    "Best experience ever",
    "Absolutely fantastic",
    "I am very happy with this",
    "This made my day",

    "I hate this",
    "This is terrible",
    "Worst purchase",
    "Absolutely horrible",
    "Very disappointed",
    "I will never buy this again"
]

labels = [
    "positive", "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative", "negative"
]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(texts)
# print(X.toarray())
#
# model = MultinomialNB()
# model.fit(X, labels)
# print("Model trained!")
#
#
# test_text = ["I absolutely love this"]
# test_vector = vectorizer.transform(test_text)
# prediction = model.predict(test_vector)
# print("Prediction:", prediction[0])
#
#
# user_input = input("Write a sentence: ")
# test_vector = vectorizer.transform([user_input])
# prediction = model.predict(test_vector)
# print("AI thinks:", prediction[0])


def train_model(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    return model, vectorizer



def load_model():
    model =  joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer


if os.path.exists("model.joblib"):
    model, vectorizer = load_model()
    print("Model loaded from disk.")
else:
    model, vectorizer = train_model(texts, labels)
    print("Model trained and saved to disk.")