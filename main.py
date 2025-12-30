texts = [
    "I love programming.",
    "Python is a great language.",
    "Artificial Intelligence is the future.",
    "OpenAI develops advanced AI models.",
    "ChatGPT is an AI language model.",
    "Worst Purchase Ever.",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
]


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
print(X.toarray())

model = MultinomialNB()
model.fit(X, labels)
print("Model trained!")


test_text = ["I really love this"]
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)
print("Prediction:", prediction[0])


user_input = input("Write a sentence: ")
test_vector = vectorizer.transform([user_input])
prediction = model.predict(test_vector)
print("AI thinks:", prediction[0])

