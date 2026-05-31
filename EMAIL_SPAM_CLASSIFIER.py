import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'message': [
        'Congratulations! You won a lottery',
        'Claim your free prize now',
        'Meeting at 10 AM tomorrow',
        'Please submit your assignment',
        'You have won a free vacation',
        'Project presentation is scheduled today'
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

X = df['message']
y = df['label']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


email = ["You have won a free mobile phone"]
email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)

print("Prediction:", prediction[0])
