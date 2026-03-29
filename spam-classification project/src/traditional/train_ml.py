from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def train_ml_models(data):
    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    return lr, nb, X_test, y_test, vectorizer