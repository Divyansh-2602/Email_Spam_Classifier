# train.py
import pandas as pd
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load dataset (update path if needed)
df = pd.read_csv("mail_data.csv")   # CSV with 'Category' and 'Message' columns

# Encode target: spam=1, ham=0
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Preprocess messages
df['transformed'] = df['Message'].apply(transform_text)

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df['transformed']).toarray()
y = df['Category'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
print("✅ Model and Vectorizer saved!")
