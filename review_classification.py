import gensim.downloader as api
import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

wv = api.load('word2vec-google-news-300')
nlp = spacy.load("en_core_web_lg")


# Preprocess text, then vectorize it
def preprocess_and_vectorize(text):
    doc = nlp(text)    # create an nlp object out of text

    # Preprocess
    # Take out stop words and punctuation, get the lemma of other words
    filtered_tokens = []    # create a list for the filtered tokens
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        filtered_tokens.append(token.lemma_)    # append the lemma of the word to the list

    # Vectorize
    # Get mean vector of list of words -- embedding of the entire paragraph
    return wv.get_mean_vector(filtered_tokens)


# Read the data
df = pd.read_csv("reviews_expanded.csv")

# Preprocess and vectorize the text entries in the news column
# Put the vector embeddings in a new column titled 'vector'
df['vector'] = df['review'].apply(lambda text: preprocess_and_vectorize(text))

# Use 20% of the reviews for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(
    df.vector.values,
    df.label,
    test_size=0.2,
    random_state=2022,
    stratify=df.label
)

# Train and test the model
# Create 2D arrays from the train and test datasets
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)

# 1. Create a GradientBoosting model object
clf = GradientBoostingClassifier()

# 2. Fit with train embeddings
clf.fit(X_train_2d, y_train)

# 3. Get the predictions for the test embeddings
y_pred = clf.predict(X_test_2d)

# 4. Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Actual Value')