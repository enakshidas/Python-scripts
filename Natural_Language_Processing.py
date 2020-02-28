# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Blogs.tsv', delimiter = '\t', quoting = 3)

# Cleaning the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review_data = re.sub('[^a-zA-Z]', ' ', dataset['review_data'][i])
    review_data = review_data.lower()
    review_data = review_data.split()
    ps = PorterStemmer()
    review_data = [ps.stem(word) for word in review_data if not word in set(stopwords.words('english'))]
    review_data = ' '.join(review_data)
    corpus.append(review_data)

# Creating the Bag of Words 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Dividing the dataset into the Training set and Testing set
from sklearn.model_selection import train_test_split
X_training_data, X_testing_data, y_training_data, y_testing_data = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Applying Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_training_data, y_training_data)

# Predicting the Test set 
y_pred = classifier.predict(X_testing_data)

# Making the Confusion Matrix for testing accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_testing_data, y_pred)