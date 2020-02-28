# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Budget_Analysis.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Putting Label Encoder and One Hot Encoder for categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training data and Testing data
from sklearn.model_selection import train_test_split
X_training_data, X_testing_data, y_training_data, y_testing_data = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_training_data, y_training_data)

# Predicting the Test set results
y_pred = regressor.predict(X_testing_data)