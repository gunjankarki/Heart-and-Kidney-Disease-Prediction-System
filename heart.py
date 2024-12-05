import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('heart.csv')

X = data.drop(columns='target',axis=1)
Y = data['target']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
trainingdata_prediction_accuracy = accuracy_score(X_train_prediction,Y_train)

print(trainingdata_prediction_accuracy)
import pickle

# Assuming 'model' is your trained machine learning model

# Save the model to a .sav file
filename = 'heart_disease.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
