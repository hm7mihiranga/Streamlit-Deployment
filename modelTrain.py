import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the iris dataset
iris = load_iris()


X, y = iris.data, iris.target


# Train a random forest classifier
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

#save the model
joblib.dump(model, 'model.joblib')