import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

model = joblib.load('model.joblib')
df = pd.read_csv('test_data.csv')

pred = model.predict_proba(df)
print(pred) 
 