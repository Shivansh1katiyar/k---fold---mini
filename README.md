import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    "area": [1000,1200,1500,1800,2000,2300,2500,2700],
    "bedrooms": [2,2,3,3,3,4,4,4],
    "age": [10,8,7,6,5,4,3,2],
    "price": [200000,220000,300000,340000,360000,420000,460000,480000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["area","bedrooms","age"]].values
y = df["price"].values

# KFold setup
kf = KFold(n_splits=4, shuffle=True, random_state=42)

model = LinearRegression()

scores = []

# K-Fold Loop
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)
    scores.append(score)

# Results
print("R2 Scores for each fold:", scores)
print("Average R2 Score:", np.mean(scores))
