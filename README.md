# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score
## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DHIVYA DHARSHINI P
RegisterNumber:  212225220028


import pandas as pd

data = pd.read_csv(r"C:\Users\acer\Downloads\Employee.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nNull Values:")
print(data.isnull().sum())

print("\nLeft Employee Count:")
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident"
]]

y = data["left"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

sample = pd.DataFrame([[0.5, 0.8, 9, 260, 6, 0]],
columns=[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident"
])

print("\nPrediction:")
print(dt.predict(sample))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=["Stay", "Left"],
    filled=True
)

plt.show()
```

## Output:
<img width="935" height="466" alt="image" src="https://github.com/user-attachments/assets/c9fc6671-9c1a-467e-9115-06fb8ceb578d" />
<img width="957" height="700" alt="image" src="https://github.com/user-attachments/assets/6090b062-c72a-499d-8c4f-75dd0739d452" />
<img width="537" height="260" alt="image" src="https://github.com/user-attachments/assets/ded37283-8982-4da9-802c-2251a3cb5d7b" />
<img width="1387" height="787" alt="image" src="https://github.com/user-attachments/assets/093dc2c1-3e95-4ea8-a74b-4d2ca37c08d9" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
