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
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RAMASRI K
RegisterNumber:  212224040267
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```

## Output:

<img width="1373" height="229" alt="image" src="https://github.com/user-attachments/assets/6169be44-6759-4f7a-8770-1f69bde75349" />

<img width="522" height="388" alt="image" src="https://github.com/user-attachments/assets/bdc79d41-8733-47c3-803d-6e9399da5e4f" />

<img width="323" height="272" alt="image" src="https://github.com/user-attachments/assets/f2d3a913-f1fb-48a2-b393-b8bf4c18d827" />

<img width="455" height="118" alt="image" src="https://github.com/user-attachments/assets/96216da1-2835-48b7-8132-06d775c0eff6" />

<img width="1350" height="239" alt="image" src="https://github.com/user-attachments/assets/e9aef591-10c3-456d-99c3-ba4909613fdd" />

<img width="1196" height="232" alt="image" src="https://github.com/user-attachments/assets/510efaf2-8917-45a4-a844-1a8f10537a3c" />

<img width="290" height="68" alt="image" src="https://github.com/user-attachments/assets/7d4b0e2e-df22-45d2-8bdc-7648e933c476" />

<img width="144" height="40" alt="image" src="https://github.com/user-attachments/assets/726552f9-1c50-4c42-b925-b2c2e2d6f4a4" />

<img width="821" height="598" alt="image" src="https://github.com/user-attachments/assets/ef2f2168-d05f-4926-8809-f026463cf531" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
