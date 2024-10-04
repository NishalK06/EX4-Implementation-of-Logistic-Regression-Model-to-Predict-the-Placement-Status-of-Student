# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. x is the feature matrix, and y is the target variable<p>
2.train_test_split splits the data.<p>
3.LogisticRegression builds the model.<p>
4.accuracy_score evaluates performance.<p>

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: K.Nishal
RegisterNumber:  2305001021
*/

import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()

data1=data.copy()
data1.head()

data1=data1.drop(['sl_no','salary'],axis=1)
data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred,x_test

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy_score)
print("Confusion matrix:\n",confusion)
print("\nClassification_report:\n",cr)

from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion,display_labels=[True,False])
cm_display.plot()

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification_Report:\n",cr)
```

## Output:
![image](https://github.com/user-attachments/assets/48b13283-2521-46ec-95d6-bbfdeecae8c6)
![image](https://github.com/user-attachments/assets/f9cb1ac9-b480-4b31-ab3b-a18316674f05)
![image](https://github.com/user-attachments/assets/ce5ac991-7ad1-4392-a66a-a09e6e098a6a)
![image](https://github.com/user-attachments/assets/1439e2cf-faa0-44de-a7d1-d9d2b4d5688f)
![image](https://github.com/user-attachments/assets/72179725-b853-49b6-839f-4ee23068e84d)
![image](https://github.com/user-attachments/assets/ed89bfa8-c219-4c62-9900-20bfdef62c76)
![image](https://github.com/user-attachments/assets/cf61cfbf-458d-4035-9109-07f8171cfcfc)
![image](https://github.com/user-attachments/assets/e7e4a3e3-2682-429a-82a7-625eb3cbe141)
![image](https://github.com/user-attachments/assets/ed3de07b-c41b-4509-b082-288777bf25c9)
![image](https://github.com/user-attachments/assets/20d809e4-340f-4be4-981a-63de7a38dafa)














## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
