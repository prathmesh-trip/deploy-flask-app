import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import pickle

loan_data=pd.read_csv('loan_train.csv')
train_data = loan_data.drop('Unnamed: 0', axis = 1)

##Replace missing values -- Continuous Variable

train_data.Credit_History.fillna(train_data['Credit_History'].median(), inplace = True)
train_data.LoanAmount.fillna(train_data['LoanAmount'].median(), inplace = True)
train_data.Loan_Amount_Term.fillna(train_data['Loan_Amount_Term'].median(), inplace = True)

##Replace missing values -- Categorical Variable

train_data.Self_Employed.fillna(train_data['Self_Employed'].value_counts().index[0], inplace = True)
train_data.Gender.fillna(train_data['Gender'].value_counts().index[0], inplace = True)
train_data.Dependents.fillna(train_data['Dependents'].value_counts().index[0], inplace = True)
train_data.Married.fillna(train_data['Married'].value_counts().index[0], inplace = True)

##Splitting the target and feature variables
X_train = train_data.drop(['Loan_Status','Loan_ID'], axis = 1)
y_train = train_data['Loan_Status']


##One hot encoding
X_train = pd.get_dummies(X_train, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Property_Area'])

##Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model_final = LogisticRegression()
model_final.fit(X_train,y_train)

pickle.dump(model_final, open('model.pkl','wb'))


## calling the model to predict 
data =[]

item = [4547, 0, 115, 360, 1,'Male','No','0','Graduate','No','Semiurban']

##ApplicantIncome
data.append(item[0])
##CoApplicantIncome
data.append(item[1])
##LoanAmount
data.append(item[2])
##LoanAmountTerm
data.append(item[3])
##Credit_History
data.append(item[4])

##Gender
if item[5] == 'Male':
    data.append(0)
    data.append(1)
else:
    data.append(1)
    data.append(0)
##Married
if item[6] == 'No':
    data.append(1)
    data.append(0)
else:
    data.append(0)
    data.append(1)
##Dependents
if item[7] == '0':
    data.append(1)
    data.append(0)
    data.append(0)
    data.append(0)
elif item[7] == '1':
    data.append(0)
    data.append(1)
    data.append(0)
    data.append(0)
elif item[7] == '2':
    data.append(0)
    data.append(0)
    data.append(1)
    data.append(0)
else:
    data.append(0)
    data.append(0)
    data.append(0)
    data.append(1)
##Graduation
if item[8] == 'Graduate':
    data.append(1)
    data.append(0)
else:
    data.append(0)
    data.append(1)

##Self_Employed
if item[9] == 'No':
    data.append(1)
    data.append(0)
else:
    data.append(0) 
    data.append(1)

##Property
if item[10] == 'Rural':
    data.append(1)
    data.append(0)
    data.append(0)
elif item[10] == 'Semiurban':
    data.append(0)
    data.append(1)
    data.append(0)
else :
    data.append(0)
    data.append(0)
    data.append(1)

print(data)

# this is single sample
print("The final result is ",model_final.predict([data]))
