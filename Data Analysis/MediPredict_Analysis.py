import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

df = pd.read_csv("Insurance_csv/insurance.csv")

df.head()

'''check for columns that are null'''

null_values=df.isnull().sum()
null_values_rows = df.isnull().any(axis=1).sum()
data=df.info()
print(data)

'''Get an overall view of the Stats in the Dataset'''
Describe_info=df.describe()
# print(Describe_info)


'''Convert Categorical columns into numerical values'''

df['sex']= df['sex'].map({'female':1,'male':0})
df['smoker']= df['smoker'].map({'yes':1,'no':0})


# ['southwest' 'southeast' 'northwest' 'northeast']
unique=df['region'].unique()
df['region'] = df['region'].map({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4})


'''Split X and Y(target). X is our features(independent variables) 
and the Y is the target(Dependent variable)'''

X = df.drop(['charges'],axis=1)
y = df['charges']


'''Train the dataset and split it into an 80/20 for 80% training data and 20'''

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

print(df)
print('X_train : ')
print(X_train.shape)
 
print('')
print('X_test : ')
print(X_test.shape)
 
print('')
print('y_train : ')
print(y_train.shape)
 
print('')
print('y_test : ')
print(y_test.shape)

'''Create an instance of linear regression and train the model'''

LinearR = LinearRegression()

LinearR.fit(X_train,y_train)

'''Predict the Y-pred for the target based on the independent variables(Features)'''

Y_pred = LinearR.predict(X_test)

'''Evaluate the performance of the Model'''

r2_score_value = r2_score(y_test,Y_pred)
mean_absolute_error_value = mean_absolute_error(y_test,Y_pred)
mean_squared_error_value = mean_squared_error(y_test,Y_pred)
interger = LinearR.intercept_
print("The R2 Score is: ", r2_score_value)
print("The mean_absolute_error_value is: ", mean_absolute_error_value)
print("The mean_squared_error_value is: ", mean_squared_error_value)
print("The interger is: ", interger)

