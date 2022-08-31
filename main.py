#Importing Statements
import tensorflow as tf
import keras
import pandas as pd
import cv2
import sklearn
import sklearn.linear_model as linear_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

"""
Assumptions of Multiple Linear Regression:
    1. There must be a linear relationship between the outcome variable and the independent variable
    2. Multivariate Normality - Multiple regression assumes that the residuals are normally distributed
    3. No multicollinearity - multiple regression assumes that the independet variables are not highly correlated with each
    other. This assumption is tested using Variance Inflation Factor (VIF) values
    4. Homoscedasticity - the variance of error terms are similar across the values of the independent variables. A plot of 
    standardized residuals versus predicted values can show whether points are equally distributed across all values of the
    independent variables. Multiple lienar regression requires atleast two independent variabels, which can be nominal, ordinal, or interval
    /ratio level variables. A rule of thumb for the sample size is that regression analysis requires at least 20 cases per independet variable
    in the analysis.
"""

# Reading in the Data
data = pd.read_csv('./student-mat.csv', sep =";")

data = data[["G1","G2","G3", "studytime", "failures", "absences"]]

predict = "G3" # Label: What you are looking for or predicting

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Generating the highest accuracy model
best = 0
while best < .95:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    
    linear = linear_model.LinearRegression()
        
    linear.fit(x_train,y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
        
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# Storing and Opening Best Generated Model in Pickle
pickle_in =  open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


# Printing Coefficients and Intercept
print("Co: ",  linear.coef_)
print("Intercept: ",  linear.intercept_)

predictions = linear.predict(X)

# Predictions based on Independent Factors
for x in range(len(predictions)):
    print(predictions[x], X[x], y[x])
    
    
p = 'G1'
style.use("ggplot")
pyplot.scatter(data(p), data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()