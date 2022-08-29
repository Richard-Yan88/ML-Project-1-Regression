#Importing Statements
import tensorflow as tf
import keras
import pandas as pd
import cv2
import sklearn
import sklearn.linear_model
from sklearn.utils import shuffle
import numpy as np


data = pd.read_csv('./student-mat.csv', sep =";")

data = data [["G1","G2","G3", "studytime", "failures", "absences"]]

print(data.head())

predict = "G3" # Label: What you are looking for or predicting

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)