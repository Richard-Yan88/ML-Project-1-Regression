#Importing Statements
import tensorflow as tf
import keras
import pandas as pd
import cv2
import sklearn


data = pd.read_csv(r'/Users/richardyan/Documents/COMPUTER SCIENCE/ML Project 1 Regression\student-mat.csv')
data = data [["G1","G2","G3", "studytime","", "failures", "absences"]]