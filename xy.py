# Import the California Housing dataset. 
#   Load the features and labels as numpy array. Split the data into training and test data in 4:1 proportion. What will be the size of training features?

from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()
X = cal_housing.data
y = cal_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)

