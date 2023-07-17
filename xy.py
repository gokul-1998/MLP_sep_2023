#   Step-1: Use Standard scaler to preprocess the data.
#   Step-2: Split the dataset in such a way that 20% data is taken for test cases.(set random state=10)
#   Step-3: Use the LinearRegression() estimator to predict the output.

# What is the R2 score you got using LinearRegression estimator on test data.

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

cal_housing = fetch_california_housing()
X = cal_housing.data
y = cal_housing.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 10)

lr = LinearRegression()
lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))




