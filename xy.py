
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
# Step-1: Use Standard scaler to preprocess the data.
#   Step-2: Split the dataset in such a way that 20% data is taken for test cases.(set random state=10)
#   Step-3: Use the LinearRegression() estimator to predict the output.
# What is the R 2  score you got using LinearRegression estimator on test data.
# What is the value of cofficient associated with variable "s3"?
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

scaler = StandardScaler()
X = scaler.fit_transform(diabetes.data)
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(lr.coef_[diabetes.feature_names.index('s3')])


# Use RidgeCV to train your model and cross_validate to optimize it.

#    Hint: Set parameters for crossvalidation as mentioned below.
        #    CV=5
        #     alpha_list=np.logspace(-4, 0, num=20)
        #     scoring='neg_mean_absolute_error'

#    What is the optimum (average) "Mean absolute error" you got on training data?

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
import numpy as np

rcv = RidgeCV(cv=5, alphas=np.logspace(-4, 0, num=20), scoring='neg_mean_absolute_error')
rcv.fit(X_train, y_train)
y_pred = rcv.score(X_train, y_train)
# What is the optimum (average) "Mean absolute error" you got on training data?



print(y_pred)


from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
import numpy as np

# Create a RidgeCV model with cross-validation
rcv = RidgeCV(cv=5, alphas=np.logspace(-4, 0, num=20), scoring='neg_mean_absolute_error')

# Fit the model to the training data
rcv.fit(X_train, y_train)

# Perform cross-validation to get the negative mean absolute error scores
cv_results = cross_validate(rcv, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

# Extract the average mean absolute error from cross-validation results
print(cv_results)
average_mae = -np.mean(cv_results['test_score'])

print(average_mae)

