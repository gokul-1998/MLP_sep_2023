
![](2023-07-24-01-39-53.png)
![](2023-07-30-02-17-54.png)
# HOW??
```
from sklearn.linear_model import Perceptron
X=[[0,1],[0,2],[2,0],[3.5,3.5]]

y=[1,2,3,4]
clf=Perceptron()
clf.fit(X,y)
print(clf.score(X,y))
# the answer is 1.0 which means the model is perfect,
# and it is not a good model, because it is overfitting
# it means the data is linearly separable
# linearly separable means the data can be separated by a line
# or a hyperplane
# here the points are linearly separable and the line is
# x1+x2=3

```

# more example 
```
X = [[0, 1], [0, 2], [2, 0], [3.5, 3.5], [3, 3], [4, 4]]
y = [1, 2, 3, 4, 1, 2]

# Create and train the Perceptron model
clf = Perceptron()
clf.fit(X, y)

# Predict labels using the trained model
y_pred = clf.predict(X)

# Calculate the number of correct predictions
correct_predictions = sum(y_pred == y)

# Calculate accuracy
accuracy = correct_predictions / len(y)
print(accuracy)

```
![](2023-08-02-00-33-53.png)
![](2023-08-02-00-35-14.png)

![](2023-07-24-01-40-35.png)

![](2023-08-02-00-37-31.png)
- What is RidgeClassifier?

![](2023-08-02-00-38-53.png)
![](2023-08-02-00-39-37.png)
![](2023-08-02-00-39-55.png)
![](2023-07-24-01-40-55.png)

![](2023-08-02-00-41-35.png)
-  fraction of samples = subsample

![](2023-08-02-00-43-35.png)
- https://youtu.be/VverL5SLowQ
- https://www.youtube.com/watch?v=3CC4N4z3GJc&pp=ygUZR3JhZGllbnRCb29zdGluZ1JlZ3Jlc3Nvcg%3D%3D

![](2023-07-24-01-41-42.png)
- most frequent will take the mode of the values in the data and predict it as the output for any input value

![](2023-07-24-01-42-17.png)
- CategoricalNB is used for categorical data
- GaussianNB is used for numerical data

![](2023-07-24-01-43-13.png)

![](2023-07-24-01-43-30.png)

![](2023-07-24-01-43-43.png)

![](2023-07-24-01-44-01.png)

![](2023-07-24-01-44-17.png)

![](2023-07-24-01-44-47.png)

![](2023-07-24-01-45-06.png)

![](2023-07-24-01-45-19.png)

![](2023-07-24-01-45-36.png)

![](2023-07-24-01-46-02.png)

![](2023-07-24-01-46-17.png)

![](2023-07-24-01-46-29.png)

![](2023-07-24-01-46-42.png)

![](2023-07-24-01-47-13.png)

![](2023-07-24-01-47-34.png)

![](2023-07-24-01-48-19.png)

![](2023-07-24-01-48-34.png)

![](2023-07-24-01-48-48.png)

![](2023-07-24-01-49-54.png)

- Ans = 4