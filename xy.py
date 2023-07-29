from sklearn.linear_model import Perceptron
X=[(0,1),(0,2),(2,0),(3.5,3.5)]
y=[1,2,3,4]
clf=Perceptron()
clf.fit(X,y)
print(clf.score(X,y))


# ans = 1.0

- this is because the perceptron is a linear classifier, and the data is linearly separable