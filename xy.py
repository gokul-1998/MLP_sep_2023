from sklearn.linear_model import Perceptron
X=[[2,2],[2,4],[4,4],[4,2]]

y=[1,2,1,2]
clf=Perceptron(tol=None,random_state=0)
clf.fit(X,y)
print(clf.score(X,y))