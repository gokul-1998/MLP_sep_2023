![](2023-07-23-20-42-49.png)

![](2023-07-23-20-44-15.png)

![](2023-07-23-20-51-14.png)
- What is SAG(Stochastic Average Gradient)?
    - https://www.youtube.com/watch?v=3LQbbvt5Ass
    - ![](2023-07-23-22-44-26.png)
- What is Stochastic Gradient Descent?
    - https://youtu.be/vMh0zPT0tLI
- What is Logistic Regression?
    - https://youtu.be/yIYKR4sgzI8
    - it is a classification algorithm
    - it is a linear classifier (it draws a line to separate the data)
    - it is a discriminative model (it models the decision boundary between classes)
    - it is a parametric model (it assumes a functional form for the decision boundary)
    - it is a probabilistic model (it models the probability of a data point belonging to a class)
- What is logestic regression with gradient descent?
    - https://medium.com/analytics-vidhya/logistic-regression-with-gradient-descent-explained-machine-learning-a9a12b38d710
    



![](2023-07-23-22-49-13.png)
![](2023-07-23-22-49-40.png)    
```
from sklearn.utlis.multiclass import type_of_target
import numpy as np
print(type_of_target(np.array([[0,1],[1,1]])))
print(type_of_target([1.0,0.0,3.0]))
print(type_of_target(['a','b','c']))

```
```
multilabel-indicator
multiclass
multiclass
```
![](2023-07-23-22-58-39.png)
Explore more

![](2023-07-23-22-59-17.png)
![](2023-07-23-23-00-02.png)
![](2023-07-23-23-02-02.png)
![](2023-07-23-23-02-30.png)
![](2023-07-23-23-02-45.png)
```
from sklearn.feature_extraction.text import CountVectorizer
corpus=['Hello Hello World great']
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(corpus)
print(X.toarray())
```
![](2023-07-23-23-06-14.png)

![](2023-07-23-23-06-38.png)
![](2023-07-23-23-08-01.png)
= 0.5 + 0.5 + 0 + 1 / 4 = 0.5

![](2023-07-23-23-09-51.png)
```
from sklearn.metrics import precision_score
y_true=[1,1,0,1,0,0,1,0,1]
y_pred=[0,1,0,1,0,1,1,1,1]
print(precision_score(y_true,y_pred))
```
![](2023-07-23-23-13-00.png)
TP=3
FP=2
precision=3/5=0.6

![](2023-07-23-23-14-41.png)
![](2023-07-23-23-15-57.png)
![](2023-07-23-23-16-09.png)

![](2023-07-23-23-16-47.png)
![](2023-07-23-23-17-06.png)
![](2023-07-23-23-17-23.png)
a) n_estimators should  be in quotes

b) param_distribution should be a dict and not a list, also invalid syntax for a list,no ':' required

c) param_distribution should be a dict and not a list , also invalid syntax for a list,no ':' required

d) correct
```

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


rcf= RandomForestClassifier()
params_distributions = {'n_estimators': [50,100, 150],
                        'max_depth': [5,10,15],
                        'min_samples_leaf': [2,4,6]}
random_search = RandomizedSearchCV(rcf, param_distributions=params_distributions, cv=5)
```

![](2023-07-23-23-23-47.png)
![](2023-07-23-23-24-55.png)
![](2023-07-23-23-25-13.png)
```
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
param_grid=[{'max_depth':range(1,20,2)}]
gs=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=10)
gs.fit(X,y)
```
- the range function goes for 10 iter, and cv=10
- so 10 * 10 = 100 

![](2023-07-24-01-21-09.png)
?

?

?

![](2023-07-24-01-22-08.png)

why?


![](2023-07-24-01-22-53.png)

?

?

?

![](2023-07-24-01-23-43.png)

?

?

?

![](2023-07-24-01-24-54.png)

?

?

?

![](2023-07-24-01-33-35.png)

![](2023-07-24-01-33-56.png)

![](2023-07-24-01-34-23.png)

![](2023-07-24-01-34-43.png)

![](2023-07-24-01-35-06.png)

