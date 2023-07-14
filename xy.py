import numpy as np
from sklearn.model_selection import RepeatedKFold
X=np.array([[1,2],[3,4],[1,2],[3,4]])
rkf=RepeatedKFold(n_splits=2,n_repeats=2,random_state=1)
yy=rkf.split(X)
print(yy)
for train,test in yy:
    print("%s %s" % (train,test))