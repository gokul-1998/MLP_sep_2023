![Alt text](image.png)
- All three are valid strategy parameter for dummy regressor

![Alt text](image-1.png)

![Alt text](image-2.png)

![Alt text](image-3.png)

![Alt text](image-4.png)

- Note : LinearRegression() is from `sklearn.linear_model`

![Alt text](image-7.png)

![Alt text](image-8.png)

![Alt text](image-9.png)

![Alt text](image-10.png)

![Alt text](image-11.png)

![Alt text](image-12.png)
- Generalization - for test data
- Empirical - for train data

![Alt text](image-13.png)

![Alt text](image-14.png)

- Dont know what R2 means in simple words other than formula? - watch this --> https://youtu.be/bMccdk8EdGo  highly recommended to understand what r2 score means.

![Alt text](image-37.png)
- evaluation and scoring will be from metrics like r2score,absolute mean error, etc...

![Alt text](image-38.png)

- mean squared log error , because inverse of exp is log

AQ 3.3

![Alt text](image-39.png)
![Alt text](image-40.png)
`
The ShuffleSplit function lies in the sklearn.model_selection module.`

![Alt text](image-41.png)
![Alt text](image-42.png)
![Alt text](image-43.png)

![Alt text](image-44.png)

```
from sklearn.model_selection import ShuffleSplit

# Create a ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Generate train-test splits
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Perform model fitting and evaluation on each split
    # ...

```

![Alt text](image-45.png)
![Alt text](image-46.png)

![Alt text](image-47.png)
![Alt text](image-48.png)