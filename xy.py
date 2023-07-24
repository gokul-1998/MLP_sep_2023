
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


rcf= RandomForestClassifier()
params_distributions = ['n_estimators': [50,100, 150],
                        'max_depth': [5,10,15],
                        'min_samples_leaf': [2,4,6]]
random_search = RandomizedSearchCV(rcf, param_distributions=params_distributions, cv=5)