![](2023-07-21-21-55-24.png)

-  Decision Trees are actually non-parametric `supervised` learning methods used for classification and regression tasks. 

![](2023-07-21-21-57-51.png)

- What is gini impurity for binary classification?
    - Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
    - It is calculated by subtracting the sum of the squared probabilities of each class from one.
    - It favors larger partitions and easy to implement.
    - It is used in the CART (Classification and Regression Tree) algorithm for classification trees, which is the algorithm that we are using for our decision trees.
    - It works with categorical target variable “Success” or “Failure”.
![](2023-07-21-22-00-00.png)
![](2023-07-21-22-00-41.png)

![](2023-07-21-22-01-19.png)

a) This statement is generally not true for decision trees. Decision trees have the potential to be highly flexible and can fit complex patterns in the data. They are known for having a tendency to overfit the training data, which can lead to low bias and high variance. However, various techniques like pruning and limiting tree depth can be used to control overfitting and reduce variance.

- overfitting leads to high variance and low bias and so will the decision tree.
    - variance is the amount that the estimate of the target function will change if different training data was used
    - bias is the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model.
    - The bias-variance tradeoff is a central problem in supervised learning. Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well, but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit, but may underfit their training data, failing to capture important regularities.


C) Not necessarily.

![](2023-07-21-22-10-28.png)

`min_samples_split`: This parameter determines the minimum number of samples required to split an internal node further. Increasing this value can help prevent the tree from creating additional splits for small subsets of data, reducing the likelihood of overfitting.

`max_depth`: This parameter controls the maximum depth of the decision tree. Limiting the depth of the tree can prevent it from becoming too complex and overfitting the training data.

`min_samples_leaf`: This parameter sets the minimum number of samples required to be in a leaf node. Increasing this value can prevent the tree from creating leaves with very few samples, which can help in reducing overfitting.