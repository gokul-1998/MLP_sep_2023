from sklearn.metrics import confusion_matrix
y_true=['cat','ant','cat','cat','ant']
y_pred=['ant','ant','cat','cat','ant']
print(confusion_matrix(y_true,y_pred,labels=['ant','cat']))
# to ploy confusion matrix in a more readable format
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
mat=confusion_matrix(y_true,y_pred,labels=['ant','cat'])
sns.heatmap(mat,square=True,annot=True,cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()
