from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pandas as pd

titanic = pd.read_csv("https://raw.githubusercontent.com/Evanmj7/Decision-Trees/master/titanic.csv",
                      index_col=0)

# Let's ensure the columns we want to treat as continuous are indeed continuous by using pd.to_numeric
# The errors = 'coerce' keyword argument will force any values that cannot be
# cast into continuous variables to become NaNs.
continuous_cols = ['age', 'fare']
for col in continuous_cols:
    titanic[col] = pd.to_numeric(titanic[col], errors='coerce')

# Set categorical cols & convert to dummies
cat_cols = ['sex', 'pclass']
for col in cat_cols:
    titanic[col] = titanic[col].astype('category').cat.codes

# Clean the dataframe. An alternative would be to retain some rows with missing values by giving
# a special value to nan for each column, eg by imputing some values, but one should be careful not to
# use information from the test set to impute values in the training set if doing this. Strictly speaking,
# we shouldn't be dropping the nans from the test set here (as we pretend we don't know what's in it) - but
# for the sake of simplicity, we will.
titanic = titanic.dropna()

# Create list of regressors
regressors = continuous_cols + cat_cols
# Predicted var
y_var = ['survived']

# Create a test (25% of data) and train set
train, test = train_test_split(titanic, test_size=0.25)

# Now let's create an empty decision tree to solve the classification problem:
clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5,
                                  ccp_alpha=0.01)
# The last option, ccp_alpha, prunes low-value complexity from the tree to help
# avoid overfitting.

# Fit the tree with the data
clf.fit(train[regressors], train[y_var])

# Let's take a look at the tree:
tree.plot_tree(clf)

# How does it perform on the train and test data?
train_accuracy = round(clf.score(train[regressors], train[y_var]), 4)
print(f'Accuracy on train set is {train_accuracy}')

test_accuracy = round(clf.score(test[regressors], test[y_var]), 4)
print(f'Accuracy on test set is {test_accuracy}')

# Show the confusion matrix
plot_confusion_matrix(clf, test[regressors], test[y_var])