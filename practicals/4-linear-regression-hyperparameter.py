import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


# Load the dataset
df = pd.read_csv('wine-quality.csv')

# Assume the last column is the target and the rest are features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Create a Decision Tree Regression model
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X_train, y_train)

# Create a Support Vector Regression model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)
"""
# Use cross-validation to select the best model
models = [linear_model, tree_model, svm_model]
best_model = models[np.argmin([mean_squared_error(y_test, model.predict(X_test)) for model in models])]

print('Best model:', type(best_model).__name__)
"""

# Define the hyperparameters and their values
linear_params = {'fit_intercept': [True, False]}
tree_params = {'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10]}
svm_params = {'C': [0.1, 1, 10, 100], 'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5]}

# Create a GridSearchCV object for each model
linear_search = GridSearchCV(linear_model, linear_params, cv=5, scoring='neg_mean_squared_error')
tree_search = GridSearchCV(tree_model, tree_params, cv=5, scoring='neg_mean_squared_error')
svm_search = GridSearchCV(svm_model, svm_params, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV objects
linear_search.fit(X_train, y_train)
tree_search.fit(X_train, y_train)
svm_search.fit(X_train, y_train)

# Print the best parameters for each model
print('Best parameters for Linear Regression:', linear_search.best_params_)
print('Best parameters for Decision Tree Regression:', tree_search.best_params_)
print('Best parameters for Support Vector Regression:', svm_search.best_params_)
