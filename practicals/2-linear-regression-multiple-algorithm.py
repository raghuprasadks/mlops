import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

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
y_pred_linear = linear_model.predict(X_test)

# Create a Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Create a Lasso Regression model
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Print the Mean Squared Error of the models on the test set
print('Linear Regression MSE:', metrics.mean_squared_error(y_test, y_pred_linear))
print('Ridge Regression MSE:', metrics.mean_squared_error(y_test, y_pred_ridge))
print('Lasso Regression MSE:', metrics.mean_squared_error(y_test, y_pred_lasso))