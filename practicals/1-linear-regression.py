import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Load the dataset
df = pd.read_csv('wine-quality.csv')

# Assume the last column is the target and the rest are features
X = df.iloc[:, :-1]
print("data type of X ",type(X))
y = df.iloc[:, -1]
print("data type of y ",type(y))

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions using the test set
y_pred = model.predict(X_test)

# Print the Mean Squared Error of the model on the test set
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

# Print the Mean Absolute Error of the model on the test set
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# Print the Root Mean Squared Error of the model on the test set
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Print the R^2 score of the model on the test set
print('R^2 Score:', metrics.r2_score(y_test, y_pred))