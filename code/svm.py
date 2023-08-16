import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint
import pickle
import cv2
import numpy as np
import csv

# Load the dataset
data = pd.read_csv("leaf_features_aug.csv")

print(len(data))

# data = data.drop(columns=['Image Name'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[["Leaf Count", "Leaf Area Covered"]], data["Weight"], test_size=0.4, random_state=42)


# # Define the parameter grid for tuning
# param_grid = {
#     'C': [0.1],         # Regularization parameter
#     'kernel': ['sigmoid'],   # Kernel function
#     'gamma': ['scale', 'auto']     # Kernel coefficient for 'rbf'
# }


# Define the parameter grid for tuning
param_grid = {
    'C': [100],         # Regularization parameter
    'kernel': ['rbf'],   # Kernel function
    'gamma': ['scale', 'auto'],
}

# Initialize the Support Vector Regression model
svm_model = SVR()

# Initialize GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator from grid search
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Make predictions using the best estimator
y_pred = best_estimator.predict(X_test)

# Compute R^2 score and mean squared error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the best hyperparameters, R^2 score, and mean squared error
print("Best Hyperparameters:", best_params)
print(f"R^2 Score: {r2:.3f}")
print(f"Mean Squared Error: {mse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")

# Plot the actual biomass values and predicted biomass values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Biomass (grams)')
plt.ylabel('Predicted Biomass (grams)')
plt.title('Biomass Estimation - Support Vector Machine')
# plt.ylim(0.05, 0.15)
plt.show()

# Plot the actual biomass values and predicted biomass values
plt.scatter(data["Leaf Area Covered"], data["Weight"])
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Leaf area covered (Pixels)')
plt.ylabel('Actual Biomass (grams)')
plt.title('Area Biomass plot - Support Vector Machine')
# plt.ylim(0.05, 0.15)
plt.show()