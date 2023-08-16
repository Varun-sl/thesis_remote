import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

# data = data.drop(columns=['Image Name'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[["Leaf Count", "Leaf Area Covered"]], data["Weight"], test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [None, 5, 10, 20, 50],
    'min_samples_split': [2, 3, 4, 5, 6]
}

# param_grid = {
#     'n_estimators': [1500],
#     'max_depth': [None],
#     'min_samples_split': [2]
# } aug

# param_grid = {
#     'n_estimators': [1000],
#     'max_depth': [5],
#     'min_samples_split': [6]
# }

# param_grid = {
#     'n_estimators': [1500],
#     'max_depth': [10],
#     'min_samples_split': [6]
#}


# Create a Random Forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

# Train the best model on the entire training set
# rf.fit(X_train, y_train)
best_rf.fit(X_train, y_train)

# Save the trained model to a file
filename = 'rf_model_new.pkl'
pickle.dump(best_rf, open(filename, 'wb'))

# Predict the biomass for the test set using the best model
y_pred = best_rf.predict(X_test)

# Compute R^2 score and mean squared error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the best hyperparameters, R^2 score, and mean squared error
print("Best Hyperparameters:", best_params)
print(f"R^2 Score: {r2:.3f}")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Mean Absolute Error: {mae:.3f}")

# Plot the actual biomass values and predicted biomass values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Biomass (grams)')
plt.ylabel('Predicted Biomass (grams)')
plt.title('Biomass Estimation - Random Forest')
plt.show()






# # Load the trained RF model from file
# model = pickle.load(open('rf_model_new.pkl', 'rb'))
#
#
# def extract_leaf_features(sharpened_image):
#     # Load the image
#     image = cv2.imread(sharpened_image)
#
#     # Convert the cropped image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Perform adaptive thresholding to obtain a binary image
#     thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
#
#     # Find contours of the leaves
#     contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Initialize variables to store leaf count and leaf area
#     leaf_count = 0
#     leaf_area = 0
#
#     # Calculate leaf count and area for each contour
#     for contour in contours:
#         # Approximate the contour to reduce the number of points
#         epsilon = 0.06 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#
#         # Exclude contours with very small areas
#         if cv2.contourArea(approx) > 400:
#             leaf_count += 1
#             leaf_area += cv2.contourArea(approx)
#
#     #Export the features to a CSV file
#     csv_file = "leaf_features_single.csv"
#     with open(csv_file, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Leaf_Count", "leaf_area"])
#         writer.writerow([leaf_count, leaf_area])
#
#     print("Leaf features exported to:", csv_file)
#
#     # Return the leaf count and leaf area
#     return leaf_count, leaf_area, thresholded
#
#
# image_path = "../images/cropped/76_3.jpg"
#
# leaf_count, leaf_area, thresholded = extract_leaf_features(image_path)
#
# # Display the thresholded image
# plt.imshow(thresholded, cmap='gray')
# plt.axis('off')
# plt.show()
#
# print("Leaf Count:", leaf_count)
# print("Leaf Area:", leaf_area)
#
# # Load the leaf features from the CSV file
# leaf_features = pd.read_csv("leaf_features_single.csv")
#
# # Select the first two columns (leaf count and leaf lengths)
# X_pred = leaf_features.iloc[:, :2]
#
# # Predict the biomass using the RF model
# biomass_pred = model.predict(X_pred)
#
# # Print the predicted biomass
# print("Predicted Biomass:", biomass_pred)