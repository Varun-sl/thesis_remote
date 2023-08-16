import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import seaborn as sns

# Load the dataset
data = pd.read_csv("leaf_features_noise.csv")
#data = data.drop(columns=['Image Name'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[["Leaf Count", "Leaf Area Covered"]], data["Weight"], test_size=0.4, random_state=42)

# Define a parameter grid for hyperparameter tuning
# param_grid = {
#     'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
#     'activation': ['relu', 'logistic'],
#     'max_iter': [500, 1000, 1500],
# }

param_grid = {
    'hidden_layer_sizes': [(64,)],
    'activation': ['logistic'],
    'max_iter': [500],
}

# Create the MLPRegressor model
model = MLPRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=8, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Compute R^2 score and mean squared error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the best hyperparameters, R^2 score, and mean squared error
print("Best Hyperparameters:", grid_search.best_params_)
print(f"R^2 Score: {r2:.3f}")
print(f"Mean Squared Error: {mse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")

# Plot the actual biomass values and predicted biomass values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Biomass (grams)')
plt.ylabel('Predicted Biomass (grams)')
plt.title('Biomass Estimation - Multi-Layer Perceptron')
plt.ylim(0.05, 0.15)
plt.show()
