import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pickle
import cv2
import numpy as np
import csv
def extract_leaf_features(image):
    # Load the image
    image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform adaptive thresholding to obtain a binary image
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Find contours of the leaves
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store leaf count and leaf area
    leaf_count = 0
    leaf_area = 0

    # Calculate leaf count and area for each contour
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.06 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Exclude contours with very small areas
        if cv2.contourArea(approx) > 400:
            leaf_count += 1
            leaf_area += cv2.contourArea(approx)

    # Return the leaf count, leaf area and Threshold image
    return leaf_count, leaf_area, thresholded

# Folder path containing images
image_folder = "../images/cropped_aug"

results = []

# Iterate through all images in the folder
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    print(image_path)
    # Extract leaf features from the sharpened image
    leaf_count, leaf_area, thresholded = extract_leaf_features(image_path)
    results.append((filename, leaf_count, leaf_area))

    # Export the features to a CSV file
csv_file = "leaf_features_new.csv"
with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Leaf Count", "Leaf Area Covered"])
        for result in results:
            writer.writerow(result)

        #print(f"Leaf features for {filename} exported to:", csv_file)