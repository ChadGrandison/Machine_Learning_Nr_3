import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns  # For better visualization of the confusion matrix
from scipy.spatial import distance

# Function to enhance edges using Canny edge detection
def enhance_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Function to remove background and detect speed signs
def remove_background(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        if circularity > 0.75:
            largest_contour = contour

    if largest_contour is not None:
        mask = np.zeros_like(image[:, :, 0])
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        foreground = cv2.bitwise_and(image, image, mask=mask)
        background = np.ones_like(image) * 255
        mask_inv = cv2.bitwise_not(mask)

        result = cv2.bitwise_or(foreground, cv2.bitwise_and(background, background, mask=mask_inv))
        return result

    return image

# Harris corner detection and feature extraction
def extract_harris_corners(image, mask):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to 3-bit grayscale (8 levels of gray) for better edge definition
    gray_3bit = (gray_image // 32) * 32

    # Apply thresholding
    _, thresh = cv2.threshold(gray_3bit, 80, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up
    kernel = np.ones((7, 7), np.uint8)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Get sign dimensions for adaptive distance calculation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        sign_diameter = min(w, h)
        min_distance = sign_diameter * 0.055  # Optimized value from our testing
    else:
        return 0  # Return 0 if no contour found

    # Improved Harris corner detection with optimized parameters
    gray = np.float32(clean_thresh)
    harris_corners = cv2.cornerHarris(gray, blockSize=10, ksize=3, k=0.04)  # Optimized parameters
    harris_corners = cv2.dilate(harris_corners, None)

    # Apply threshold to detect strong corners
    corner_threshold = 0.3 * harris_corners.max()  # Optimized threshold
    corner_coords = np.argwhere(harris_corners > corner_threshold)
    filtered_coords = []

    # Create zero mask for excluding zeros
    zero_mask = np.zeros_like(mask)
    black_pixels = cv2.inRange(image, (0, 0, 0), (80, 80, 80))
    black_contours, _ = cv2.findContours(black_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in black_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.75 < aspect_ratio < 1.25:  # Aspect ratio for detecting zeros
            cv2.drawContours(zero_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Filter corners with adaptive minimum distance and zero exclusion
    for coord in corner_coords:
        y, x = coord
        if zero_mask[y, x] == 0:  # Exclude points inside zeros
            if all(distance.euclidean((x, y), (px, py)) >= min_distance for py, px in filtered_coords):
                filtered_coords.append((y, x))

    return len(filtered_coords)
# Function to detect if the image contains a speed sign (binary classification)
def detect_speed_sign(image):
    image_no_background = remove_background(image)

    # Extract red pixel density and circularity
    edges = enhance_edges(image_no_background)
    outer_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0
    red_pixel_density_perimeter = 0

    if outer_contours:
        outer_contour = max(outer_contours, key=cv2.contourArea)
        area = cv2.contourArea(outer_contour)
        perimeter = cv2.arcLength(outer_contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)

        mask = np.zeros(image_no_background.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [outer_contour], -1, (255), thickness=cv2.FILLED)

        red_pixels = cv2.inRange(image_no_background, (0, 0, 100), (80, 80, 255))
        red_pixel_count = np.sum(red_pixels[mask > 0] > 0)
        total_outer_pixels = np.sum(mask > 0)
        red_pixel_density_perimeter = red_pixel_count / total_outer_pixels if total_outer_pixels > 0 else 0

    # Return the features for binary classification
    return np.array([red_pixel_density_perimeter, circularity])

# Function to classify the speed sign (multi-class classification)
def classify_speed(image):
    image_no_background = remove_background(image)

    # Mask creation for the outer contour
    edges = enhance_edges(image_no_background)
    outer_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_perimeter_ratio = 0
    black_pixel_ratio = 0
    area_of_black_pixels = 0
    harris_corners_count = 0

    if outer_contours:
        outer_contour = max(outer_contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(outer_contour, True)

        mask = np.zeros(image_no_background.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [outer_contour], -1, (255), thickness=cv2.FILLED)

        black_pixels = cv2.inRange(image_no_background, (0, 0, 0), (80, 80, 80))
        black_pixel_count = np.sum(black_pixels > 0)

        total_area_count = np.sum(mask > 0)
        area_of_black_pixels = black_pixel_count
        black_pixel_ratio = area_of_black_pixels / total_area_count if total_area_count > 0 else 0

        black_contours, _ = cv2.findContours(black_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_digit_perimeter = sum(cv2.arcLength(contour, True) for contour in black_contours)
        digit_perimeter_ratio = total_digit_perimeter / perimeter if perimeter > 0 else 0

        # Harris corner detection added as a feature
        harris_corners_count = extract_harris_corners(image_no_background, mask)

    # Return the features for multi-class classification (50, 70, 120)
    return np.array([digit_perimeter_ratio, harris_corners_count, black_pixel_ratio, area_of_black_pixels])

# Combine both binary detection and multi-class classification
def extract_features(image):
    speed_sign_features = detect_speed_sign(image)
    speed_class_features = classify_speed(image)
    return np.concatenate([speed_sign_features, speed_class_features])

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 5))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


# Add this function after your existing plot functions
def analyze_feature_importance(classifier, feature_names):
    # Get feature importances
    importances = classifier.feature_importances_

    # Create a DataFrame with features and their importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance in Random Forest Classifier')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

    # Print detailed report
    print("\nFeature Importance Report:")
    print("------------------------")
    for idx, row in feature_importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    # Print cumulative importance
    print("\nCumulative Importance:")
    feature_importance_df['Cumulative'] = feature_importance_df['Importance'].cumsum()
    print(feature_importance_df)

    return feature_importance_df

# Load the training and testing datasets
train_image_path = r"C:\Users\chadg\OneDrive\Desktop\SEM 7\Machine Learning and Perception\Project\Photo frames\a_training_set"
test_image_path = r"C:\Users\chadg\OneDrive\Desktop\SEM 7\Machine Learning and Perception\Project\Photo frames\b_testing_set"

data = []
target = []

# Load training images and extract features
for file_name in os.listdir(train_image_path):
    image_path_full = os.path.join(train_image_path, file_name)
    image = cv2.imread(image_path_full)

    if image is not None:
        label = file_name.split('_')[0]
        features = extract_features(image)
        data.append(features)
        target.append(label)

# Ensure data isn't empty
if data and target:
    data = np.array(data)
    target = np.array(target)

    # Splitting data for training (60%) and testing (40%)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=42)

    classifier = RandomForestClassifier(
        n_estimators=400,  # Keep this as it's optimized
        max_depth=10,  # Add this to limit tree depth (was None)
        min_samples_split=6,  # Increase from 4 to 6
        min_samples_leaf=4,  # Increase from 2 to 4
        max_features='sqrt',  # Keep this
        class_weight='balanced',  # Keep this
        max_samples=0.7,  # Reduce from 0.8 to 0.7
        oob_score=True,  # Keep this
        bootstrap=True,  # Keep this
        ccp_alpha=0.001,  # Add small cost-complexity pruning
        random_state=42
    )

    # Train the Random Forest classifier
    classifier.fit(X_train, y_train)

    # Evaluate on the training set
    y_train_pred = classifier.predict(X_train)
    print("Classification Report for Training Set:")
    print(classification_report(y_train, y_train_pred))

    # Confusion matrix for the training set
    print("Confusion Matrix for Training Set:")
    plot_confusion_matrix(y_train, y_train_pred, np.unique(y_train))

    # Plot learning curve
    plot_learning_curve(classifier, X_train, y_train)

    # Evaluate on the test set
    y_test_pred = classifier.predict(X_test)
    print("Classification Report for Test Set:")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix for the test set
    print("Confusion Matrix for Test Set:")
    plot_confusion_matrix(y_test, y_test_pred, np.unique(y_test))

    # Define feature names
    feature_names = [
        'Red Pixel Density',
        'Circularity',
        'Digit Perimeter Ratio',
        'Harris Corner Count',
        'Black Pixel Ratio',
        'Black Pixel Area'
    ]

    # Analyze and plot feature importance
    print("\nAnalyzing Feature Importance...")
    importance_df = analyze_feature_importance(classifier, feature_names)

    # You can also access feature importances directly
    feature_importance = classifier.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Additional visualization: Horizontal bar plot with error bars
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance with Random Forest (with Standard Deviation)')
    plt.tight_layout()
    plt.show()

    # Create the directory if it doesn't exist
    save_directory = r"C:\Users\chadg\OneDrive\Desktop\SEM 7\Machine Learning and Perception\4\take 2"
    os.makedirs(save_directory, exist_ok=True)

    # Save the trained model
    model_filename = os.path.join(save_directory, 'speed_sign_classifier_4.2.joblib')
    dump(classifier, model_filename)
    print(f"Model saved as {model_filename}")
