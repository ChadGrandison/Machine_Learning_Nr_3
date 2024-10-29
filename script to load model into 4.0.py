import cv2
import numpy as np
import os
from joblib import load
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


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


def main():
    # Load the trained model
    model_path = (r"C:\Users\chadg\OneDrive\Desktop\SEM 7\Machine Learning and Perception\4\take 2"
                  r"\speed_sign_classifier_4.2.joblib")

    classifier = load(model_path)
    print("Model loaded successfully")

    # Path to the test images
    test_image_path = (r"C:\Users\chadg\OneDrive\Desktop\SEM 7\Machine Learning and Perception\Project\Photo frames"
                       r"\random signs")

    # Set confidence threshold (adjust this value based on testing)
    CONFIDENCE_THRESHOLD = 0.65  # xx% confidence required for a prediction

    # Lists to store results
    predictions = []
    true_labels = []
    results_dict = {}

    # Counters for statistics
    total_images = 0
    unknown_count = 0
    known_correct = 0
    known_incorrect = 0
    false_positives = 0  # When non-target signs are classified as 50/70/120

    # Process each image
    for file_name in os.listdir(test_image_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            image_path = os.path.join(test_image_path, file_name)
            image = cv2.imread(image_path)

            if image is not None:
                # Extract features
                features = extract_features(image)

                # Get prediction probabilities
                pred_probs = classifier.predict_proba([features])[0]
                max_prob = np.max(pred_probs)
                prediction = classifier.classes_[np.argmax(pred_probs)]

                # Get true label from filename
                true_label = file_name.split('_')[0]

                # Determine if prediction should be "unknown"
                if max_prob < CONFIDENCE_THRESHOLD:
                    prediction = "unknown"
                    unknown_count += 1
                else:
                    if true_label in ['50', '70', '120']:
                        if prediction == true_label:
                            known_correct += 1
                        else:
                            known_incorrect += 1
                    else:
                        false_positives += 1

                # Store results
                predictions.append(prediction)
                true_labels.append(true_label)
                results_dict[file_name] = {
                    'True Label': true_label,
                    'Predicted': prediction,
                    'Confidence': max_prob,
                    'Correct': prediction == true_label if true_label in ['50', '70',
                                                                          '120'] else prediction == "unknown"
                }

                print(f"File: {file_name}")
                print(f"True Label: {true_label}")
                print(f"Predicted: {prediction}")
                print(f"Confidence: {max_prob:.2%}")
                print(f"Status: {'Correct' if results_dict[file_name]['Correct'] else 'Incorrect'}")
                print("-" * 50)

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total images processed: {total_images}")
    print(f"Images classified as unknown: {unknown_count} ({unknown_count / total_images:.1%})")
    print(f"Correct classifications (50/70/120): {known_correct}")
    print(f"Incorrect classifications (50/70/120): {known_incorrect}")
    print(f"False positives (other signs classified as 50/70/120): {false_positives}")

    # Calculate accuracy for known signs
    known_signs_total = sum(1 for label in true_labels if label in ['50', '70', '120'])
    if known_signs_total > 0:
        known_accuracy = known_correct / known_signs_total
        print(f"\nAccuracy for known signs (50/70/120): {known_accuracy:.1%}")

    # Calculate rejection rate for unknown signs
    unknown_signs_total = sum(1 for label in true_labels if label not in ['50', '70', '120'])
    if unknown_signs_total > 0:
        rejection_rate = (unknown_signs_total - false_positives) / unknown_signs_total
        print(f"Rejection rate for unknown signs: {rejection_rate:.1%}")

    # Create confusion matrix only for actual classifications (excluding unknowns)
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "unknown"]
    if valid_indices:
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]

        print("\nConfusion Matrix (excluding unknowns):")
        plot_confusion_matrix(valid_true_labels, valid_predictions,
                              np.unique([label for label in valid_true_labels if label in ['50', '70', '120', '?']]))

    # Print detailed examples of misclassifications
    print("\nDetailed Misclassifications:")
    for file_name, result in results_dict.items():
        if not result['Correct']:
            print(f"File: {file_name}")
            print(f"True: {result['True Label']}, Predicted: {result['Predicted']}")
            print(f"Confidence: {result['Confidence']:.2%}")
            print("-" * 30)


if __name__ == "__main__":
    main()