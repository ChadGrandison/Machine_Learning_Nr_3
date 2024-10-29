# 🚦 Speed Sign Detection Project

## 🌟 Important Files
1. [`training_model_4.0.py`](./path/to/training_model_4.0.py) - Trains the model on speed sign dataset
2. [`to_load_model_into_4.0.py`](./path/to/to_load_model_into_4.0.py) - Tests the trained model on new images

## 🚀 Quick Start Guide

### Step 1: Download Required Scripts
- Download both Python scripts from this repository:
  - ⭐ `training_model_4.0.py`
  - ⭐ `to_load_model_into_4.0.py`

### Step 2: Set Up Training Model
1. Open `training_model_4.0.py`
2. Locate and modify these paths according to your local machine:
```python
# Training images path
train_image_path = r"YOUR_PATH_TO_TRAINING_IMAGES"

# Where to save the trained model
save_directory = r"WHERE_YOU_WANT_TO_SAVE_MODEL"
```

### Step 3: Train the Model
1. Run `training_model_4.0.py`
2. Wait for training to complete
3. Note where your model is saved (check console output)
4. The model will be saved as 'speed_sign_classifier_4.0.joblib'

### Step 4: Set Up Testing Script
1. Open `to_load_model_into_4.0.py`
2. Modify these paths:
```python
# Path to your trained model
model_path = r"PATH_TO_YOUR_SAVED_MODEL/speed_sign_classifier_4.0.joblib"

# Path to test images
test_image_path = r"PATH_TO_YOUR_TEST_IMAGES"
```

### Step 5: Test Your Model
1. Create a folder with test images
   - Can include speed signs (50, 70, 120)
   - Can include other traffic signs to test robustness
2. Run `to_load_model_into_4.0.py`
3. View results in console output

## 📋 Requirements
- Python 3.x
- Required libraries:
  - opencv-python
  - numpy
  - scikit-learn
  - joblib
  - matplotlib
  - seaborn
  - scipy

## 🎯 Expected Output
The test script will show:
- Individual predictions for each image
- Overall classification report
- Confusion matrix
- Accuracy per class
- Statistics about unknown signs

## ⚠️ Common Issues
1. "File not found" errors
   - Double-check all file paths
   - Use raw strings (r"path/to/file")
   - Ensure paths match your local machine

2. Missing libraries
   - Install using pip:
   ```bash
   pip install opencv-python numpy scikit-learn joblib matplotlib seaborn scipy
   ```

## 🤝 Need Help?
If you're having issues:
1. Check all file paths are correct
2. Ensure all required libraries are installed
3. Verify image folder contains valid image files (.png, .jpg, .jpeg)

## 📝 Note
The model is trained to detect speed signs (50, 70, 120 km/h) and includes robustness testing for other signs. A confidence threshold of 0.6 is used to filter out non-speed signs.

---
Enjoy testing your speed sign classifier! 🚗💨
