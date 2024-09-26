import cv2


def convert_to_3bit(img):
    """
    Convert image to 3-bit colors.
    """
    # Divide by 128 and multiply by 255 to reduce colors to 3-bit.
    img_3bit = (img // 128) * 255
    return img_3bit


# Load the image
input_image_path = 'image_no_greenscreen.png'  # Path of image file
img = cv2.imread(input_image_path)

if img is None:         # Check if image is loaded correctly
    print("Image not found or error when loading.")
    exit()

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, (35, 40, 40), (110, 255, 255))
# mask_inv = cv2.bitwise_not(mask)


img_3bit = convert_to_3bit(img)     # Perform color conversion function

# Show original image and 3-bit image.
# cv2.imshow('Original image', img)
# cv2.imshow('3-bit image', img_3bit)

# Save 3-bit image
output_image_path = '3bit_image.png'  # Output path
cv2.imwrite(output_image_path, img_3bit)
