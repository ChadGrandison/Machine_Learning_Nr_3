# import the necessary packages
import cv2
import numpy as np


def blur_image(image):
    image_blur = cv2.GaussianBlur(image, (13, 13), 0)
    return image_blur

def remove_greenscreen(image, lower_green=(35, 40, 40), upper_green=(110, 255, 255)):
    #(image, lower_green=(40, 40, 40), upper_green=(70, 255, 255))
    """
    Removes the greenscreen by making green pixels transparent.
    input:
    image: original image with greenscreen background.
    lower_green: Lower border for green color range in HSV (values are adjustable)
    upper_green: Upper border for green color range in HSV (values are adjustable)
    return:
    Image with transparent background.
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Add alpha channel (transparency) to image.
    bgr_channels = cv2.split(image)
    alpha_channel = np.ones(bgr_channels[0].shape, dtype=bgr_channels[0].dtype) * 255 # Alphachannel intensity 100%(255)

    # Convert green area to transparent alpha channel
    alpha_channel[mask == 255] = 0  # Convert alpha channel to 0 (transparent) in green area.

    # Add alpha channel to original image
    image_with_alpha = cv2.merge((*bgr_channels, alpha_channel))

    return image_with_alpha


def convert_to_3bit(image):
    """
    Convert image to 3-bit colors.
    """
    # Divide by 128 and multiply by 255 to reduce colors to 3-bit.
    img_3bit = (image // 128) * 255

    # Voeg een alfakanaal (transparantie) toe aan de afbeelding
    bgr_channels = cv2.split(img_3bit)
    alpha_channel = np.ones(bgr_channels[0].shape,
                            dtype=bgr_channels[0].dtype) * 255  # Maak het alfakanaal volledig zichtbaar (255)

    # Maak groene gebieden in het alfakanaal transparant
    alpha_channel[mask == 255] = 0  # Waar het groen is, wordt het alfakanaal 0 (transparant)

    image_3bit_with_alpha = cv2.merge((*bgr_channels, alpha_channel))

    return image_3bit_with_alpha



# Load the image
input_image_path = 'frame_882.png'  # Name of image (inlcuding extension). Must be in same directory as script.
#input_image_path = '20240916_140254_00193.png'  # Name of image (inlcuding extension). Must be in same directory as script.
image = cv2.imread(input_image_path)            # Load the image

# Check if image has loaded correctly
if image is None:
    print("Image not found or an error occurred  loading.")    # print string if no image was loaded
    exit()

# Converteer de afbeelding naar HSV-kleurruimte
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Maak een masker voor de groene kleuren
mask = cv2.inRange(hsv, (35, 40, 40), (110, 255, 255))

image_with_blur = blur_image(image)

cv2.imwrite('image_with_blur.png', image_with_blur)

# Verwijder de greenscreen
image_no_greenscreen = remove_greenscreen(image_with_blur)

# Sla de afbeelding met transparantie op als PNG (PNG ondersteunt transparantie)
output_image_path = 'image_no_greenscreen.png'  # Verander dit naar het gewenste uitvoerpad
cv2.imwrite(output_image_path, image_no_greenscreen)

image_3bit_with_alpha = convert_to_3bit(image_with_blur)     # Convert image to 3-bit colors

# Sla de 3-bit afbeelding op
output_image_path = '3bit_image_no_greenscreen.png'  # Verander dit naar het gewenste uitvoerpad
cv2.imwrite(output_image_path, image_3bit_with_alpha)

# Toon de afbeeldingen
#cv2.imshow('Originele afbeelding', image)                           # Toon originele afbeelding
#cv2.imshow('Afbeelding zonder greenscreen', image_no_greenscreen)   # Toon afbeelding zonder greenscreen
#cv2.imshow('3-bit image without background', image_3bit_with_alpha) # Toon de 3-bit afbeelding




# Wacht op een toets en sluit alle vensters
cv2.waitKey(0)
cv2.destroyAllWindows()
