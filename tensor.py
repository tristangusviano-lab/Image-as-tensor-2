import cv2
import numpy as np

#check if the image exists and is valid
img = cv2.imread("images (4).jpeg") #Reads image

if img is None:
    raise FileNotFoundError("File not found")

if img.ndim != 3 or img.shape[2] != 3:
    raise ValueError("Image must have 3 colour channels")

height, width, channels = img.shape

if height < 224 or width < 224:
    raise ValueError("Image is too small")

# Crop the image to 224x224 pixels
center_x = width // 2
center_y = height // 2

cropped = img[center_y-112:center_y+112,
              center_x-112:center_x+112]

#configuration for the filter mode
mode = "BERSERK" 

filtered = cropped.copy()

if mode == "BERSERK":
    filtered[:, :, 0] = 0       
    filtered[:, :, 1] = 0       

elif mode == "NIGHT_VISION":
    
    # 1. Convert to grayscale to isolate light intensity
    filtered = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # 2. Enhance contrast in dark areas using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    filtered = clahe.apply(filtered)

    # 3. Reduces transparency and enhances visibility in low-light conditions
    filtered = cv2.addWeighted(filtered, 1.5, filtered, 0.5, 0)
    
    # 4. Map intensity to the Green channel (BGR format)
    zeros = np.zeros_like(filtered)
    filtered = cv2.merge([zeros, filtered, zeros])
    
else:
    raise ValueError("Invalid filter mode")

cv2.imshow("Filtered Image", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()