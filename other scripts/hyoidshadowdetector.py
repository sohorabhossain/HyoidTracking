import cv2
import time
import numpy as np
import pandas as pd

# 1. Load the image from disk
# Replace 'path/to/image.jpg' with your actual file path
img = cv2.imread('D:\\my_works\\HyoidTracking\\sampleMendelsohnImage.jpg')

# Check if the image was successfully loaded
if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Display the image in a window
    cv2.imshow('Loaded Image', img)


    # 3. Wait for a key press (0 means wait indefinitely)
    # This prevents the window from closing immediately
    cv2.waitKey(0)

    # 4. Close all open OpenCV windows
    cv2.destroyAllWindows()