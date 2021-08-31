import cv2
import numpy as np

"""
Function to add HSV color filter to frames
"""
def add_HSV_filter(frame, camera):
    # Blurring the frame
    blur = cv2.bilateralFilter(frame,9,75,75)

    # Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    l_b_r = np.array([30, 0, 0])  # Lower limit for green object right frame
    u_b_r = np.array([75, 255, 200])  # Upper limit for green object right frame
    l_b_l = np.array([30, 0, 0])  # Lower limit for green object left frame
    u_b_l = np.array([75, 255, 200])  # Upper limit for green object left frame

    if(camera == 1):
        mask = cv2.inRange(hsv, l_b_r, u_b_r)
    else:
        mask = cv2.inRange(hsv, l_b_l, u_b_l)

    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask
