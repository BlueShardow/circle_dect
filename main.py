import cv2
import numpy as np
import os

def resize_to_720p(img):
    """
    Resize the image to have a height of 720 pixels while maintaining the original aspect ratio.
    """
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_height = 720
    new_width = int(new_height * aspect_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def detect_circles(img):
    """
    Detect circles in the image using Hough Circle Transform.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=200, minRadius=25, maxRadius=99999)

    # Draw circles on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2) # outer circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3) # center point

    return img, circles

image_path = "/Users/pl1001515/Downloads/can1.jpeg"

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")

else:
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Unable to load the image '{image_path}'. Check file path or file integrity.")

    else:
        # Resize image to 720p
        resized_img = resize_to_720p(img)

        # Detect circles
        detected_img, circles = detect_circles(resized_img)

        # Display image
        cv2.imshow('Detected Circles', detected_img)

        # Wait for user to press any key
        cv2.waitKey(0)
        cv2.destroyAllWindows()
