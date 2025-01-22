import cv2
import numpy as np
import os

def resize_to_480p(img):
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_height = 480
    new_width = int(new_height * aspect_ratio)

    return cv2.resize(img, (new_width, new_height))

def enhance_contrast(img):
    """"Gray scales and HOG is used to increase contrast"""
    if len(img.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(5, 5))
        img = clahe.apply(img)

        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    elif len(img.shape) == 3:  # Color
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced_l = cv2.normalize(enhanced_l, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_lab = cv2.merge((enhanced_l, a, b))

        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    else:
        return img

def sharpen_image(img):
    """"Helps define edges better"""
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])

    return cv2.filter2D(img, -1, kernel)

def detect_best_circle(img, param1_range=(35, 150), param2_range=(100, 250), minRadius=25, maxRadius=99999):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    gray = sharpen_image(gray)
    gray = cv2.medianBlur(gray, ksize=5)

    best_circle = None
    max_radius = 0

    height, width = gray.shape

    """"Goes through a bunchn of parameters to find the 'best' circle"""
    for param2 in range(param2_range[0], param2_range[1], 10):
        print ("0")

        for param1 in range(param1_range[0], param1_range[1], 10):
            print ("1")

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=param1, param2=param2,
                                       minRadius=minRadius, maxRadius=maxRadius)
            if circles is not None:
                circles = np.uint16(np.around(circles))

                for circle in circles[0, :]:
                    x, y, radius = circle

                    # Ensure the entire circle is within the image bounds
                    if (x - radius >= 0 and y - radius >= 0 and 
                        x + radius < width and y + radius < height):
                        if radius > max_radius:
                            max_radius = radius
                            best_circle = (x, y, radius)

    return best_circle

def draw_circle(img, circle):
    if circle is not None:
        x, y, radius = circle
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    return img

# Main script
image_path = "/Users/pl1001515/Downloads/can.jpg"

# debugging start
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
gray0 = enhance_contrast(gray)
gray1 = resize_to_480p(gray0)
gray2 = sharpen_image(gray1)
gray3 = cv2.GaussianBlur(gray2, (17, 17), 0)
gray4 = sharpen_image(gray3)
#_, gray5 = cv2.threshold(gray4, 100, 255, cv2.THRESH_BINARY)
gray6 = cv2.medianBlur(gray4, ksize=5)
# debugging end

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")

else:
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load the image '{image_path}'. Check file path or file integrity.")

    else:
        enhanced_img = enhance_contrast(img)
        resized_img = resize_to_480p(enhanced_img)
        sharpened_img = sharpen_image(resized_img)
        best_circle = detect_best_circle(sharpened_img)
        output_img = draw_circle(sharpened_img, best_circle)

        cv2.imshow("Best Circle with Enhanced Contrast", output_img)
        
        # debugging start
        cv2.imshow("gray", gray)
        cv2.imshow("gray0", gray0)
        cv2.imshow("gray1", gray1)
        cv2.imshow("gray2", gray2)
        cv2.imshow("gray3", gray3)
        cv2.imshow("gray4", gray4)
        #cv2.imshow("gray5", gray5)
        cv2.imshow("gray6", gray6)
        # debugging end

        cv2.waitKey(0)
        cv2.destroyAllWindows()
