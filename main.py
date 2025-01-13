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

    """Try to detect circles"""
    for param2 in range(param2_range[0], param2_range[1], 10):
        print ("0")

        for param1 in range(param1_range[0], param1_range[1], 10):
            print("1")

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=param1, param2=param2,
                                       minRadius=minRadius, maxRadius=maxRadius)

            if circles is not None:
                circles = np.uint16(np.around(circles))

                for circle in circles[0, :]:
                    x, y, radius = circle

                    if (x - radius >= 0 and y - radius >= 0 and 
                        x + radius < width and y + radius < height):

                        if radius > max_radius:
                            max_radius = radius
                            best_circle = (x, y, radius)

    print("Done c")

    return best_circle

def detect_best_ellipse(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 300, 500)
    cv2.imshow("Detected Edges", edges)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    
    best_ellipse = None
    max_area = 0

    for param1 in range(400, 500):
        print("a")

        for param2 in range(200, 300):
            print("b")

            edges = cv2.Canny(gray, param1, param2)

            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (major_axis, minor_axis), angle = ellipse
                    area = major_axis * minor_axis
            
                    # Validate axes and area
                    if 0 < major_axis < 200 and 0 < minor_axis < 200:
                        if area > max_area:
                            max_area = area
                            best_ellipse = ellipse

    print("Done e")

    return best_ellipse

def draw_shape(img, shape):
    print("Shape data:", shape)

    if shape is not None:
        if len(shape) == 3 and isinstance(shape[0], tuple) and isinstance(shape[1], tuple): # Ellipse
            center = tuple(map(int, shape[0]))
            axes = tuple(map(int, shape[1]))
            angle = int(shape[2])
            
            # Safeguard for invalid axes
            if axes[0] > 0 and axes[1] > 0:
                cv2.ellipse(img, center, axes, angle, 0, 360, (255, 0, 0), 2)

            else:
                print(f"Error: Invalid ellipse axes {axes}")
        else:
            print(f"Error: Unrecognized shape format {shape}")
    else:
        print("No shape to draw.")

    return img

def display_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()

    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", contour_img)

# Main script _________________________________________________________________
image_path = "can.jpeg"

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
        display_contours(sharpened_img)
        best_circle = detect_best_circle(sharpened_img)

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

        if best_circle is not None:
            output_img = draw_shape(sharpened_img, best_circle)

        else:
            best_ellipse = detect_best_ellipse(sharpened_img)

            if best_ellipse is not None:
                output_img = draw_shape(sharpened_img, best_ellipse)

            else:
                print("No valid circles or ellipses found.")

        cv2.imshow("Best Circle with Enhanced Contrast", output_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
