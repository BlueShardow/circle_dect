import cv2
import numpy as np
import os
import math

def resize_with_scaling(img, target_height=480):
    height, width = img.shape[:2]
    scaling_factor = target_height / height
    new_width = int(width * scaling_factor)
    resized_img = cv2.resize(img, (new_width, target_height))

    return resized_img, scaling_factor

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

def detect_best_circle(img, param1_range=(50, 150), param2_range=(150, 250), minRadius=25, maxRadius=99999):
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
    gray = sharpen_image(gray)
    gray = cv2.medianBlur(gray, ksize=5)

    edges = cv2.Canny(gray, 300, 500)

    cv2.imshow("Detected Edges", edges)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    
    best_ellipse = None
    best_score = 0
    max_area = 0
    param1e = 0
    param2e = 0
    inter = 0
    ellipse_list = [None] * 2147483647
    angle_best = 0

    height, width = img.shape[:2]
    x_center = width // 2
    y_center = height // 2

    loop_index = 0

    for param1 in range(500, 750, 25):
        print("a")

        for param2 in range(300, 650, 25):
            loop_index += 1

            if loop_index % 100 == 0:
                print(f"Loop index: {loop_index}")

            print("b")

            edges = cv2.Canny(gray, param1, param2)

            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (major_axis, minor_axis), angle = ellipse
                    area = math.pi * (major_axis / 2) * (minor_axis / 2)
            
                    if (0 < major_axis < 999 and 0 < minor_axis < 999) and ((x > 50 and y > 50) and (x < 430 and y < 600)) and (100000 >area > 30000):
                        ellipse_list[inter] = ellipse
                        inter += 1

                        # Compute how "horizontal" the ellipse is
                        horizontal_bias = max(0, 90 - min(abs(angle - 90), abs(angle - 270)))
                        score = 0

                        if angle == 90 or angle == 270:
                            score = ((area ** 4) * (horizontal_bias ** 1.15)) / (((1 * (angle + 1)) * ((.75 * abs((x_center - x) + 1) / width))) * ((.5 * abs((y_center - y) + 1) / (.25 * height + 1))))

                        elif angle == 0 or angle == 180:
                            score = 0

                        else:
                            if 90 - 25 <= angle <= 90 + 25:
                                score = ((area ** 4) * (horizontal_bias ** 1.15)) / (((1 * (angle + 1)) * ((.75 * abs((x_center - x) + 1) / width))) * ((.5 * abs((y_center - y) + 1) / (.25 * height + 1))))

                            elif 270 - 25 <= angle <= 270 + 25:
                                score = ((area ** 4) * (horizontal_bias ** 1.15)) / (((1 * (180 - angle + 1)) * (.75 * (abs((x_center - x) + 1) / width))) * (.5 * (abs((y_center - y) + 1) / (.25 * height + 1))))

                        if score > best_score and 100000 > area > 30000:
                            best_ellipse = ellipse
                            best_score = score
                            param1e = param1
                            param2e = param2
                            angle_best = angle
                            max_area = area

    print("Done e")
    print("\n", "Best Score:", best_score, "\n")
    print("\n", "Max Area:", max_area, "\n")

    return best_ellipse, param1e, param2e, ellipse_list, angle_best

def draw_shape_with_scaling(img, shape, param1e, param2e, scaling_factor):
    """Draw shapes on the image, scaled according to the resizing factor."""
    if shape is None:
        return img
    
    try:
        # Circle: (x, y, radius)
        if len(shape) == 3 and all(isinstance(i, (int, float)) for i in shape):
            x, y, radius = map(int, shape)
            scaled_x = int(x * scaling_factor)
            scaled_y = int(y * scaling_factor)
            scaled_radius = int(radius * scaling_factor)
            cv2.circle(img, (scaled_x, scaled_y), scaled_radius, (0, 255, 0), 2)

        # Ellipse: ((x, y), (major, minor), angle)
        elif len(shape) == 3 and isinstance(shape[0], tuple):
            cv2.ellipse(img, shape, (255, 0, 0), 2)

            edges = cv2.Canny(gray, param1e, param2e)
            cv2.imshow("edges", edges)

            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = img.copy()

            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            cv2.imshow("contours", contour_img)

            (x, y), (major, minor), angle = shape
            scaled_x = int(x * scaling_factor)
            scaled_y = int(y * scaling_factor)
            scaled_major = int(major * scaling_factor)
            scaled_minor = int(minor * scaling_factor)
            cv2.ellipse(img, ((scaled_x, scaled_y), (scaled_major, scaled_minor), angle), (255, 0, 0), 2)
    
    except Exception as e:
        print(f"Error drawing shape: {e}")
    
    return img

def display_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()

    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", contour_img)

def display_ellipses(img_shape, ellipse_list):
    canvas = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)  # Create a black canvas
    
    for ellipse in ellipse_list:
        if ellipse is not None:
            cv2.ellipse(canvas, ellipse, (0, 255, 0), 1) 
    
    cv2.imshow("All Ellipses", canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main script _________________________________________________________________
image_path = "/Users/pl1001515/Downloads/can.jpeg"

_, ratio = resize_with_scaling(cv2.imread(image_path))

# debugging start
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

gray0 = enhance_contrast(gray)
gray1, _ = resize_with_scaling(gray0)  # Unpack only the resized image
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
        resized_img, _ = resize_with_scaling(enhanced_img)
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
        # debugging end

        if best_circle is not None:
            output_img = draw_shape_with_scaling(sharpened_img, best_circle, 0, 0, ratio)

        else:
            best_ellipse, param1e, param2e, ellipse_list, angle_best = detect_best_ellipse(sharpened_img)

            if best_ellipse is not None:
                print(param1e, param2e)
                print(angle_best)

                output_img = draw_shape_with_scaling(sharpened_img, best_ellipse, param1e, param2e, ratio)

            else:
                print("No valid circles or ellipses found.")

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
        
        #if best_ellipse is not None:
            #display_ellipses(sharpened_img.shape, ellipse_list)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
