# This file will be for:
# 1. Finding the different routes in the frame
# 2. Asking the climber which route he wants to climb
# 3. Updating the detections


import numpy as np
import cv2
from supervision.detection.core import Detections
COVER_AREA = 0.2 # area that color needs to be cover for the hold to be that color!

# COLOR RANGES

# Red color range
red_lower = np.array([0, 100, 100], np.uint8)
red_upper = np.array([10, 255, 255], np.uint8)

# Orange color range (including lighter tones)
orange_lower = np.array([1, 190, 200], np.uint8)
orange_upper = np.array([18, 255, 255], np.uint8)

# Yellow color range
yellow_lower = np.array([20, 100, 100], np.uint8)
yellow_upper = np.array([35, 255, 255], np.uint8)

# Green color range
green_lower = np.array([40, 50, 100], np.uint8)
green_upper = np.array([90, 255, 255], np.uint8)

# Blue color range
blue_lower = np.array([95, 50, 50], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)

# Pink color range
pink_lower = np.array([155, 70, 200], np.uint8)
pink_upper = np.array([175, 255, 255], np.uint8)

# Purple color range
purple_lower = np.array([115, 40, 40], np.uint8)
purple_upper = np.array([145, 255, 255], np.uint8)

# White color range
white_lower = np.array([0, 0, 200], np.uint8)
white_upper = np.array([180, 30, 255], np.uint8)

# Black color range
black_lower = np.array([0, 0, 0], np.uint8)
black_upper = np.array([180, 55, 40], np.uint8)



def calculate_area(coordinates):
    x1, y1, x2, y2 = coordinates
    return abs(x2 - x1) * abs(y2 - y1)

def identify_routes(image, detections):
    # Convert the image from BGR to HSV(hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get colour masks
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    pink_mask = cv2.inRange(hsvFrame, pink_lower, pink_upper)
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)


    # Find contours for each color mask
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    routes = {} # dictionary of routes :)

    for detection in detections:
        red_detection = identify_color_hold(image, red_contours, detection, "Red")
        if red_detection:
            routes.setdefault("Red", []).extend([detection])

        orange_detection = identify_color_hold(image, orange_contours, detection, "Orange")
        if orange_detection:
            routes.setdefault("Orange", []).extend([detection])

        yellow_detection = identify_color_hold(image, yellow_contours, detection, "Yellow")
        if yellow_detection:
            routes.setdefault("Yellow", []).extend([detection])

        green_detection = identify_color_hold(image, green_contours, detection, "Green")
        if green_detection:
            routes.setdefault("Green", []).extend([detection])

        blue_detection = identify_color_hold(image, blue_contours, detection, "Blue")
        if blue_detection:
            routes.setdefault("Blue", []).extend([detection])

        pink_detection = identify_color_hold(image, pink_contours, detection, "Pink")
        if pink_detection:
            routes.setdefault("Pink", []).extend([detection])

        purple_detection = identify_color_hold(image, purple_contours, detection, "Purple")
        if purple_detection:
            routes.setdefault("Purple", []).extend([detection])

        white_detection = identify_color_hold(image, white_contours, detection, "White")
        if white_detection:
            routes.setdefault("White", []).extend([detection])

        black_detection = identify_color_hold(image, black_contours, detection, "Black")
        if black_detection:
            routes.setdefault("Black", []).extend([detection])



    # Modify the routes dictionary to convert detection lists to Detections instances
    for color, detection_list in routes.items():
        # Extract all bounding box coordinates
        bounding_boxes = [detection[0] for detection in detection_list]
        detections_array = np.array(bounding_boxes)

        # Ensure the shape of array is (n, 4)
        if detections_array.ndim != 2 or detections_array.shape[1] != 4:
            raise ValueError(f"Invalid shape for {color} detections. Expected (n, 4) array.")

        # convert to detections type
        detections_instance = Detections(detections_array)
        routes[color] = detections_instance
    return routes

def display_routes(routes):
    pass

colours = {"Red": (0, 0, 255), "Orange":(0, 165, 255), "Yellow":(0, 255, 255), \
           "Green":(0, 255, 0), "Blue":(255, 0, 0), "Pink":(203, 192, 255), \
           "Purple":(128, 0, 128), "White":(255, 255, 255), "Black":(0, 0, 0)}

def identify_color_hold(image, contours, detection, colour_name):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_curr_color = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for red holds
        for contour in contours:
            colour_area = cv2.contourArea(contour)
            if  colour_area  >= (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_curr_color  = True
                    # color hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), colours[colour_name], 2)
                    cv2.putText(image, colour_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colours[colour_name])
    if its_curr_color:
        return detection