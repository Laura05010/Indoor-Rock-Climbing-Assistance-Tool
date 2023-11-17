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
        red_detection = identify_red_hold(image, red_contours, detection)
        if red_detection:
            routes.setdefault("red", []).extend([detection])

        orange_detection = identify_orange_hold(image, orange_contours, detection)
        if orange_detection:
            routes.setdefault("orange", []).extend([detection])

        yellow_detection = identify_yellow_hold(image, yellow_contours, detection)
        if yellow_detection:
            routes.setdefault("yellow", []).extend([detection])

        green_detection = identify_green_hold(image, green_contours, detection)
        if green_detection:
            routes.setdefault("green", []).extend([detection])

        blue_detection = identify_blue_hold(image, blue_contours, detection)
        if blue_detection:
            routes.setdefault("blue", []).extend([detection])

        pink_detection = identify_pink_hold(image, pink_contours, detection)
        if pink_detection:
            routes.setdefault("pink", []).extend([detection])

        purple_detection = identify_purple_hold(image, purple_contours, detection)
        if purple_detection:
            routes.setdefault("purple", []).extend([detection])

        white_detection = identify_white_hold(image, white_contours, detection)
        if white_detection:
            routes.setdefault("white", []).extend([detection])

        black_detection = identify_black_hold(image, black_contours, detection)
        if black_detection:
            routes.setdefault("black", []).extend([detection])




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

def identify_red_hold(image, red_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_red = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for red holds
        for red_contour in red_contours:
            red_area = cv2.contourArea(red_contour)
            if red_area  >= (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(red_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_red = True
                    # Red hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(red_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    if its_red:
        return detection


def identify_orange_hold(image, orange_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_orange = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for orange holds
        for orange_contour in orange_contours:
            orange_area = cv2.contourArea(orange_contour)
            if orange_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(orange_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_orange = True
                    # Orange hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(orange_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(image, "Orange", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255))
    if its_orange:
        return detection
    return None

def identify_yellow_hold(image, yellow_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_yellow = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for yellow holds
        for yellow_contour in yellow_contours:
            yellow_area = cv2.contourArea(yellow_contour)
            if yellow_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(yellow_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_yellow = True
                    # Yellow hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(yellow_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(image, "Yellow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))
    if its_yellow:
        return detection
    return None

def identify_green_hold(image, green_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_green = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for green holds
        for green_contour in green_contours:
            green_area = cv2.contourArea(green_contour)
            if green_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(green_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_green = True
                    # Green hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(green_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    if its_green:
        return detection
    return None


def identify_blue_hold(image, blue_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_blue = False

    if area > 300: # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for blue holds
        for blue_contour in blue_contours:
            blue_area = cv2.contourArea(blue_contour)
            if blue_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(blue_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_blue = True
                    # colour hold found within detection, process it
                    # cv2.drawContours(image, [blue_contour], -1, (255, 0, 0), 2)  # Draw blue contour
                    x, y, w, h = cv2.boundingRect(blue_contour)
                    image = cv2.rectangle(image, (x, y),(x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(image, "Blue", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
    if its_blue:
        return detection
    return None

def identify_pink_hold(image, pink_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_pink = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for pink holds
        for pink_contour in pink_contours:
            pink_area = cv2.contourArea(pink_contour)
            if pink_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(pink_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_pink = True
                    # Pink hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(pink_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (203, 192, 255), 2)
                    cv2.putText(image, "Pink", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (203, 192, 255))
    if its_pink:
        return detection
    return None

def identify_purple_hold(image, purple_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_purple = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for purple holds
        for purple_contour in purple_contours:
            purple_area = cv2.contourArea(purple_contour)
            if purple_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(purple_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_purple = True
                    # Purple hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(purple_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 2)
                    cv2.putText(image, "Purple", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 128))
    if its_purple:
        return detection
    return None

def identify_white_hold(image, white_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_white = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for white holds
        for white_contour in white_contours:
            white_area = cv2.contourArea(white_contour)
            if white_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(white_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_white = True
                    # White hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(white_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(image, "White", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    if its_white:
        return detection
    return None

def identify_black_hold(image, black_contours, detection):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    its_black = False

    if area > 300:  # area threshold for holds
        x1, y1, x2, y2 = detection_coordinates
        # Check for black holds
        for black_contour in black_contours:
            black_area = cv2.contourArea(black_contour)
            if black_area  >=  (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(black_contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    its_black = True
                    # Black hold found within detection, process it
                    x, y, w, h = cv2.boundingRect(black_contour)
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    cv2.putText(image, "Black", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
    if its_black:
        return detection
    return None

