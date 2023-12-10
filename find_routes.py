# This file will be for:
# 1. Finding the different routes in the frame
# 2. Asking the climber which route they wants to climb
# 3. Updating the detections


import numpy as np
import supervision as sv
import cv2
from supervision.detection.core import Detections
COVER_AREA = 0.13 # area that color needs to be cover for the hold to be that color!

# COLOR RANGES IN HSV (Hue, Value, Saturation)
# Great posts to get right ranges:
# https://stackoverflow.com/questions/59623675/hsv-color-ranges-for-vibgyor-colors/59623829#59623829

red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
orange_lower = np.array([11, 100, 100])
orange_upper = np.array([20, 255, 255])
yellow_lower = np.array([22, 100, 100])
yellow_upper = np.array([35, 255, 255])
green_lower = np.array([40, 50, 0])
green_upper = np.array([88, 255, 255])
blue_lower = np.array([95, 50, 50])
blue_upper = np.array([120, 255, 255])
pink_lower = np.array([165, 50, 70])
pink_upper = np.array([180, 160, 250])
purple_lower = np.array([121, 50, 30])
purple_upper = np.array([160, 250, 250])
white_lower = np.array([0,0,100])
white_upper = np.array([180,40,155])
black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 40, 50]) 


def calculate_area(coordinates):
    x1, y1, x2, y2 = coordinates
    return abs(x2 - x1) * abs(y2 - y1)


def identify_routes(image, detections):
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_masks = {
        "Red": cv2.inRange(hsvFrame, red_lower, red_upper),
        "Orange": cv2.inRange(hsvFrame, orange_lower, orange_upper),
        "Yellow": cv2.inRange(hsvFrame, yellow_lower, yellow_upper),
        "Green": cv2.inRange(hsvFrame, green_lower, green_upper),
        "Blue": cv2.inRange(hsvFrame, blue_lower, blue_upper),
        "Pink": cv2.inRange(hsvFrame, pink_lower, pink_upper),
        "Purple": cv2.inRange(hsvFrame, purple_lower, purple_upper),
        "Black": cv2.inRange(hsvFrame, black_lower, black_upper),
        "White": cv2.inRange(hsvFrame, white_lower, white_upper)
    }

    color_contours = {color: cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for color, mask in color_masks.items()}

    routes = {}  # The routes

    for detection in detections:
        detection_coordinates = detection[0]
        x1, y1, x2, y2 = map(int, detection_coordinates)

        color_detected = False
        for color_name, contours in color_contours.items():
            color_detection = identify_color_hold(image, contours, detection, color_name)
            if color_detection:
                routes.setdefault(color_name, []).extend([detection])
                # colored_detections.append((x1, y1, x2, y2))  # Add the detection to the set of colored detections
                color_detected = True
                break

        # No specific color is detected
        if not color_detected:
            routes.setdefault("Uncoloured", []).extend([detection])
            # colored_detections.append((x1, y1, x2, y2))
            continue

    # Necessary to make sure the detections are of
    for color, detection_list in routes.items():
        bounding_boxes = [detection[0] for detection in detection_list]
        detections_array = np.array(bounding_boxes)

        if detections_array.ndim != 2 or detections_array.shape[1] != 4:
            raise ValueError(f"Invalid shape for {color} detections. Expected (n, 4) array.")

        detections_instance = Detections(detections_array)
        routes[color] = detections_instance

    return routes


# BGR VALUES :)
colours = {"Red": (0, 0, 255), "Orange":(0, 165, 255), "Yellow":(0, 255, 255), \
           "Green":(0, 255, 0), "Blue":(255, 0, 0), "Pink":(108, 105, 255), \
           "Purple":(128, 0, 128), "White":(255, 255, 255), "Black":(0, 0, 0),\
           "Uncoloured":(64, 64, 64)}

def identify_color_hold(image, contours, detection, colour_name):
    detection_coordinates = detection[0]
    area = calculate_area(detection_coordinates)

    # its_curr_color = False

    if area > 300:  # area threshold for holds
        # x1, y1, x2, y2 = detection_coordinates
        x1, y1, x2, y2 = map(int, detection_coordinates)
        # Check for red holds
        for contour in contours:
            colour_area = cv2.contourArea(contour)
            if colour_area >= (COVER_AREA * area):  # making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    # Color hold found within detection, draw a box the size of the original detection
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), colours[colour_name], 3)
                    cv2.putText(image, colour_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colours[colour_name])
                    return detection
    return None

def get_user_route(image, detection_routes):
    """Go through the dictionary of routes and make sure that get the user's preferred route"""
    user_chose_route = False
    # Display available colors to user
    route_colours = []
    print(detection_routes)
    while not user_chose_route:
        print("These are the available routes:")
        for index, color in enumerate(detection_routes.keys()):
            route_colours.append(color)
            print(f"{index}. {color}")

        # Ask the user select route colour
        selected_number = int(input("Please enter the number that corresponds to the route: "))

        # Ensure the input is within the valid range & return selected route
        if (selected_number < len(detection_routes)):
            colour_name = route_colours[selected_number]
            selected_detections = detection_routes[colour_name]
            # Perform actions with the selected detection data
            print(f"You selected the {colour_name} route!")
            # print(selected_detections)
            b_val, g_val, r_val = colours[colour_name]
            display_detections(image, selected_detections, colour_name, colours[colour_name])
            selected_color = sv.Color(b_val, g_val, r_val)
            user_chose_route = True
            return selected_detections, selected_color, colour_name
        else:
            print("Invalid input. Please enter a number within the provided range.")

def display_detections(image, detections, colour_name, color_bgr=(0, 255, 0)):
    for detection in detections:
        detection_coordinates = detection[0]
        x1, y1, x2, y2 =  map(int, detection_coordinates)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(image, colour_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr)

def average_detection_size(detections):
    total_width = 0
    total_height = 0
    count = 0

    for detection in detections:
        x1, y1, x2, y2 = detection[0]  # Assuming detection format is [x1, y1, x2, y2]
        total_width += abs(x2 - x1)
        total_height += abs(y2 - y1)
        count += 1

    if count > 0:
        average_width = total_width / count
        average_height = total_height / count
        return (average_width, average_height)
    else:
        return (0, 0)  # Return 0 if there are no detections

updated_squares = []
def add_square(event, x, y, flags, param):
    global updated_squares
    if event == cv2.EVENT_LBUTTONDOWN:
        image, avg_width, avg_height = param

        x1 = x - avg_width // 2
        y1 = y - avg_height // 2
        x2 = x + avg_width // 2
        y2 = y + avg_height // 2

        # Ensure x1, y1, x2, y2 are within the bounds of the image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Update the image with the new square
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.imshow("Draw Holds", image)

        updated_squares.append((x1, y1, x2, y2))


def add_detections(image, route_to_update, route_color, colour_name):
    global updated_squares
    print("Do you want to add holds to the route? (yes/no)")
    user_choice = input().lower()
    avg_width, avg_height = map(int, average_detection_size(route_to_update))

    if user_choice == "yes" or user_choice == "y":
        updated_squares = []
        print("CLICK ON THE CENTER OF THE NEW HOLD TO ADD IT\n")
        print("Press the d key to stop adding holds\n")

        cv2.namedWindow("Draw Holds")
        cv2.setMouseCallback("Draw Holds", add_square, param=(image, avg_width,avg_height))

        while True:
            cv2.putText(image, "Adding holds...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 100), 2)
            cv2.imshow("Draw Holds", image)
            display_detections(image, route_to_update, colour_name, colours[colour_name])

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                cv2.destroyWindow("Draw Holds")
                break
        if len(updated_squares) == 0:
            print("No new holds added to the route!")
            return route_to_update

        route = [detection[0] for detection in route_to_update]
        for square in updated_squares:
            square_array = np.array(square, dtype=np.float32)
            route.append(square_array)

        # Update detections with the modified route
        updated_detections = Detections(np.array(route))
        print("DONE ADDING HOLDS TO THE ROUTE!")
        return updated_detections
    else:
        print("No new holds added to the route!")
        return route_to_update

