# This file will be for:
# 1. Finding the different routes in the frame
# 2. Asking the climber which route they wants to climb
# 3. Updating the detections


import numpy as np
import supervision as sv
import cv2
from supervision.detection.core import Detections
from pynput import keyboard
import threading
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

    routes = {}
    for detection in detections:
        detection_coordinates = detection[0]
        mask = detection[1]
        confidence = detection[2]
        class_id = detection[3]
        tracker_id = detection[4]
        data = detection[5]  # This contains additional data, potentially class names

        x1, y1, x2, y2 = map(int, detection_coordinates)

        color_detected = False
        for color_name, contours in color_contours.items():
            color_detection = identify_color_hold(image, contours, detection, color_name)
            if color_detection:
                routes.setdefault(color_name, []).append(detection)
                color_detected = True
                break

        if not color_detected:
            routes.setdefault("Uncoloured", []).append(detection)

    for color, detection_list in routes.items():
        bounding_boxes = np.array([detection[0] for detection in detection_list])
        confidence_array = np.array([detection[2] for detection in detection_list], dtype=np.float32)
        class_id_array = np.array([detection[3] for detection in detection_list], dtype=np.int32)

        if bounding_boxes.ndim != 2 or bounding_boxes.shape[1] != 4:
            raise ValueError(f"Invalid shape for {color} detections. Expected (n, 4) array.")

        # Collect 'data' as an array of class names
        data_array = np.array([detection[5]['class_name'] for detection in detection_list])

        detections_instance = Detections(
            xyxy=bounding_boxes,
            confidence=confidence_array,
            class_id=class_id_array,
            data={'class_name': data_array}
        )

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

    if area > 300:  # Area threshold for holds
        x1, y1, x2, y2 = map(int, detection_coordinates)

        # Ensure the image is writable
        image = np.asarray(image)
        if not image.flags.writeable:
            image.setflags(write=1)

        # Check for red holds
        for contour in contours:
            colour_area = cv2.contourArea(contour)
            if colour_area >= (COVER_AREA * area):  # Making sure color is covering most of the box
                color_x, color_y, color_w, color_h = cv2.boundingRect(contour)
                if x1 <= color_x <= x2 and y1 <= color_y <= y2:
                    # Color hold found within detection, draw a box the size of the original detection
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), colours[colour_name], 3)
                    cv2.putText(image, colour_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colours[colour_name], 2)
                    return detection

    return None

def get_user_route(image, detection_routes):
    """Go through the dictionary of routes and make sure that get the user's preferred route"""
    user_chose_route = False
    # Display available colors to user
    route_colours = []
    # print(detection_routes)
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
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colour_REP_ADD, 3)
        cv2.imshow("Draw Holds", image)

        updated_squares.append((x1, y1, x2, y2))


marked_points = []
colour_REP_REMOVE = (147,20,255)
colour_REP_ADD = (200, 213, 48)
def get_click_point(event, x, y, flags, param):
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
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colour_REP_REMOVE, 3)
        cv2.imshow("Remove Holds", image)

        marked_points.append((x,y))

stop_adding_holds = False
stop_removing_holds = False

def on_press_adding(key):
    global stop_adding_holds
    try:
        if key.char == 'd':
            stop_adding_holds = True
            return False  # Stop the listener when 'd' is pressed
    except AttributeError:
        pass

def on_press_removing(key):
    global stop_removing_holds
    try:
        if key.char == 'd':
            stop_removing_holds = True
            return False  # Stop the listener when 'd' is pressed
    except AttributeError:
        pass



def add_detections(image, route_to_update, route_color, colour_name):
    global stop_adding_holds, updated_squares
    stop_adding_holds = False
    updated_squares = []

    # Start a keyboard listener in a separate thread before asking the user for input
    adding_listener = keyboard.Listener(on_press=on_press_adding)
    adding_thread = threading.Thread(target=adding_listener.start)
    adding_thread.start()

    print("Do you want to add holds to the route? (yes/no)")
    user_choice = input().lower()[-1] # just take the last charcter to avoid buffer saving prev inputs
    avg_width, avg_height = map(int, average_detection_size(route_to_update))

    if user_choice in "yes":
        print("CLICK ON THE CENTER OF THE NEW HOLD TO ADD IT\n")
        print("Press the 'd' key to stop adding holds\n")

        cv2.namedWindow("Draw Holds")
        cv2.setMouseCallback("Draw Holds", add_square, param=(image, avg_width, avg_height))

        while not stop_adding_holds:
            cv2.putText(image, "Adding holds...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_REP_ADD, 2)
            cv2.imshow("Draw Holds", image)
            display_detections(image, route_to_update, colour_name, colours[colour_name])
            if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
                break

        cv2.destroyWindow("Draw Holds")
        stop_adding_holds = True  # Ensure the listener stops
        adding_listener.stop()  # Stop the listener
        adding_thread.join()    # Wait for the thread to finish

        if len(updated_squares) == 0:
            print("No new holds added to the route!")
            return route_to_update

        # Collect existing route detections
        route = [detection[0] for detection in route_to_update]

        # Add the new squares to the route
        for square in updated_squares:
            square_array = np.array(square, dtype=np.float32)
            route.append(square_array)

        # Create arrays of made-up values
        confidence = np.ones(len(route), dtype=np.float32)
        class_id = np.zeros(len(route), dtype=int)
        class_name = np.full(len(route), '0', dtype='<U1')  # or whatever default class name you want

        # Update detections with the modified route and additional fields
        updated_detections = Detections(
            xyxy=np.array(route),
            mask=None,
            confidence=confidence,
            class_id=class_id,
            tracker_id=None,
            data={'class_name': class_name}
        )

        print("\nDONE ADDING HOLDS TO THE ROUTE!")
        return updated_detections
    else:
        print("\nNo new holds added to the route!")
        return route_to_update



def remove_detections(image, route_to_update, route_color, colour_name):
    global stop_removing_holds, marked_points
    stop_removing_holds = False
    marked_points = []

    # Start a keyboard listener in a separate thread before asking the user for input
    removing_listener = keyboard.Listener(on_press=on_press_removing)
    removing_thread = threading.Thread(target=removing_listener.start)
    removing_thread.start()

    print("Do you want to remove holds from the route? (yes/no)")
    user_choice = input().lower()[-1] # just take the last character
    print(f"YOU CHOSE: {user_choice}")
    avg_width, avg_height = map(int, average_detection_size(route_to_update))

    if user_choice in "yes":
        print("CLICK ON THE HOLD TO MARK IT FOR REMOVAL\n")
        print("Press the 'd' key to stop removing holds\n")

        cv2.namedWindow("Remove Holds")
        cv2.setMouseCallback("Remove Holds", get_click_point, param=(image, avg_width, avg_height))

        while not stop_removing_holds:
            cv2.putText(image, "Removing holds...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_REP_REMOVE, 2)
            cv2.imshow("Remove Holds", image)
            display_detections(image, route_to_update, colour_name, colours[colour_name])

            if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
                break

        cv2.destroyWindow("Remove Holds")
        stop_removing_holds = True  # Ensure the listener stops
        removing_listener.stop()  # Stop the listener
        removing_thread.join()    # Wait for the thread to finish

        if len(marked_points) == 0:
            print("No holds removed from the route!")
            return route_to_update

        # Initialize empty lists to collect the components
        bounding_boxes, confidence_array, class_id_array, data_array = [], [], [], []

        # Iterate over route_to_update and process detections
        for detection in route_to_update:
            detection_coordinates, confidence, class_id, data = detection[0], detection[2], detection[3], detection[5]
            keep_detection = True
            for point in marked_points:
                x, y = point
                x1, y1, x2, y2 = detection_coordinates
                if x1 <= x <= x2 and y1 <= y <= y2:
                    keep_detection = False
                    break

            if keep_detection:
                bounding_boxes.append(detection_coordinates)
                confidence_array.append(confidence)
                class_id_array.append(class_id)
                data_array.append(data)

        if len(bounding_boxes) == 0:
            print("All holds were removed!")
            return Detections(xyxy=np.array([]),
                              confidence=np.array([]),
                              class_id=np.array([]),
                              data={'class_name': np.array([])})

        # Convert lists to NumPy arrays
        bounding_boxes = np.array(bounding_boxes)
        confidence_array = np.array(confidence_array)
        class_id_array = np.array(class_id_array)
        data_array = np.array(data_array)

        # Update the Detections instance with the modified detections
        updated_detections_instance = Detections(
            xyxy=bounding_boxes,
            confidence=confidence_array,
            class_id=class_id_array,
            data={'class_name': data_array}
        )
        print("DONE REMOVING HOLDS FROM THE ROUTE!")
        return updated_detections_instance
    else:
        print("No holds removed from the route!")
        return route_to_update