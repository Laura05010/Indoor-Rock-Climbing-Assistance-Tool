import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ultralytics import YOLO
import supervision as sv

from queue import Queue
import threading
import time

import calibrate
import find_routes
import audio_feedback
import audio_input
import time

from pynput import keyboard

# Defining global variables
R_FOOT = ["right_ankle", "right_heel", "right_foot_index"]
L_FOOT = ["left_ankle", "left_heel", "left_foot_index"]
R_HAND = ["right_pinky", "right_index", "right_thumb", "right_wrist"]
L_HAND = ["left_pinky", "left_index", "left_thumb", "left_wrist"]

HAND_FOOT = 0
RIGHT_LEFT = 0

TARGET_HOLD = None

# selected_limb = 'right_hand'
limb_lock = threading.Lock()

def on_press(key):
    # global selected_limb
    global HAND_FOOT
    global RIGHT_LEFT

    try:
        # Check if the pressed key is one of the designated keys
        if key.char == 'r':
            with limb_lock:
                RIGHT_LEFT = 0
        elif key.char == 'l':
            with limb_lock:
                RIGHT_LEFT = 1
        # Add other limbs...
        if key.char == 'h':
            with limb_lock:
                HAND_FOOT = 0
        elif key.char == 'f':
            with limb_lock:
                HAND_FOOT = 1
    except AttributeError:
        pass  # Handle special keys here if needed

def check_grab_hold(limb, hold, grabbed_areas, GRAB_THRESHOLD):
    start_time = time.time()

    while time.time() - start_time < 3:
        current_distance = get_relative_distance(limb, hold)
        if current_distance > GRAB_THRESHOLD:
            return  # Exit the function if the distance exceeds the threshold

        time.sleep(0.1)

    # If the loop completes, it means the hold is grabbed
    with limb_lock:  # Use lock for thread safety
        if not is_exact_detection_in_list(hold, grabbed_areas):
            grabbed_areas.append(hold)
            # audio_feedback.calibrated_sound()

def calculate_angle(a,b,c):
    # First, Mid, End
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],
                                                            a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if 180.0 < angle:
        angle = 360 - angle
        
    return angle 

def display_hand(image, hand_pts):
    # DISPLAY AREA OF RIGHT HAND ---------------
    # Calculate the center of the circle in 3D space (x, y, z)
    center_3d = np.mean(hand_pts, axis=0)
    # Calculate the radius of the circle in 3D space based on the average 
    # distance from the center to each point
    distances = [np.linalg.norm([p[0] - center_3d[0], 
                                 p[1] - center_3d[1], 
                                 p[2] - center_3d[2]]) for p in hand_pts]

    scaling_factor = 3  # scaling factor must be int
    radius_3d = scaling_factor * int(sum(distances) / len(distances))

    # Draw the circle in 3D space (-1 thickness for a filled circle)
    cv2.circle(image, (int(center_3d[0]), int(center_3d[1])), radius_3d, 
               (245, 117, 66), thickness=-1)

def foot_pts(ankle, heel, index, frame_shape_1, frame_shape_0):
    return np.array([
        [int(ankle.x * frame_shape_1), int(ankle.y * frame_shape_0)],
        [int(heel.x * frame_shape_1), int(heel.y * frame_shape_0)],
        [int(index.x * frame_shape_1), int(index.y * frame_shape_0)]
        ], np.int32)

def hand_pts(pinky, index, thumb, wrist, frame_shape_1, frame_shape_0):
    return np.array([
        [int(pinky.x * frame_shape_1), int(pinky.y * frame_shape_0)],
        [int(index.x * frame_shape_1), int(index.y * frame_shape_0)],
        [int(thumb.x * frame_shape_1), int(thumb.y * frame_shape_0)],
        [int(wrist.x * frame_shape_1), int(wrist.y * frame_shape_0)]
        ], np.int32)

def display_coords(d):
    max_key_length = max(len(key) for key in d.keys())
    max_x_length = max(len(str(value.x)) for value in d.values())
    max_y_length = max(len(str(value.y)) for value in d.values())
    max_z_length = max(len(str(value.z)) for value in d.values())

    for point, coords in d.items():
        formatted_x = str(coords.x).rjust(max_x_length)
        formatted_y = str(coords.y).rjust(max_y_length)
        formatted_z = str(coords.z).rjust(max_z_length)
        print(f"{point.ljust(max_key_length)}: x = {formatted_x}, y = {formatted_y}, z = {formatted_z}")
    print("\n")

def is_within_hold(limb, detection):
    x, y = limb.x, limb.y
    x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
    # Check if limb coordinates are within the bounding box
    return x1 <= x <= x2 and y1 <= y <= y2

# TODO: function that checks what holds the person is on
# A hold corresponding to right hand, left hand, right foot, left foot
def get_curr_position(d, detections):
    extremities = ["right_foot", "left_foot", "right_hand", "left_hand"]
    for limb, coords in d.items():
        for detection in detections:
            # Within bounds
            if (limb in extremities) and is_within_hold(coords, detection):
                # save the coordinates
                pass

def find_closest_hold(hand_point, detections, grabbed_areas):
    closest_detection = None
    min_distance = float('inf')
    print(f"Detections: {detections}\n", end='\r')
    # print(f"Grabbed areas: {grabbed_areas}\n", end='\r')
    for detection in detections:
        if is_exact_detection_in_list(detection, grabbed_areas):
            continue

        distance = get_relative_distance(hand_point, detection)
        if distance < min_distance:
            min_distance = distance
            closest_detection = detection
    return closest_detection

def is_exact_detection_in_list(target_detection, detection_list):
    for detection in detection_list:
        # Assuming each detection is structured as [rock_hold_pos, ...]
        det_coords = detection[0]  # Extract coordinates of the detection
        target_coords = target_detection[0]  # Extract coordinates of the target detection
        if np.array_equal(det_coords, target_coords):
            return True
    return False

def get_center_point(d, limb, right_foot_pts, left_foot_pts, right_hand_pts,
                     left_hand_pts):
    if limb in R_FOOT:
        return np.mean(right_foot_pts, axis=0)
    elif limb in L_FOOT:
        return np.mean(left_foot_pts, axis=0)
    elif limb in R_HAND:
        return np.mean(right_hand_pts, axis=0)
    elif limb in L_HAND:
        return np.mean(left_hand_pts, axis=0)
    return np.array([d[limb].x, d[limb].y], np.int32)

def get_relative_distance(center_limb_pt, rock_hold):
    # points of rock_hold
    rock_hold_pos = rock_hold[0]
    x1, y1, x2, y2 = \
        rock_hold_pos[0], rock_hold_pos[1], rock_hold_pos[2], rock_hold_pos[3]
    # print("Rock_coords:", x1, y1, x2, y2)
    mean_rock_coord = np.mean(np.array([[x1, y1], [x2, y2]]), axis=0)
    # print("M:", mean_rock_coord)
    # print("C:", center_limb_pt[:2])
    return np.linalg.norm(abs(center_limb_pt[:2] - mean_rock_coord))

def audio_feedback_manager(audio_queue):
    while True:
        distance = audio_queue.get()
        audio_feedback.play_distance(distance)
        audio_queue.task_done()

def audio_input_manager():
    global HAND_FOOT
    global RIGHT_LEFT

    while True:
        print("Here")
        new_hf, new_rl = audio_input.input_audio()
        if new_hf != -1 and new_rl != -1:
            HAND_FOOT, RIGHT_LEFT = new_hf, new_rl
        print("TESTING:", HAND_FOOT, RIGHT_LEFT)

# def pose_est_hold_detect():
def pose_est_hold_detect(audio_queue):
    global HAND_FOOT
    global RIGHT_LEFT
    global TARGET_HOLD

    # JUST THE POSE
    global selected_limb
    cap = cv2.VideoCapture(0)

    model = YOLO('bestHuge.pt')
    # box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    dark_grey= sv.Color(64, 64, 64)
    box_annotator = sv.BoxAnnotator(color=dark_grey, thickness=2,
                                    text_thickness=2, text_scale=1)
    detections = []
    next_target_hold = None
    grabbed_areas = []  # List to store the coordinates of grabbed holds
    GRAB_AREA_THRESHOLD = 50  # Define a proximity threshold
    GRAB_THRESHOLD = 100  # How far away does the hand need to be to constitute a grab

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.8,
                      min_tracking_confidence=0.8) as pose:
        start_time = time.time()
        calibrated = False # Keeps track if you are at calibration phrase

        frame_counter = 0
        routes = {}
        while cap.isOpened():
            ret, frame = cap.read()
            # frame = cv2.imread('test_images/test_1.jpg') # for testing specific images

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            if not calibrated:
                detections, frame, image, calibrated = \
                    calibrate.calibrate_holds(start_time, detections, model,
                                              frame, box_annotator, image,
                                              calibrated)
                find_routes.identify_routes(image, detections)
                temp_routes = find_routes.identify_routes(image, detections)
                if len(temp_routes) > len(routes): # get the max amount of routes
                    routes = temp_routes
                if calibrated:
                    selected_route, route_color, colour_name = find_routes.get_user_route(image,routes)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    selected_route = find_routes.add_detections(frame, selected_route, route_color, colour_name)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    selected_route = find_routes.remove_detections(frame, selected_route, route_color, colour_name)
                    # print(selected_route) 
                    audio_feedback.calibrated_sound()
                    detections = selected_route # UPDATE DETECTIONS WITH FINAL ROUTE

            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # annotate the scene with the selected route's detections
                box_annotator = sv.BoxAnnotator(color=route_color, thickness=3,
                                    text_thickness=2, text_scale=1)
                frame = box_annotator.annotate(scene=image,
                                               detections=selected_route,
                                               skip_label=True)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(
                        color=(245,66,230), thickness=2, circle_radius=2))

                left_thumb_point = None
                right_thumb_point = None

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    pose_landmark = mp_pose.PoseLandmark

                    d = {}  # body dictionary

                    # Upper body coordinates
                    d["left_shoulder"] = \
                        landmarks[pose_landmark.LEFT_SHOULDER.value]
                    d["right_shoulder"] = \
                        landmarks[pose_landmark.RIGHT_SHOULDER]

                    d["left_elbow"] = landmarks[pose_landmark.LEFT_ELBOW.value]
                    d["right_elbow"] = \
                        landmarks[pose_landmark.RIGHT_ELBOW.value]

                    frame_shape_0, frame_shape_1 = \
                        frame.shape[0], frame.shape[1]

                    # LEFT HAND
                    d["left_pinky"] = landmarks[pose_landmark.LEFT_PINKY.value]
                    d["left_index"] = landmarks[pose_landmark.LEFT_INDEX.value]
                    d["left_thumb"] = landmarks[pose_landmark.LEFT_THUMB.value]
                    d["left_wrist"] = landmarks[pose_landmark.LEFT_WRIST.value]
                    left_hand_pts = hand_pts(d["left_pinky"], d["left_index"], 
                                             d["left_thumb"], d["left_wrist"], 
                                             frame_shape_1, frame_shape_0)

                    # RIGHT HAND
                    d["right_pinky"] = \
                        landmarks[pose_landmark.RIGHT_PINKY.value]
                    d["right_index"] = \
                        landmarks[pose_landmark.RIGHT_INDEX.value]
                    d["right_thumb"] = \
                        landmarks[pose_landmark.RIGHT_THUMB.value]
                    d["right_wrist"] = \
                        landmarks[pose_landmark.RIGHT_WRIST.value]
                    right_hand_pts = hand_pts(d["right_pinky"],
                                              d["right_index"],
                                              d["right_thumb"],
                                              d["right_wrist"],
                                              frame_shape_1, frame_shape_0)

                    # Lower body coordinates
                    d["left_knee"] = landmarks[pose_landmark.LEFT_KNEE.value]
                    d["right_knee"] = landmarks[pose_landmark.RIGHT_KNEE.value]

                    # LEFT FOOT
                    d["left_ankle"] = landmarks[pose_landmark.LEFT_ANKLE.value]
                    d["left_heel"] = landmarks[pose_landmark.LEFT_HEEL.value]
                    d["left_foot_index"] = \
                        landmarks[pose_landmark.LEFT_FOOT_INDEX.value]
                    left_foot_pts = foot_pts(d["left_ankle"], d["left_heel"],
                                             d["left_foot_index"],
                                             frame_shape_1, frame_shape_0)

                    # RIGHT FOOT
                    d["right_ankle"] = \
                        landmarks[pose_landmark.RIGHT_ANKLE.value]
                    d["right_heel"] = landmarks[pose_landmark.RIGHT_HEEL.value]
                    d["right_foot_index"] = \
                        landmarks[pose_landmark.RIGHT_FOOT_INDEX.value]
                    right_foot_pts = foot_pts(d["right_ankle"], d["right_heel"],
                                              d["right_foot_index"],
                                              frame_shape_1, frame_shape_0)
                    
                    right_thumb_point = get_center_point(d, "right_thumb",
                                             right_foot_pts, left_foot_pts,
                                             right_hand_pts, left_hand_pts)
                    
                    left_thumb_point = get_center_point(d, "left_thumb",
                                             right_foot_pts, left_foot_pts,
                                             right_hand_pts, left_hand_pts)


                    # Display Coordinates
                    # display_coords(d)

                    extremities = [[right_hand_pts, left_hand_pts],
                                   [right_foot_pts, left_foot_pts]]

                    # for detection in detections:
                    #     point = np.mean(extremities[HAND_FOOT][RIGHT_LEFT], 
                    #                     axis=0)
                    #     distance = get_relative_distance(point, detection)
                    #     if frame_counter % 5 == 0:
                    #         audio_queue.put(distance)
                    #     print(f"Relative position: {distance} " + " " * 20, end='\r')
                    if TARGET_HOLD is not None:
                        point = np.mean(extremities[HAND_FOOT][RIGHT_LEFT], 
                                        axis=0)
                        distance = get_relative_distance(point, TARGET_HOLD)
                        if frame_counter % 5 == 0:
                            audio_queue.put(distance)
                    

                    # center_2d = np.mean(right_hand_pts, axis=0)[:2]

                    cv2.fillPoly(image, [right_foot_pts], (245, 117, 66))
                    cv2.fillPoly(image, [left_foot_pts], (245, 117, 66))

                    display_hand(image,right_hand_pts)
                    display_hand(image,left_hand_pts)

                    # Calculate angle
                    # angle = calculate_angle(shoulder, elbow, wrist)
                    # Visualize angle
                    # cv2.putText(image, str(angle),
                    #                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    #                     )

                except:
                    pass

                try:
                    print("--------------------\n", end='\r')
                    # print(RIGHT_LEFT, HAND_FOOT)

                    global selected_limb

                    limb = None

                    # if selected_limb == 'right_hand':
                    if RIGHT_LEFT == 0 and HAND_FOOT == 0:
                        print("Right hand selected\n", end='\r')
                        limb = right_thumb_point
                    # elif selected_limb == 'left_hand':
                    elif RIGHT_LEFT == 1 and HAND_FOOT == 0:
                        print("Left hand selected\n", end='\r')
                        limb = left_thumb_point
                    elif RIGHT_LEFT == 0 and HAND_FOOT == 1:
                        print("Right foot selected\n", end='\r')
                        limb = left_thumb_point
                    elif RIGHT_LEFT == 1 and HAND_FOOT == 1:
                        print("Left foot selected\n", end='\r')
                        limb = left_thumb_point

                    # Find the closest hold that hasn't been grabbed yet
                    next_target_hold = find_closest_hold(limb, 
                                                        detections, 
                                                        grabbed_areas)

                    next_target_hold = list(next_target_hold)

                    TARGET_HOLD = next_target_hold

                    print(f"Next target hold: {next_target_hold}\n", end='\r')
                    print(f"Grabbed areas: {grabbed_areas}\n", end='\r')

                    distance_to_next_hold = get_relative_distance(
                        limb, next_target_hold)
                    print(f"Distance to next hold: {distance_to_next_hold} units\n", end='\r')

                    if distance_to_next_hold < GRAB_THRESHOLD:
                        threading.Thread(target=check_grab_hold, 
                                 args=(limb, TARGET_HOLD, grabbed_areas, GRAB_THRESHOLD)).start()
                        # print("Hold grabbed!\n", end='\r')

                        # if not is_exact_detection_in_list(next_target_hold, 
                        #                                 grabbed_areas):
                        #     grabbed_areas.append(next_target_hold)
                        #     # audio_feedback.calibrated_sound()

                    print("--------------------\n", end='\r')

                except Exception as exception:
                    print(exception)

                # print(results.pose_landmarks)
                # Render detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(
                        color=(245,66,230), thickness=2, circle_radius=2))
                
                frame_counter = (frame_counter + 1) % 5

                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                #                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                #                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            cv2.imshow('Pose Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    # Begin audio feedback thread
    audio_queue = Queue()
    # detection_thread = threading.Thread(target=pose_est_hold_detect, 
    #                                     args=(audio_queue, ))
    audio_feedback_thread = threading.Thread(target=audio_feedback_manager, 
                                    args=(audio_queue, ), daemon=True)
    audio_input_thread = threading.Thread(target=audio_input_manager,
                                          daemon=True)

    # detection_thread.start()
    audio_feedback_thread.start()
    audio_input_thread.start()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # threading.Thread(target=pose_est_hold_detect, args=(audio_queue,)).start()

    # pose_est_hold_detect()
    pose_est_hold_detect(audio_queue)

if "__main__" == __name__:
    main()
