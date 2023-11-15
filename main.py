import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ultralytics import YOLO
import supervision as sv

import time

import calibrate


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
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
    radius_3d = scaling_factor * int(sum(distances ) / len(distances))

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

def get_center_point(d, limb, right_foot_pts, left_foot_pts, right_hand_pts, 
                     left_hand_pts):
    r_foot = ["right_ankle", "right_heel", "right_foot_index"]
    l_foot = ["left_ankle", "left_heel", "left_foot_index"]
    r_hand = ["right_pinky", "right_index","right_thumb", "right_wrist"]
    l_hand = ["left_pinky", "left_index","left_thumb", "left_wrist"]
    if limb in r_foot:
        return np.mean(right_foot_pts, axis=0)
    elif limb in l_foot:
        return np.mean(left_foot_pts, axis=0)
    elif limb in r_hand:
        return np.mean(right_hand_pts, axis=0)
    elif limb in l_hand:
        return np.mean(left_hand_pts, axis=0)
    return np.array([d[limb].x, d[limb].y], np.int32)
    
# TODO: function that checks what holds the person is on
# A hold corresponding to right hand, left hand, right foot, left foot
def get_curr_position(d, detections):
    extremities = ["right_foot", "left_foot", "right_hand","left_hand"]
    for limb, coords in d.items():
        for detection in detections:
            # yields opposite corners
            x1, y1, x2, y2 = \
                detection[0], detection[1], detection[2], detection[3]
            limb_x, limb_y = coords.x, coords.y

            # Within bounds
            if (limb in extremities) and is_within_hold(coords, detection):
                # save the coordinates
                pass

def get_relative_position(center_limb_pt, rock_hold):
    # points of rock_hold
    x1, y1, x2, y2 = \
        rock_hold[0][0], rock_hold[0][1], rock_hold[0][2], rock_hold[0][3]
    # print("Rock_coords:", x1, y1, x2, y2)
    mean_rock_coord = np.mean(np.array([[x1, y1], [x2, y2]]), axis=0)
    # print("M:", mean_rock_coord)
    # print("C:", center_limb_pt[:2])
    return np.linalg.norm(abs(center_limb_pt[:2] - mean_rock_coord))

def pose_est_hold_detect():
    # JUST THE POSE
    cap = cv2.VideoCapture(0)

    model = YOLO('bestHuge.pt')
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    detections = []

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.8,
                      min_tracking_confidence=0.8) as pose:
        start_time = time.time()
        calibrated = False # Keeps track if you are at calibration phrase

        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            if not calibrated:
                detections, frame, image, calibrated = \
                    calibrate.calibrate_holds(start_time, detections, model, 
                                              frame, box_annotator, image, 
                                              calibrated)

            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # annotate the scene with the detections
                frame = box_annotator.annotate(scene=image, 
                                               detections=detections)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

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

                    # Display Coordinates
                    # display_coords(d)

                    for detection in detections:
                        # currently only for right_hand
                        point = get_center_point(d, "right_thumb", 
                                                 right_foot_pts, left_foot_pts, 
                                                 right_hand_pts, left_hand_pts)
                        get_relative_position(point, detection)
                        print(f"Relative position: {get_relative_position(point, detection)} " + " " * 20, end='\r')

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

                # print(results.pose_landmarks)
                # Render detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(
                        color=(245,66,230), thickness=2, circle_radius=2))

                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                #                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                #                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            cv2.imshow('Pose Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    for lndmrk in mp_pose.PoseLandmark:
        print(lndmrk)
    
    pose_est_hold_detect()

if "__main__" == __name__:
    main()
