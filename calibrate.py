import cv2
import supervision as sv
import time

def calibrate_holds(start_time, detections, model, frame, box_annotator, image,
                    calibrated):
    print(f"Calibrating... " + " " * 20, end='\r')

    calibrate_time = 15
    # check if it's been calibrate_time seconds
    # break when you reach calibrate_time seconds
    elapsed_time = time.time() - start_time

    if elapsed_time <= calibrate_time:
        detections = sv.Detections.from_ultralytics(model(frame,
                                                          verbose=False)[0])
        detections = detections[detections.confidence > 0.5]
        frame = box_annotator.annotate(scene=image, detections=detections,
                                       skip_label=True)

        cv2.putText(frame, "Calibrating...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 100), 2)

        print(f"Calibrating... " + " " * 20, end='\r')

        cv2.imshow('Pose Detection', frame)  # Update the window
        # elapsed_time = time.time() - start_time # Update elapsed time
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        calibrated = True # stop calibrating after 30 seconds
        print(f"FINISHED CALIBRATING!" + " " * 20, end='\r')
    return detections, frame, image, calibrated

def main():
    print("Testing Calibration...")
    calibrate_holds(0, None, None)

if "__main__" == __name__:
    main()
