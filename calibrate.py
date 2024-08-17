import cv2
import supervision as sv
import time
import numpy as np

def calibrate_holds(start_time, detections, model, frame, box_annotator, image, 
                    calibrated):
    print(f"Calibrating... " + " " * 20, end='\r')
    # check if it's been 10 seconds
    # break when you reach 10 seconds

    calibrate_time = 20
    # check if it's been calibrate_time seconds
    # break when you reach calibrate_time seconds
    elapsed_time = time.time() - start_time

    if elapsed_time <= calibrate_time:
        detections = sv.Detections.from_ultralytics(model(frame,
                                                          verbose=False)[0])
        detections = detections[detections.confidence > 0.75]
        # frame = box_annotator.annotate(scene=image, detections=detections, 
        #                                skip_label=True)
        #print(detections)
        frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        
        # Ensure frame is writable
        frame = np.asarray(frame)
        if not frame.flags.writeable:
            frame.setflags(write=1)

        cv2.putText(frame, "Calibrating...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 100), 2)

        # cv2.putText(frame, "Calibrating...", (50, 50), fontScale=1, color=(0, 0, 100), thickness=2)

        print(f"Calibrating... " + " " * 20, end='\r')

        cv2.imshow('Calibrating', frame)  # Update the window
        elapsed_time = time.time() - start_time # Update elapsed time
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