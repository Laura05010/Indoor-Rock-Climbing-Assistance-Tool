import cv2
from ultralytics import YOLO
import supervision as sv
from collections import Counter
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
detections = []
test_image = 'test_images/test_3.jpg'

def get_detections(frame):
    model = YOLO('bestHuge.pt')
    detections = sv.Detections.from_ultralytics(model(frame, verbose=False)[0])
    detections = detections[detections.confidence > 0.5]
    return detections

# rounding pixel coord to nearest 10th
def round_pixel(pixel):
    return (round(pixel[0], -1), round(pixel[1], -1), round(pixel[2], -1))


# def process_detection(detection):
#     image = cv2.imread('test_1.jpg')
#     detection_coordinates = detection[0]
#     x1, y1, x2, y2 = map(int, detection_coordinates)
#     temp = image[y1:y2, x1:x2, :]

#     lst = [round_pixel(pixel) for row in temp for pixel in row]
#     c = Counter(lst)
#     return c.most_common(1)[0][0]

def process_detection(detection, background_threshold=0.5):
    image = cv2.imread(test_image)
    detection_coordinates = detection[0]
    x1, y1, x2, y2 = map(int, detection_coordinates)
    temp = image[y1:y2, x1:x2, :]

    lst = [round_pixel(pixel) for row in temp for pixel in row]
    c = Counter(lst)

    # Check if the most common color is too close to the background count
    most_common, count = c.most_common(1)[0]
    total_pixels = len(lst)
    if count / total_pixels < background_threshold:
        # Get the second most common color
        second_most_common = c.most_common(2)[1][0]
        return second_most_common
    else:
        return most_common


def average_color(color_list):
    # Initialize variables to store sums of R, G, B components
    total_r = 0
    total_g = 0
    total_b = 0

    # Iterate through each tuple in the list
    for color_tuple in color_list:
        total_r += color_tuple[0]
        total_g += color_tuple[1]
        total_b += color_tuple[2]

    # Calculate the average of each component
    avg_r = total_r // len(color_list)
    avg_g = total_g // len(color_list)
    avg_b = total_b // len(color_list)
    return sv.Color(int(avg_r), int(avg_g), int(avg_b))

def visualize_detections(frame, detections, labels):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    for k, anon in box_annotators.items():
        detections = detections[[labels[i] == k for i in range(len(detections))]]
        frame = box_annotators[k].annotate(scene=image, detections=detections, skip_label=True)

    cv2.imshow('Detections', frame)
    cv2.waitKey()
    return detections


if __name__ == "__main__":
    frame = cv2.imread(test_image)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    detections = get_detections(image)

    with Pool() as pool:
        holds_by_colour = list(pool.map(process_detection, detections)) # mutiprocessing , makes code run 10 secs faster

    # DBSCAN parameters
    eps = 30  # Adjust epsilon
    algorithm = 'auto'  # Choose algorithm

    # DBSCAN clustering
    km = DBSCAN(eps=eps, algorithm=algorithm)
    km.fit(holds_by_colour)
    labels = km.labels_  # Your DBSCAN labels
    print("labels",labels)
    print(holds_by_colour)


    annotator_colours = {}
    for label, color_tuple in zip(labels, holds_by_colour):
        annotator_colours.setdefault(label, []).append(color_tuple)

    print("DICTIONARY=============")
    # print(annotator_colours)

    box_annotators = {k: sv.BoxAnnotator(color=average_color(color), thickness=4, text_thickness=4, text_scale=1) for k, color in annotator_colours.items()}
    print(box_annotators)

    visualize_detections(image,detections,labels)
