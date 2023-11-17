# https://realpython.com/python-opencv-color-spaces/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
# COLOR RANGES

# Red color range
red_lower = np.array([0, 100, 100]) # GOOD
red_upper = np.array([10, 255, 255]) # GOOD

# Orange color range (including lighter tones)
orange_lower = np.array([1, 190, 200]) # GOOD
orange_upper = np.array([18, 255, 255]) # GOOD

# Yellow color range
yellow_lower = np.array([20, 100, 100]) # GOOD
yellow_upper = np.array([35, 255, 255]) # GOOD

# Green color range
green_lower = np.array([40, 50, 100])  # GOOD
green_upper = np.array([90, 255, 255]) # GOOD

# Blue color range
blue_lower = np.array([95, 50, 50]) # GOOD
blue_upper = np.array([120, 255, 255]) # GOOD

# Pink color range
pink_lower = np.array([155, 70, 200]) # GOOD
pink_upper = np.array([175, 255, 255]) # GOOD

# Purple color range
purple_lower = np.array([115, 40, 40])  # GOOD
purple_upper = np.array([145, 255, 255]) # GOOD

# White color range
white_lower = np.array([0, 0, 200]) # GOOD
white_upper = np.array([180, 30, 255]) # GOOD

# Black color range
black_lower = np.array([0, 0, 0]) # GOOD
black_upper = np.array([180, 55, 40]) # GOOD



def visualize_hsv_range(hsv_lower, hsv_upper):
    hue_values = np.linspace(hsv_lower[0], hsv_upper[0], 100)  # Creating a range of hue values
    saturation = hsv_upper[1]  # Keeping the saturation constant
    value = hsv_upper[2]  # Keeping the value constant

    # Generating an HSV spectrum using the hue range and constant saturation/value
    hsv_spectrum = np.array([[hue, saturation, value] for hue in hue_values], dtype=np.uint8)
    # Converting the HSV spectrum to RGB for visualization
    rgb_spectrum = cv2.cvtColor(np.reshape(hsv_spectrum, (1, -1, 3)), cv2.COLOR_HSV2RGB)
    # Reshaping the RGB spectrum for plotting
    rgb_spectrum = np.reshape(rgb_spectrum, (1, -1, 3))

    plt.imshow(rgb_spectrum)
    plt.xlabel("Hue")
    plt.ylabel("Saturation and Value")
    plt.title("HSV Color Range")
    plt.show()



visualize_hsv_range(orange_lower, orange_upper)
# visualize_limits(black_lower, black_upper)


