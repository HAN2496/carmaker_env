import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Redefine the functions and reprocess the image to obtain the approximate contour of the road
image_path = 'datafiles/road.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
road_color = np.array([158, 226, 190])
mask = cv2.inRange(image_rgb, road_color - np.array([10, 10, 10]), road_color + np.array([10, 10, 10]))


# Function to calculate the angle between two vectors
def calculate_angle(v1, v2):
    # Calculate the dot product
    dot_prod = np.dot(v1, v2)
    # Calculate the magnitude of the vectors
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    # Calculate the angle in radians
    angle_rad = np.arccos(dot_prod / (mag_v1 * mag_v2))
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to calculate the direction vector between two points
def calculate_direction_vector(point1, point2):
    return np.array(point2) - np.array(point1)

# Reload the original image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find contours from the mask again
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the road
largest_contour = max(contours, key=cv2.contourArea)

# Simplify the contour to get key points
epsilon = 0.005 * cv2.arcLength(largest_contour, True)
approximate_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

# Draw the simplified contour on the original image to create road_with_approx_contour
road_with_approx_contour = image_rgb.copy()
cv2.drawContours(road_with_approx_contour, [approximate_contour], -1, (255, 0, 0), 2)

# Calculate the direction vectors for each segment
approximate_contour_points = approximate_contour[:, 0, :]  # Reshape the contour array

df = pd.DataFrame()

direction_vectors = [calculate_direction_vector(approximate_contour_points[i], approximate_contour_points[i + 1])
                     for i in range(len(approximate_contour_points) - 1)]

# Calculate angles between each pair of consecutive direction vectors
turn_angles = [calculate_angle(direction_vectors[i], direction_vectors[i + 1])
               for i in range(len(direction_vectors) - 1)]

# Display the image with the approximate contour of the road and annotate angles
plt.figure(figsize=(10, 5))
plt.imshow(road_with_approx_contour)
# Annotate the angles on the image
for i, point in enumerate(approximate_contour_points[:-1]):
    plt.text(point[0], point[1], str(i+1), color='red', fontsize=12)
    if i < len(turn_angles):
        plt.text((point[0] + approximate_contour_points[i + 1][0]) / 2,
                 (point[1] + approximate_contour_points[i + 1][1]) / 2,
                 f"{turn_angles[i]:.1f}°", color='blue', fontsize=12)
plt.axis('off')  # Turn off axis numbers
plt.show()

# The angles calculated for each turn
turn_angles
