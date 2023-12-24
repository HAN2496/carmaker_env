import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

contour_points_df = pd.read_csv('datafiles/contour_points.csv')
# Extracting the x and y coordinates of the points marked in red
x_coords = contour_points_df['x']
y_coords = contour_points_df['y']

# Plotting the points
plt.figure(figsize=(10, 5))
plt.plot(x_coords, y_coords, 'ro-')  # 'ro-' means red color, circle markers, and solid line
plt.title("Plot of Red Points on the Road Contour")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.gca().invert_yaxis()  # Invert y-axis to match the image orientation
plt.grid(True)
plt.show()
