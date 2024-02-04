import random

import cv2
import numpy as np


def find_rotation_theta(img_data, show=False):
    image_lines = np.copy(img_data)

    h, w = img_data.shape[:2]

    # Preprocess the image
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    outside_edge = cv2.Canny(blurred, 320, 350)
    internal_edges = cv2.Canny(blurred, 150, 250) - outside_edge

    # Apply Hough Line Transformation
    lines = cv2.HoughLines(internal_edges, 1, np.pi / 180, int(0.0163 * max(h, w)))
    rhos = []
    thetas = []

    # Draw the detected lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            rhos.append(rho)
            thetas.append(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 20000 * (-b))
            y1 = int(y0 + 20000 * a)
            x2 = int(x0 - 20000 * (-b))
            y2 = int(y0 - 20000 * a)
            cv2.line(image_lines, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # Sort lines into two groups based on theta
    # Use the median of each group for further calculations
    median_theta_group_1 = random.choice(thetas)
    group1, group2 = [], []
    for theta in thetas:
        if np.abs(theta - median_theta_group_1) < np.pi / 2:
            group1.append(theta)
            median_theta_group_1 = np.median(group1)
        else:
            group2.append(theta)

    # Calculate the median of each group
    median_group1_theta = np.median([group1]) if len(group1) > 0 else None
    median_group2_theta = np.median([group2]) if len(group2) > 0 else None

    if show:
        cv2.imshow('Edges', internal_edges)
        cv2.imshow('Hough Line Transform', image_lines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (median_group1_theta if len(group1) > len(group2) else median_group2_theta) - np.pi / 2
