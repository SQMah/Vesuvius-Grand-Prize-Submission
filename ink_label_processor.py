import os

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from typing import List

from cfg import CFG

label_dir = CFG.processed_labels_dir


class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1  # top left x
        self.y1 = y1  # top left y
        self.x2 = x2  # bottom right x
        self.y2 = y2  # bottom right y

    def __str__(self):
        return f"({self.x1}, {self.y1}) ({self.x2}, {self.y2})"

    def __repr__(self):
        return f"({self.x1}, {self.y1}) ({self.x2}, {self.y2})"

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def calculate_overlap_area(self, box2):
        # Calculate the (x, y) coordinates of the intersection rectangle
        x_left = max(self.x1, box2.x1)
        y_top = max(self.y1, box2.y1)
        x_right = min(self.x2, box2.x2)
        y_bottom = min(self.y2, box2.y2)

        # Check if there is no overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The overlap area
        return (x_right - x_left) * (y_bottom - y_top)

    def get_xyxy(self):
        return self.x1, self.y1, self.x2, self.y2

    @staticmethod
    def overlaps(box1: 'BoundingBox', box2: 'BoundingBox') -> bool:
        A_x1, A_y1, A_x2, A_y2 = box1.x1, box1.y1, box1.x2, box1.y2
        B_x1, B_y1, B_x2, B_y2 = box2.x1, box2.y1, box2.x2, box2.y2

        # Check if Box A is to the left of Box B
        if A_x2 < B_x1:
            return False

        # Check if Box A is to the right of Box B
        if A_x1 > B_x2:
            return False

        # Check if Box A is above Box B
        if A_y2 < B_y1:
            return False

        # Check if Box A is below Box B
        if A_y1 > B_y2:
            return False

        # If none of the above, the boxes overlap
        return True

    @staticmethod
    def from_xyxy(input_tuple):
        return BoundingBox(*input_tuple)

    @staticmethod
    def from_xywh(input_tuple):
        return BoundingBox(input_tuple[0], input_tuple[1], input_tuple[0] + input_tuple[2],
                           input_tuple[1] + input_tuple[3])

    @staticmethod
    def combine_overlapping_boxes(input_boxes: List['BoundingBox']) -> List['BoundingBox']:
        if not input_boxes:
            return []

        combined_boxes = []
        input_boxes = sorted(input_boxes, key=lambda b: (b.x1, b.y1))

        # Start with the first box
        current_box = input_boxes[0]

        for box in input_boxes[1:]:
            # Check if current box overlaps with the next box
            if BoundingBox.overlaps(current_box, box):
                # Combine the current box with the overlapping box
                new_x1 = min(current_box.x1, box.x1)
                new_y1 = min(current_box.y1, box.y1)
                new_x2 = max(current_box.x2, box.x2)
                new_y2 = max(current_box.y2, box.y2)
                current_box = BoundingBox(new_x1, new_y1, new_x2, new_y2)
            else:
                # No overlap, add the current box to the list and move to the next
                combined_boxes.append(current_box)
                current_box = box

        # Add the last combined box
        combined_boxes.append(current_box)

        return combined_boxes

    def get_img_from_box(self, img_data):
        return img_data[self.y1: self.y2, self.x1: self.x2]


def window_is_valid(window_img_bbox: BoundingBox, ink_bounding_boxes: List[BoundingBox],
                    box_contain_threshold=0.5):
    window_area = window_img_bbox.area()

    for bounding_box in ink_bounding_boxes:
        overlap_area = window_img_bbox.calculate_overlap_area(bounding_box)
        if overlap_area / window_area >= box_contain_threshold:
            return True

    return False


def get_ink_text_bounding_boxes(img, dilation_horizontal=220, dilation_vertical=1, visualize=False) -> List[
    BoundingBox]:
    # Threshold the image to get a binary mask of the white shapes
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

    # Dilate to merge close white regions
    kernel = np.ones((dilation_vertical, dilation_horizontal), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes for these contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [BoundingBox.from_xywh(box) for box in bounding_boxes]
    bounding_boxes = [BoundingBox(box.x1 + dilation_horizontal // 4, box.y1, box.x2 - dilation_horizontal // 4,
                                  box.y2) for box in
                      bounding_boxes]

    if visualize:
        # Draw the bounding boxes on the image
        for box in bounding_boxes:
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 5)
        cv2.imshow("Bounding boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bounding_boxes


# def draw_line_from_points(points, img):
#     # If there's only one point, can't determine a line
#     if len(points) <= 1:
#         return
#
#     # Separate the list of points into X and Y coordinates for linear regression
#     X, Y = zip(*points)
#     X = np.array(X).reshape(-1, 1)
#     Y = np.array(Y)
#
#     # Linear regression to find best fit line for the centroids
#     reg = LinearRegression().fit(X, Y)
#
#     # Compute the start and end coordinates of the line for drawing
#     start_point = (0, int(reg.predict([[0]])[0]))
#     end_point = (img.shape[1], int(reg.predict([[img.shape[1]]])[0]))
#
#     cv2.line(img, start_point, end_point, (0, 0, 255), 2)  # Drawing line in red color


# def cluster_boxes_into_lines(boxes, eps=600, min_samples=3):
#     # Calculate centroids for each box
#     centroids = [(x + w / 2, y + h / 2) for x, y, w, h in boxes]
#
#     # Use DBSCAN clustering to cluster centroids
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
#
#     lines = {}
#     for label, box in zip(clustering.labels_, boxes):
#         if label in lines:
#             lines[label].append(box)
#         else:
#             lines[label] = [box]
#
#     print(lines)
#
#     return list(lines.values())
#
#
# def get_hough_lines(centroids, img_shape, threshold):
#     blank = np.zeros(img_shape, np.uint8)
#     for (x, y) in centroids:
#         cv2.circle(blank, (int(x), int(y)), 1, (255, 255, 255), -1)
#
#     lines = cv2.HoughLines(blank, 1, np.pi / 180, threshold)
#     return lines
#
#
# def cluster_boxes_with_hough(boxes, lines, dist_threshold=10):
#     if lines is None:
#         return []
#
#     clusters = {}
#     for rho, theta in lines[:, 0]:
#         cos_t, sin_t = np.cos(theta), np.sin(theta)
#         for box in boxes:
#             x, y, w, h = box
#             cx, cy = x + w / 2, y + h / 2
#             distance = abs(cx * cos_t + cy * sin_t - rho)
#             if distance < dist_threshold:  # threshold distance for a box to belong to a line
#                 if (rho, theta) in clusters:
#                     clusters[(rho, theta)].append(box)
#                 else:
#                     clusters[(rho, theta)] = [box]
#
#     return list(clusters.values())


def get_img_data_with_certain_non_ink(img_data, boxes, certain_color=CFG.certain_no_ink_color):
    new_img_data = np.zeros(img_data.shape, np.uint8)
    for x, y, w, h in boxes:
        bbox_data = img_data[y: y + h, x: x + w]
        bbox_data[bbox_data == 0] = certain_color
        new_img_data[y: y + h, x: x + w] = bbox_data
    return new_img_data


if __name__ == "__main__":
    # Test the function
    # image_path = './orig_labels/20230827161847_inklabels.png'
    # img = cv2.imread(image_path)
    # h, w = img.shape[:2]
    # boxes = get_white_shape_bounding_boxes(img)
    #
    # img_data_with_certain_non_ink = get_img_data_with_certain_non_ink(img, boxes)
    #
    # # Visualize
    # for x, y, w, h in boxes:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    #
    # cv2.imshow("Bounding boxes", img)
    # cv2.imshow("Resulting image", img_data_with_certain_non_ink)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for image_name in os.listdir(label_dir):
        if image_name.endswith('.png'):
            print(f"Processing {image_name}")
            image_path = os.path.join(label_dir, image_name)
            img = cv2.imread(image_path, 0)
            h, w = img.shape[:2]
            boxes = get_ink_text_bounding_boxes(img)

            img_data_with_certain_non_ink = get_img_data_with_certain_non_ink(img, boxes)

            cv2.imshow(image_name, img_data_with_certain_non_ink)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
