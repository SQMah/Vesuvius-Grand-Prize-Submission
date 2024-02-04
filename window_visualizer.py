from typing import List

import cv2
import numpy as np
from ink_label_processor import window_is_valid, get_ink_text_bounding_boxes, BoundingBox
import albumentations as A
import matplotlib.pyplot as plt
from custom_augmentations import FourthAugment


def process_image(image_data, window_size, stride, white_threshold, visualize=False):
    # Load the grayscale image
    image = image_data
    ink_bounding_boxes = get_ink_text_bounding_boxes(image)

    # Dimensions of the image
    height, width = image.shape

    # Count of boxes
    box_count = 0

    # Create a copy for drawing boxes
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert to binary (white and black)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    valid_boxes = []

    # Iterate over the image
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Extract the window
            window_bounding_box = BoundingBox(x, y, x + window_size, y + window_size)

            # Check if white percentage is above the threshold
            if window_is_valid(window_bounding_box, ink_bounding_boxes, white_threshold):
                # Draw a box
                cv2.rectangle(output_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 20)
                box_count += 1
                valid_boxes.append(window_bounding_box)

    print(f"Number of boxes drawn: {box_count}")
    # Display the result
    if visualize:
        cv2.imshow('Boxes on Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return valid_boxes


def visualize_augmentations(img_data, size: int, boxes: List[BoundingBox]):
    """Use matplotlib to visualize the augmentations in a grid, based on the boxes returned from process_image"""
    augmentations_list = [
        # A.ToFloat(max_value=65535.0),
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        FourthAugment(p=1.0),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),
        # A.ChannelShuffle(p=0.5),
        A.GridDistortion(p=0.5, num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                         normalized=True),
        A.ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=50, border_mode=cv2.BORDER_CONSTANT, value=0.0),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.75
        ),
        A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.15, p=0.9,
                           border_mode=cv2.BORDER_CONSTANT, value=0.0),
        A.OneOf([
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
            # A.OpticalDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            A.PiecewiseAffine(p=0.5),  # IAAPiecewiseAffine
            A.MotionBlur(),
        ], p=0.9),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2),
                        mask_fill_value=0, p=0.5),
    ]
    augmentations = A.Compose(augmentations_list)
    box_data = [box.get_img_from_box(img_data) for box in boxes]
    empty_image = np.zeros((size, size), dtype=np.uint8)
    augmented_images = [augmentations(image=empty_image, mask=box)[
                            'mask'] for box in box_data]  # Apply the augmentations to the images in the boxes
    # Display original images using matplotlib
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(box_data[i - 1])
    plt.show()
    # Display grid of augmented images using matplotlib
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(augmented_images[i - 1])
    plt.show()


if __name__ == '__main__':
    # Example usage
    image_path = './labels/20231012184423_inklabels.png'
    window_size = 256  # Example window size
    stride = 256  # Example stride length
    white_threshold = 0.5  # Fraction threshold of white
    visualize_bboxes = True

    # Process the image
    img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    boxes = process_image(img_data, window_size, stride, white_threshold, visualize=visualize_bboxes)
    if not visualize_bboxes:
        visualize_augmentations(img_data, window_size, boxes)
