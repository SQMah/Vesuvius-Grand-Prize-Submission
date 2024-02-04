import os
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import shutil
from enum import Enum
from typing import Dict
from rotate_from_hough_lines import find_rotation_theta
from segment_utils import get_segment_id_paths_dict

# Set max concurrent workers to number of cpu cores
max_concurrent_processes = os.cpu_count()

# Set max concurrent threads to 65, for 65 layers
max_concurrent_threads = 65


class RotationMethod(Enum):
    HOUGH_LINES = "hough_lines"
    MANUAL = "manual"


def respectfully_rotate_image(img_data, rotate_degrees):
    h, w = img_data.shape[:2]
    center_x = w / 2
    center_y = h / 2
    rotate_radians = np.radians(rotate_degrees)

    new_width = int(abs(w * np.cos(rotate_radians)) + abs(h * np.sin(rotate_radians)))
    new_height = int(abs(w * np.sin(rotate_radians)) + abs(h * np.cos(rotate_radians)))

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotate_degrees, 1)

    # Adjust the rotation matrix to accommodate the new dimensions
    rotation_matrix[0, 2] += (new_width - w) // 2
    rotation_matrix[1, 2] += (new_height - h) // 2

    # Perform the rotation while keeping all data
    return cv2.warpAffine(img_data, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)


def rotate_image_and_save(img_path, rotate_degrees):
    print(f"Rotating {img_path} by {round(rotate_degrees, 2)} degrees")
    img_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    rotated_img_data = respectfully_rotate_image(img_data, rotate_degrees)
    shutil.move(img_path, f"{img_path}.old")
    cv2.imwrite(img_path, rotated_img_data)
    print(f"Rotated {img_path}")


def restore_old_image(img_path):
    if os.path.exists(f"{img_path}.old"):
        print(f"Restoring to original state for {img_path}")
        os.remove(img_path)
        shutil.move(f"{img_path}.old", img_path)


def restore_old_images_in_dir(img_dir):
    for filename in os.listdir(img_dir):
        if filename.endswith(".tif") or filename.endswith(".png"):
            restore_old_image(os.path.join(img_dir, filename))


def rotate_all_images_in_segment(segment_id: str, segment_paths_dict: Dict[str, str], rotations_path: str,
                                 label_path: str,
                                 method=RotationMethod.MANUAL):
    # Set up directories and paths
    segment_data_dir = segment_paths_dict[segment_id]
    segment_layers_dir = os.path.join(segment_data_dir, "layers")
    segment_label_path = os.path.join(label_path, f"{segment_id}_inklabels.png")

    # Find theta to rotate by method
    degrees_to_rotate = 0
    if method == RotationMethod.HOUGH_LINES:
        original_tif = list(filter(lambda x: x.endswith(".tif"), os.listdir(segment_data_dir)))[0]
        original_tif_img_path = os.path.join(segment_data_dir, original_tif)
        original_tif_data = cv2.imread(original_tif_img_path, cv2.IMREAD_UNCHANGED)
        degrees_to_rotate = find_rotation_theta(original_tif_data, show=False)
    elif method == RotationMethod.MANUAL:
        degree_to_rotate_clockwise_txt_path = os.path.join(rotations_path, f"{segment_id}.txt")
        if not os.path.exists(degree_to_rotate_clockwise_txt_path):
            degrees_to_rotate = 0
        else:
            degrees_to_rotate = -float(open(degree_to_rotate_clockwise_txt_path).read())

    # Skip if theta to rotate is 0
    if degrees_to_rotate == 0:
        print(f"Skipping {segment_data_dir} because theta to rotate is 0.")
        return

    # Check if the segment has already been rotated
    rotation_result_txt_path = os.path.join(segment_data_dir, "rotated.txt")
    if os.path.exists(rotation_result_txt_path):
        # Open the txt file and read the rotation angle
        with open(rotation_result_txt_path, "r") as f:
            prev_degrees = float(f.read())
        # If the rotation angle is the same as the current rotation angle, skip
        if prev_degrees == degrees_to_rotate:
            print(f"Skipping {segment_data_dir} because it has already been rotated by {prev_degrees}.")
            return
        # If the rotation angle is different, restore the old images and continue
        else:
            restore_old_images_in_dir(segment_data_dir)
            restore_old_images_in_dir(segment_layers_dir)
            restore_old_image(segment_label_path)
    else:
        restore_old_images_in_dir(segment_data_dir)
        restore_old_images_in_dir(segment_layers_dir)
        restore_old_image(segment_label_path)

    # Rotate the images
    for filename in os.listdir(segment_data_dir):
        if filename.endswith(".tif") or filename.endswith(".png") and not os.path.islink(os.path.join(
                segment_data_dir, filename)):
            # Use a thread pool executor to speed up the rotation
            with ThreadPoolExecutor(max_workers=max_concurrent_threads) as executor:
                executor.submit(rotate_image_and_save, os.path.join(segment_data_dir, filename), degrees_to_rotate)
    for filename in os.listdir(segment_layers_dir):
        if filename.endswith(".tif"):
            # Use a thread pool executor to speed up the rotation
            with ThreadPoolExecutor(max_workers=max_concurrent_threads) as executor:
                executor.submit(rotate_image_and_save, os.path.join(segment_layers_dir, filename), degrees_to_rotate)
    rotate_image_and_save(segment_label_path, degrees_to_rotate)

    # Write theta in img_dir in a txt file called rotated.txt
    with open(rotation_result_txt_path, "w") as f:
        f.write(str(degrees_to_rotate))

    print(f"[DONE] Rotating images in {segment_data_dir}")


def rotate_all_images_in_data_dir(data_base_dir):
    segment_paths_dict = get_segment_id_paths_dict(data_base_dir)
    rotations_path = os.path.join("", "rotations")
    label_path = os.path.join("", "labels")
    with ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
        for segment_id in segment_paths_dict:
            executor.submit(rotate_all_images_in_segment, segment_id, segment_paths_dict, rotations_path, label_path)


def restore_all_old_images(segment_paths_dict: Dict[str, str], label_path: str):
    for segment_id in segment_paths_dict:
        segment_data_dir = segment_paths_dict[segment_id]
        segment_layers_dir = os.path.join(segment_data_dir, "layers")
        segment_label_path = os.path.join(label_path, f"{segment_id}_inklabels.png")
        if os.path.exists(segment_data_dir):
            restore_old_images_in_dir(segment_data_dir)
            print(f"Deleting rotated.txt for {segment_data_dir}")
            rotated_txt_path = os.path.join(segment_data_dir, "rotated.txt")
            if os.path.exists(rotated_txt_path):
                os.remove(os.path.join(rotated_txt_path))
        if os.path.exists(segment_layers_dir):
            restore_old_images_in_dir(segment_layers_dir)
        if os.path.exists(segment_label_path):
            restore_old_image(segment_label_path)


if __name__ == "__main__":
    # image = cv2.imread('./data/scroll1_hari/20230827161847/20230827161847.tif')
    # image = cv2.imread('./data/scroll1_hari/20230925002745/20230925002745.tif')
    # cv2.imshow("rotated", respectfully_rotate_image(image, find_rotation_theta(image, show=False)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # rotate_all_images_in_data_dir("data")
    restore_all_old_images(get_segment_id_paths_dict("data"), "labels")
