import os
import cv2
from segment_utils import get_segment_id_paths_dict
from concurrent.futures import ThreadPoolExecutor
import concurrent

num_workers = 64


def get_segment_size(segment_dir):
    print(f"Processing {segment_dir}")
    mask_name_list = list(filter(lambda x: "_mask" in x, os.listdir(segment_dir)))
    if not mask_name_list:
        print(f"Skipping {segment_dir} because no mask was found.")
        return -1
    mask_name = mask_name_list[0]
    mask_path = os.path.join(segment_dir, mask_name)
    img = cv2.imread(mask_path)
    size = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to keep only white regions
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # 250 is an arbitrary threshold, tweak it if necessary

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the size of the white contour(s)
    for contour in contours:
        size += cv2.contourArea(contour)

    return size


def find_all_contour_sizes(data_dir):
    sizes = {}
    segment_paths_dict = get_segment_id_paths_dict(data_dir)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_size_dict = {executor.submit(get_segment_size, segment_paths_dict[segment_id]): segment_id for segment_id
                            in segment_paths_dict}
        for future in concurrent.futures.as_completed(future_size_dict):
            segment_id = future_size_dict[future]
            size = future.result()
            sizes[segment_id] = size
    sorted_sizes = dict(sorted(sizes.items(), key=lambda item: item[1], reverse=True))
    return sorted_sizes


def keep_last_in_sequence(d):
    sorted_keys = sorted(k for k in d.keys() if isinstance(k, int))
    keys_to_remove = set()

    prev_key = None
    for key in sorted_keys:
        if prev_key is not None and key == prev_key + 1:
            keys_to_remove.add(prev_key)
        prev_key = key

    print(f"Keys to remove: {keys_to_remove}")
    for key in keys_to_remove:
        del d[key]

    return d


if __name__ == "__main__":
    val = keep_last_in_sequence(find_all_contour_sizes('./data'))
    print(val)
    print(list(val.keys()))
