import gc
import os
import numpy as np
import cv2
from cfg import CFG
from ink_label_processor import get_ink_text_bounding_boxes
import ray
from tqdm import tqdm
import json
import hashlib
import h5py
import functools
from dataclasses import dataclass
from enum import Enum
import psutil
from ink_label_processor import BoundingBox, window_is_valid

from segment_utils import get_segment_id_paths_dict

os.environ['RAY_record_ref_creation_sites'] = "1"
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = "1"


def get_all_segment_paths(data_base_dir):
    return list(get_segment_id_paths_dict(data_base_dir).values())


def make_symlink(orig_path, new_path):
    if os.path.isdir(orig_path):
        os.symlink(orig_path, new_path, target_is_directory=True)
    else:
        os.symlink(orig_path, new_path)


def normalize_args(func, args, kwargs):
    # Get argument names
    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

    # Combine args and kwargs into a single dictionary
    arguments = dict(zip(arg_names, args))

    # Check each argument, if it's a list, sort it
    for key, value in arguments.items():
        if isinstance(value, list):
            value.sort()  # Sort the list in place
        arguments.update(kwargs)

    # Sort arguments to ensure consistent ordering
    sorted_arguments = sorted(arguments.items())

    # Serialize sorted arguments
    return json.dumps(sorted_arguments, sort_keys=True)


@dataclass
class HDF5CacheResult:
    file_path: str
    results: tuple[str]


def hdf5_cache(*hdf5_dataset_names: str):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a unique hash key based on function arguments
            normalized_args = normalize_args(func, args, kwargs)
            print(f"normalized args {normalized_args}")
            hash_key = hashlib.sha1(normalized_args.encode()).hexdigest()
            h5_filename = f'{hash_key}.h5'
            json_filename = f'{hash_key}.json'
            if not os.path.isdir("./h5"):
                os.mkdir('./h5')
            h5_path = os.path.join("./h5", h5_filename)
            json_path = os.path.join("./h5", json_filename)
            print(f"String expecting {h5_path}")

            # Check if the HDF5 file exists and has the cached result
            if os.path.exists(h5_path):
                print(f"Found cached result for {func.__name__} with hash key {hash_key}")
                return HDF5CacheResult(h5_path, hdf5_dataset_names)

            # Compute the result and store it
            results = func(*args, **kwargs)
            if not isinstance(results, tuple) or len(results) != len(hdf5_dataset_names):
                raise ValueError("Function return value and group names length mismatch")

            with h5py.File(h5_path, 'a') as f:
                for result, group in zip(results, hdf5_dataset_names):
                    f.create_dataset(group, data=result)

            # Write metadata to a JSON file
            if not os.path.exists(json_path):
                with open(json_path, 'w') as json_file:
                    json_file.write(normalized_args)

            return HDF5CacheResult(h5_path, results)

        return wrapper

    return decorator_cache


# ============== Model-related utils =============
def read_segment_image_data(fragment_id, window_size, dilation_horizontal, dilation_vertical, start_idx=15, end_idx=45,
                            sixteen_bit=False):
    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_channels // 2
    end = mid + CFG.in_channels // 2
    idxs = range(start_idx, end_idx)

    for i in idxs:
        if sixteen_bit:
            image = cv2.imread(os.path.join(CFG.train_val_dir, f"{fragment_id}/layers/{i:02}.tif"),
                               cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imread(os.path.join(CFG.train_val_dir, f"{fragment_id}/layers/{i:02}.tif"),
                               cv2.IMREAD_GRAYSCALE)

        pad0 = (window_size - image.shape[0] % window_size)
        pad1 = (window_size - image.shape[1] % window_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5)

        if 'frag' in fragment_id:
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if sixteen_bit:
            image = np.clip(image, 0, 200 * 256)
        else:
            image = np.clip(image, 0, 200)
        images.append(image)
    images = np.stack(images, axis=2)
    ink_label_mask = cv2.imread(os.path.join(CFG.train_val_dir, f"{fragment_id}/{fragment_id}_inklabels.png"), 0)
    fragment_mask = cv2.imread(
        os.path.join(CFG.train_val_dir, f"{fragment_id}/{fragment_id}_mask.png"), 0)

    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1] // 2, fragment_mask.shape[0] // 2),
                                   interpolation=cv2.INTER_AREA)
        ink_label_mask = cv2.resize(ink_label_mask, (ink_label_mask.shape[1] // 2, ink_label_mask.shape[0] // 2),
                                    interpolation=cv2.INTER_AREA)

    ink_bboxes = get_ink_text_bounding_boxes(ink_label_mask, dilation_horizontal=dilation_horizontal,
                                             dilation_vertical=dilation_vertical)
    ink_label_mask = ink_label_mask.astype('float32')
    ink_label_mask /= 255
    print(f"Done reading image {fragment_id}")

    return images, ink_label_mask, fragment_mask, ink_bboxes


@ray.remote(num_returns=6)
def process_data_segment(fragment_id, stride, window_size, training_segment_ids, dilation_horizontal,
                         dilation_vertical, depth, box_contain_threshold):
    depth_delta = depth // 2
    data_image, ink_label_mask, data_mask, ink_bboxes = read_segment_image_data(fragment_id, window_size,
                                                                                dilation_horizontal,
                                                                                dilation_vertical,
                                                                                start_idx=30-depth_delta,
                                                                                end_idx=30+depth_delta)
    height, width = data_image.shape[:2]
    train_images = []
    train_ink_masks = []

    valid_images = []
    valid_ink_masks = []
    valid_xy_xys = []

    print(f"Processing {fragment_id}")
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window_bounding_box: BoundingBox = BoundingBox(x, y, x + window_size, y + window_size)
            if fragment_id in training_segment_ids:
                if window_is_valid(window_img_bbox=window_bounding_box, ink_bounding_boxes=ink_bboxes,
                                   box_contain_threshold=box_contain_threshold):
                    ink_label_data = window_bounding_box.get_img_from_box(ink_label_mask)
                    if ink_label_data.shape != (window_size, window_size):
                        continue
                    train_images.append(window_bounding_box.get_img_from_box(data_image))
                    train_ink_masks.append(ink_label_data[..., None])
            else:
                if window_is_valid(window_img_bbox=window_bounding_box,
                                   ink_bounding_boxes=ink_bboxes, box_contain_threshold=box_contain_threshold):
                    ink_label_data = window_bounding_box.get_img_from_box(ink_label_mask)
                    if ink_label_data.shape != (window_size, window_size):
                        continue
                    valid_images.append(window_bounding_box.get_img_from_box(data_image))
                    valid_ink_masks.append(ink_label_data[..., None])
                    valid_xy_xys.append(window_bounding_box.get_xyxy())
    print(f"Done processing {fragment_id}")
    return (fragment_id, np.array(train_images, copy=True), np.array(train_ink_masks, copy=True),
            np.array(valid_images, copy=True), np.array(valid_ink_masks, copy=True), np.array(valid_xy_xys, copy=True))


class DatasetNames(Enum):
    TRAIN_IMAGES = 'train_images'
    TRAIN_INK_MASKS = 'train_ink_masks'
    VALID_IMAGES = 'valid_images'
    VALID_INK_MASKS = 'valid_ink_masks'
    VALID_XY_XYS = 'valid_xy_xys'
    VALID_SEGMENT_IDXS = 'valid_segment_idxs'


@hdf5_cache(
    DatasetNames.TRAIN_IMAGES.value,
    DatasetNames.TRAIN_INK_MASKS.value,
    DatasetNames.VALID_IMAGES.value,
    DatasetNames.VALID_INK_MASKS.value,
    DatasetNames.VALID_XY_XYS.value,
    DatasetNames.VALID_SEGMENT_IDXS.value
)
def get_train_valid_dataset_with_ray(training_segment_ids, validation_segment_ids, stride, window_size, depth=30,
                                     box_contain_threshold=0.5, dilation_horizontal=220,
                                     dilation_vertical=1):
    object_store_memory_size = int(psutil.virtual_memory().total * 0.5)
    context = ray.init(object_store_memory=object_store_memory_size)  # Initialize Ray
    print(context.dashboard_url)

    validation_segment_idxs = {val_segment_id: i for i, val_segment_id in enumerate(validation_segment_ids)}

    processing_futures_by_segment_id = {
        segment_id: process_data_segment.remote(segment_id, stride, window_size, training_segment_ids,
                                                dilation_horizontal, dilation_vertical, depth, box_contain_threshold)
        for segment_id in training_segment_ids + validation_segment_ids}

    # Collect and aggregate results
    train_images, train_ink_masks, valid_images, valid_ink_masks, valid_xy_xys, valid_segment_idx = [], [], [], [], [], []
    with tqdm(total=len(processing_futures_by_segment_id)) as pbar:
        while processing_futures_by_segment_id:
            done_ids, _ = ray.wait(
                [processing_futures_by_segment_id[segment_id][0] for segment_id in processing_futures_by_segment_id],
                num_returns=1)
            for id_future in done_ids:
                segment_id = ray.get(id_future)
                futures = processing_futures_by_segment_id.pop(segment_id)
                ti = ray.get(futures[1])
                tim = ray.get(futures[2])
                vi = ray.get(futures[3])
                vim = ray.get(futures[4])
                vxys = ray.get(futures[5])
                train_images.extend(ti)
                train_ink_masks.extend(tim)
                valid_images.extend(vi)
                valid_ink_masks.extend(vim)
                valid_xy_xys.extend(vxys)
                valid_segment_idx.extend(np.full(len(valid_xy_xys), validation_segment_idxs.get(segment_id, -1)))
                pbar.update(1)
                print(f"Merged {segment_id} to main process.")
    ray.shutdown()

    return train_images, train_ink_masks, valid_images, valid_ink_masks, valid_xy_xys, valid_segment_idx
